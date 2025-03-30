use crate::protocol::types::{ClientCapabilities, GPUClient, GPUInfo, HeartbeatUpdate, PredictionRequest, PredictionResponse, ModelType};
use anyhow::Result;
use chrono::{Utc, DateTime};
use reqwest::Client;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::time;
use uuid::Uuid;
use crate::client::model_manager::ModelManager;
use tokio::sync::Mutex;
use tracing;
use axum::{
    Router,
    routing::{post, get},
    extract::{State, Json},
    response::IntoResponse,
    handler::Handler,
    body::Body,
    http::Request,
    response::Response,
};
use pyo3::prelude::*;
use std::collections::HashMap;
use serde::Serialize;
use std::process::Child;
use std::process::Command;
use std::error::Error as StdError;

#[derive(Clone)]
pub struct GPUClientManager {
    server_url: String,
    client_id: Uuid,
    ip_addr: IpAddr,
    port: u16,
    session: Client,
    running: Arc<AtomicBool>,
    model_loaded: Arc<AtomicBool>,
    model_manager: Arc<Mutex<ModelManager>>,
    ngrok_process: Arc<Mutex<Option<Child>>>,
    ngrok_url: Arc<Mutex<Option<String>>>,
    status: Arc<Mutex<String>>,
}

impl GPUClientManager {
    pub fn new(server_url: String, port: u16, public_ip: Option<String>) -> Result<Self> {
        let client_id = Uuid::new_v4();
        let ip_addr = get_local_ip(public_ip)?;
        let session = Client::new();

        Ok(Self {
            server_url,
            client_id,
            ip_addr,
            port,
            session,
            running: Arc::new(AtomicBool::new(false)),
            model_loaded: Arc::new(AtomicBool::new(false)),
            model_manager: Arc::new(Mutex::new(ModelManager::new())),
            ngrok_process: Arc::new(Mutex::new(None)),
            ngrok_url: Arc::new(Mutex::new(Some(format!("{}:{}", ip_addr, port)))),
            status: Arc::new(Mutex::new("online".to_string())),
        })
    }

    pub async fn start(&mut self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);
        
        // Register with the server
        self.register().await?;
        
        // Start the prediction server
        let manager = Arc::new(self.clone());
        let client_id = self.client_id;
        let ip_addr = self.ip_addr.to_string();
        let port = self.port;
        
        let app = Router::new()
            .route("/predict", post(predict))
            .route("/health", get(health_check))
            .route("/status", get(move || async move {
                Json(ClientStatus {
                    client_id,
                    ip_address: ip_addr,
                    port,
                    status: "active".to_string(),
                    last_heartbeat: Utc::now(),
                    loaded_models: Vec::new(),
                })
            }))
            .with_state(manager.clone());
            
        let addr = (Ipv4Addr::new(0, 0, 0, 0), self.port);
        tracing::info!("Starting prediction server on {}:{}", addr.0, addr.1);
        
        // Start heartbeat loop in a completely separate high-priority task
        // This uses a dedicated thread with its own tokio runtime to ensure
        // it's not blocked by other operations
        let heartbeat_manager = manager.clone();
        let running = self.running.clone();
        
        // Spawn the heartbeat task in a completely separate thread
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
                
            rt.block_on(async move {
                tracing::info!("Starting heartbeat loop in dedicated thread");
                while running.load(Ordering::SeqCst) {
                    // Send heartbeat with comprehensive error handling
                    match heartbeat_manager.send_heartbeat().await {
                        Ok(_) => {
                            tracing::debug!("Heartbeat sent successfully at {}", Utc::now());
                        }
                        Err(e) => {
                            tracing::error!("Failed to send heartbeat: {}", e);
                        }
                    }
                    
                    // Sleep for exactly 1 second regardless of how long the heartbeat takes
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                tracing::info!("Heartbeat loop stopped");
            });
        });

        // Start the server in the main task
        let server_handle = tokio::spawn(async move {
            match tokio::net::TcpListener::bind(addr).await {
                Ok(listener) => {
                    if let Err(e) = axum::serve(listener, app).await {
                        tracing::error!("Server error: {}", e);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to bind server: {}", e);
                }
            }
        });

        // Wait for the server task to complete
        match server_handle.await {
            Ok(_) => tracing::info!("Server task completed"),
            Err(e) => tracing::error!("Server task failed: {}", e),
        }
        
        Ok(())
    }

    async fn register(&self) -> Result<()> {
        let gpu_info = get_gpu_memory_info()?;
        let model_manager = self.model_manager.lock().await;
        let mut model_cids = HashMap::new();
        
        // Get loaded models and their CIDs
        for (model_type, model) in &model_manager.loaded_models {
            model_cids.insert(model_type.clone(), model.model_cid.clone());
        }

        let loaded_models = model_manager.get_loaded_models();
        let capabilities = ClientCapabilities {
            models: loaded_models.clone(),
            model_cids,
            gpu_available: gpu_info.is_some(),
        };

        // Use direct IP and port instead of ngrok URL
        let client = GPUClient {
            client_id: self.client_id,
            ip_address: format!("{}", self.ip_addr),
            port: self.port,
            gpu_info: gpu_info.unwrap_or_else(|| GPUInfo {
                device_name: "CPU".to_string(),
                total_memory: 0.0,
                allocated_memory: 0.0,
                reserved_memory: 0.0,
                free_memory: 0.0,
                cuda_version: "N/A".to_string(),
                compute_capability: "N/A".to_string(),
            }),
            loaded_models,
            last_heartbeat: Utc::now(),
            status: "online".to_string(),
            capabilities,
        };

        tracing::info!(
            "Registering client {} at {}:{} with server at {}",
            self.client_id,
            client.ip_address,
            self.port,
            self.server_url
        );

        let response = self.session
            .post(&format!("{}/register", self.server_url))
            .json(&client)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            tracing::error!("Failed to register client: {}", error_text);
            return Err(anyhow::anyhow!("Failed to register client: {}", error_text));
        }

        tracing::info!("Successfully registered client");
        Ok(())
    }

    async fn heartbeat_loop(&self) -> Result<()> {
        tracing::info!("Starting heartbeat loop");
        while self.running.load(Ordering::SeqCst) {
            match self.send_heartbeat().await {
                Ok(_) => {
                    tracing::debug!("Heartbeat sent successfully");
                }
                Err(e) => {
                    tracing::error!("Failed to send heartbeat: {}", e);
                    // Don't break the loop on error, just continue after a short delay
                    time::sleep(Duration::from_secs(1)).await;
                }
            }
            time::sleep(Duration::from_secs(1)).await;
        }
        tracing::info!("Heartbeat loop stopped");
        Ok(())
    }

    async fn send_heartbeat(&self) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Get current GPU info first, as it's independent and doesn't require locks
        let gpu_info = match get_gpu_memory_info() {
            Ok(Some(info)) => info,
            Ok(None) => GPUInfo {
                device_name: "CPU".to_string(),
                total_memory: 0.0,
                allocated_memory: 0.0,
                reserved_memory: 0.0,
                free_memory: 0.0,
                cuda_version: "N/A".to_string(),
                compute_capability: "N/A".to_string(),
            },
            Err(e) => {
                tracing::warn!("Failed to get GPU info: {}, using default values", e);
                GPUInfo {
                    device_name: "CPU".to_string(),
                    total_memory: 0.0,
                    allocated_memory: 0.0,
                    reserved_memory: 0.0,
                    free_memory: 0.0,
                    cuda_version: "N/A".to_string(),
                    compute_capability: "N/A".to_string(),
                }
            }
        };

        // Initialize with default values
        let mut loaded_models = Vec::new();
        let mut model_cids = HashMap::new();
        let mut current_status = "online".to_string();
        
        // Try to acquire locks with a timeout to prevent deadlocks
        // Try to get the model manager lock with a short timeout
        let model_manager_lock_result = tokio::time::timeout(
            Duration::from_millis(100), 
            self.model_manager.lock()
        ).await;
        
        if let Ok(model_manager) = model_manager_lock_result {
            // We have the model manager lock, get the data we need
            loaded_models = model_manager.get_loaded_models();
            model_cids = model_manager.loaded_models.iter()
                .map(|(model_type, model)| (model_type.clone(), model.model_cid.clone()))
                .collect();
                
            // Drop the model manager lock immediately
            drop(model_manager);
        } else {
            tracing::warn!("Timeout waiting for model_manager lock during heartbeat, using empty values");
        }
            
        // Try to get the status lock
        let status_lock_result = tokio::time::timeout(
            Duration::from_millis(100),
            self.status.lock()
        ).await;
        
        if let Ok(status) = status_lock_result {
            current_status = status.clone();
            // Drop the status lock immediately
            drop(status);
        } else {
            tracing::warn!("Timeout waiting for status lock during heartbeat, using default status");
        }

        // Create the heartbeat update
        let update = HeartbeatUpdate {
            client_id: self.client_id,
            loaded_models: loaded_models.clone(),
            status: current_status,
            last_heartbeat: Utc::now(),
            ip_address: Some(format!("{}:{}", self.ip_addr, self.port)),
            capabilities: ClientCapabilities {
                models: loaded_models,
                model_cids,
                gpu_available: gpu_info.total_memory > 0.0,
            },
            gpu_info,
        };

        // Send heartbeat to server with timeout and better error handling
        let send_result = tokio::time::timeout(
            Duration::from_secs(3), // Timeout after 3 seconds
            self.session.post(&format!("{}/heartbeat/{}", self.server_url, self.client_id))
                .json(&update)
                .send()
        ).await;
        
        let response = match send_result {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => {
                return Err(anyhow::anyhow!("Failed to send heartbeat: {}", e));
            },
            Err(_) => {
                return Err(anyhow::anyhow!("Heartbeat request timed out after 3 seconds"));
            }
        };

        // Check response status
        if !response.status().is_success() {
            let status = response.status();
            
            // Use timeout for reading the response text as well
            let text_result = tokio::time::timeout(
                Duration::from_secs(1),
                response.text()
            ).await;
            
            let error_text = match text_result {
                Ok(Ok(text)) => text,
                Ok(Err(e)) => format!("Failed to read error response: {}", e),
                Err(_) => "Timed out reading error response".to_string(),
            };
            
            tracing::error!(
                "Server rejected heartbeat with status {}: {}",
                status,
                error_text
            );
            
            return Err(anyhow::anyhow!(
                "Server rejected heartbeat with status {}: {}",
                status,
                error_text
            ));
        }

        let elapsed = start_time.elapsed();
        tracing::debug!(
            "Heartbeat sent successfully for client {} at {}:{} (took {:?})",
            self.client_id,
            self.ip_addr,
            self.port,
            elapsed
        );

        Ok(())
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    fn is_model_loaded(&self) -> bool {
        self.model_loaded.load(Ordering::SeqCst)
    }

    async fn load_model(&self) -> Result<()> {
        // TODO: Implement actual model loading
        self.model_loaded.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn generate_image(
        &self,
        _prompt: &str,
        _inference_steps: u32,
        _guidance_scale: f32,
    ) -> Result<String, Box<dyn std::error::Error>> {




        // TODO: Implement actual image generation
        Ok("base64_encoded_image".to_string())
    }

    pub fn handle_prediction_request(&self, request: PredictionRequest) -> Result<PredictionResponse, Box<dyn StdError + Send + Sync>> {
        // Set Python environment variables before handling the request
        std::env::set_var("PYTHONHOME", self.get_python_home()?);
        std::env::set_var("PYTHONPATH", self.get_python_path()?);

        // Update status to busy
        self.status.lock().await.store("busy".to_string(), Ordering::SeqCst);

        // Initialize if not already initialized
        {
            let mut model_manager = self.model_manager.lock().await;
            if !model_manager.is_initialized() {
                tracing::info!("Initializing Python module...");
                if let Err(e) = model_manager.initialize().await {
                    tracing::error!("Failed to initialize Python module: {}", e);
                    // Set status back to online
                    {
                        let mut status = self.status.lock().await;
                        *status = "online".to_string();
                    }
                    return Ok(PredictionResponse {
                        success: false,
                        prompt: None,
                        generation_time_ms: None,
                        parameters: None,
                        timestamp: None,
                        image_base64: None,
                        error: Some(format!("Failed to initialize Python module: {}", e)),
                    });
                }
            }
        }

        let result = match request.model_type {
            ModelType::CovidXRay => {
                if let Some(image_url) = request.image_url {
                    // Handle COVID X-Ray prediction
                    tracing::info!("Loading COVID X-Ray model...");
                    let model_load_result = {
                        let mut model_manager = self.model_manager.lock().await;
                        model_manager.load_covid_model(&request.model_cid).await
                    };

                    match model_load_result {
                        Ok(_) => {
                            tracing::info!("COVID X-Ray model loaded successfully");
                            
                            // Get device info from Python
                            let device_info = {
                                let model_manager = self.model_manager.lock().await;
                                model_manager.get_device_info().await
                            };

                            let device_info = match device_info {
                                Ok(info) => info,
                                Err(e) => {
                                    tracing::error!("Failed to get device info: {}", e);
                                    return Ok(PredictionResponse {
                                        success: false,
                                        prompt: None,
                                        generation_time_ms: None,
                                        parameters: None,
                                        timestamp: None,
                                        image_base64: None,
                                        error: Some(format!("Failed to get device info: {}", e)),
                                    });
                                }
                            };
                            
                            tracing::info!(
                                "Processing COVID X-Ray image from {} using model on device: {}",
                                image_url,
                                device_info
                            );
                            
                            // Process the X-ray image
                            let process_result = {
                                let model_manager = self.model_manager.lock().await;
                                model_manager.process_xray(&image_url).await
                            };

                            match process_result {
                                Ok(result) => {
                                    // Convert probabilities to parameters format
                                    let mut parameters = HashMap::new();
                                    for (class, prob) in result.probabilities {
                                        if let Ok(value) = prob.trim_end_matches('%').parse::<f32>() {
                                            parameters.insert(class, value);
                                        }
                                    }
                                    
                                    Ok(PredictionResponse {
                                        success: true,
                                        prompt: None,
                                        generation_time_ms: Some(result.prediction_time_ms),
                                        parameters: Some(parameters),
                                        timestamp: Some(result.timestamp),
                                        image_base64: None,
                                        error: None,
                                    })
                                }
                                Err(e) => {
                                    tracing::error!("Failed to process X-ray image: {}", e);
                                    Ok(PredictionResponse {
                                        success: false,
                                        prompt: None,
                                        generation_time_ms: None,
                                        parameters: None,
                                        timestamp: None,
                                        image_base64: None,
                                        error: Some(format!("Failed to process X-ray image: {}", e)),
                                    })
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to load COVID X-Ray model: {}", e);
                            Ok(PredictionResponse {
                                success: false,
                                prompt: None,
                                generation_time_ms: None,
                                parameters: None,
                                timestamp: None,
                                image_base64: None,
                                error: Some(format!("Failed to load model: {}", e)),
                            })
                        }
                    }
                } else {
                    Ok(PredictionResponse {
                        success: false,
                        prompt: None,
                        generation_time_ms: None,
                        parameters: None,
                        timestamp: None,
                        image_base64: None,
                        error: Some("image_url is required for COVID X-Ray model".to_string()),
                    })
                }
            }
            ModelType::StableDiffusion => {
                if let (Some(prompt), Some(quality_preset)) = (request.prompt, request.quality_preset) {
                    let inference_steps = quality_preset.get_inference_steps() as i32;
                    let guidance_scale = quality_preset.get_guidance_scale();
                    
                    tracing::info!(
                        "Generating image with prompt: {}, steps: {}, guidance: {}",
                        prompt, inference_steps, guidance_scale
                    );

                    // Get or load the model
                    tracing::info!("Loading Stable Diffusion model...");
                    let model_load_result = {
                        let mut model_manager = self.model_manager.lock().await;
                        model_manager.get_model(&request.model_cid).await
                    };

                    let _model = match model_load_result {
                        Ok(model) => model,
                        Err(e) => {
                            tracing::error!("Failed to load Stable Diffusion model: {}", e);
                            return Ok(PredictionResponse {
                                success: false,
                                prompt: None,
                                generation_time_ms: None,
                                parameters: None,
                                timestamp: None,
                                image_base64: None,
                                error: Some(format!("Failed to load model: {}", e)),
                            });
                        }
                    };
                    
                    // Generate the image
                    tracing::info!("Generating image...");
                    let generate_result = {
                        let model_manager = self.model_manager.lock().await;
                        model_manager.generate_image(
                            &prompt,
                            inference_steps,
                            guidance_scale,
                        ).await
                    };

                    let result = match generate_result {
                        Ok(result) => result,
                        Err(e) => {
                            tracing::error!("Failed to generate image: {}", e);
                            return Ok(PredictionResponse {
                                success: false,
                                prompt: None,
                                generation_time_ms: None,
                                parameters: None,
                                timestamp: None,
                                image_base64: None,
                                error: Some(format!("Failed to generate image: {}", e)),
                            });
                        }
                    };

                    Ok(PredictionResponse {
                        success: true,
                        prompt: Some(result.prompt),
                        generation_time_ms: Some(result.generation_time_ms),
                        parameters: Some(result.parameters),
                        timestamp: Some(result.timestamp),
                        image_base64: Some(result.generated_image),
                        error: None,
                    })
                } else {
                    Ok(PredictionResponse {
                        success: false,
                        prompt: None,
                        generation_time_ms: None,
                        parameters: None,
                        timestamp: None,
                        image_base64: None,
                        error: Some("prompt and quality_preset are required for Stable Diffusion model".to_string()),
                    })
                }
            }
        };

        // Set status back to online with a short-lived lock
        {
            let mut status = self.status.lock().await;
            *status = "online".to_string();
        }

        result
    }

    async fn start_prediction_server(&self) -> Result<()> {
        let manager = Arc::new(self.clone());
        let client_id = self.client_id;
        let ip_addr = self.ip_addr.to_string();
        let port = self.port;
        
        let app = Router::new()
            .route("/predict", post(predict))
            .route("/health", get(health_check))
            .route("/status", get(move || async move {
                tracing::info!("Received status request");
                Json(ClientStatus {
                    client_id,
                    ip_address: ip_addr,
                    port,
                    status: "active".to_string(),
                    last_heartbeat: Utc::now(),
                    loaded_models: Vec::new(),
                })
            }))
            .with_state(manager);

        // Bind to all interfaces (both IPv4 and IPv6)
        let addr = std::net::SocketAddr::new(
            std::net::IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            self.port
        );
        tracing::info!("Starting prediction server on {}", addr);
        
        // Create a TcpListener with specific socket options
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("Server is listening for connections");
        
        axum::serve(listener, app).await?;

        Ok(())
    }

    fn get_python_home(&self) -> Result<String, Box<dyn StdError>> {
        let exe_dir = std::env::current_exe()?
            .parent()
            .ok_or_else(|| Error::msg("Failed to get executable directory"))?
            .to_path_buf();
        Ok(exe_dir.join("python").to_string_lossy().to_string())
    }

    fn get_python_path(&self) -> Result<String, Box<dyn StdError>> {
        let exe_dir = std::env::current_exe()?
            .parent()
            .ok_or_else(|| Error::msg("Failed to get executable directory"))?
            .to_path_buf();
        Ok(exe_dir.join("python").join("site-packages").to_string_lossy().to_string())
    }
}

// Move predict function outside impl block and add proper trait bounds
#[axum::debug_handler]
async fn predict(
    State(manager): State<Arc<GPUClientManager>>,
    Json(request): Json<PredictionRequest>,
) -> Response {
    match manager.handle_prediction_request(request).await {
        Ok(response) => {
            tracing::info!("Received prediction response from client {}", manager.client_id);
            Json(response).into_response()
        }
        Err(e) => {
            tracing::error!(
                "Error getting prediction from client {}: {}",
                manager.client_id,
                e
            );
            Json(PredictionResponse {
                success: false,
                prompt: None,
                generation_time_ms: None,
                parameters: None,
                timestamp: None,
                image_base64: None,
                error: Some(e.to_string()),
            }).into_response()
        }
    }
}

impl Drop for GPUClientManager {
    fn drop(&mut self) {
        // We can't do async operations in Drop, so we'll just log a warning
        tracing::warn!("GPUClientManager is being dropped. Make sure to call stop() before dropping.");
    }
}

fn get_local_ip(public_ip: Option<String>) -> Result<IpAddr> {
    if let Some(ip) = public_ip {
        Ok(ip.parse()?)
    } else {
        // For now, just return localhost if no public IP is provided
        Ok("127.0.0.1".parse()?)
    }
}

fn get_gpu_info() -> Result<Option<GPUInfo>> {
    Python::with_gil(|py| {
        // Import torch module with better error handling
        let torch = match py.import("torch") {
            Ok(module) => module,
            Err(e) => {
                tracing::warn!("PyTorch is not installed. Please install it using: pip install torch");
                return Ok(Some(GPUInfo {
                    device_name: "CPU".to_string(),
                    total_memory: 0.0,
                    allocated_memory: 0.0,
                    reserved_memory: 0.0,
                    free_memory: 0.0,
                    cuda_version: "N/A".to_string(),
                    compute_capability: "N/A".to_string(),
                }));
            }
        };

        // Check CUDA availability
        let cuda = match torch.getattr("cuda") {
            Ok(cuda) => cuda,
            Err(e) => {
                tracing::error!("Failed to get cuda attribute: {}", e);
                return Ok(None);
            }
        };

        let is_available = match cuda.getattr("is_available") {
            Ok(is_available) => is_available,
            Err(e) => {
                tracing::error!("Failed to get is_available attribute: {}", e);
                return Ok(None);
            }
        };

        let cuda_available: bool = match is_available.call0() {
            Ok(result) => match result.extract() {
                Ok(available) => available,
                Err(e) => {
                    tracing::error!("Failed to extract CUDA availability: {}", e);
                    return Ok(None);
                }
            },
            Err(e) => {
                tracing::error!("Failed to call is_available: {}", e);
                return Ok(None);
            }
        };

        if !cuda_available {
            tracing::info!("CUDA is not available");
            return Ok(None);
        }

        // Get device name
        let get_device_name = match cuda.getattr("get_device_name") {
            Ok(func) => func,
            Err(e) => {
                tracing::error!("Failed to get get_device_name function: {}", e);
                return Ok(None);
            }
        };

        let device_name: String = match get_device_name.call1((0,)) {
            Ok(result) => match result.extract() {
                Ok(name) => name,
                Err(e) => {
                    tracing::error!("Failed to extract device name: {}", e);
                    return Ok(None);
                }
            },
            Err(e) => {
                tracing::error!("Failed to call get_device_name: {}", e);
                return Ok(None);
            }
        };

        // Get device properties
        let get_device_properties = match cuda.getattr("get_device_properties") {
            Ok(func) => func,
            Err(e) => {
                tracing::error!("Failed to get get_device_properties function: {}", e);
                return Ok(None);
            }
        };

        let device_props = match get_device_properties.call1((0,)) {
            Ok(props) => props,
            Err(e) => {
                tracing::error!("Failed to call get_device_properties: {}", e);
                return Ok(None);
            }
        };

        let total_memory: f64 = match device_props.getattr("total_memory") {
            Ok(memory) => match memory.extract() {
                Ok(mem) => mem,
                Err(e) => {
                    tracing::error!("Failed to extract total memory: {}", e);
                    return Ok(None);
                }
            },
            Err(e) => {
                tracing::error!("Failed to get total_memory attribute: {}", e);
                return Ok(None);
            }
        };

        // Get memory info
        let memory_allocated = match cuda.getattr("memory_allocated") {
            Ok(func) => match func.call1((0,)) {
                Ok(result) => match result.extract() {
                    Ok(mem) => mem,
                    Err(e) => {
                        tracing::error!("Failed to extract allocated memory: {}", e);
                        return Ok(None);
                    }
                },
                Err(e) => {
                    tracing::error!("Failed to call memory_allocated: {}", e);
                    return Ok(None);
                }
            },
            Err(e) => {
                tracing::error!("Failed to get memory_allocated function: {}", e);
                return Ok(None);
            }
        };

        let memory_reserved = match cuda.getattr("memory_reserved") {
            Ok(func) => match func.call1((0,)) {
                Ok(result) => match result.extract() {
                    Ok(mem) => mem,
                    Err(e) => {
                        tracing::error!("Failed to extract reserved memory: {}", e);
                        return Ok(None);
                    }
                },
                Err(e) => {
                    tracing::error!("Failed to call memory_reserved: {}", e);
                    return Ok(None);
                }
            },
            Err(e) => {
                tracing::error!("Failed to get memory_reserved function: {}", e);
                return Ok(None);
            }
        };

        // Get CUDA version
        let version = match torch.getattr("version") {
            Ok(version) => match version.getattr("cuda") {
                Ok(cuda_version) => match cuda_version.extract::<String>() {
                    Ok(ver) => ver,
                    Err(e) => {
                        tracing::error!("Failed to extract CUDA version: {}", e);
                        return Ok(None);
                    }
                },
                Err(e) => {
                    tracing::error!("Failed to get cuda version attribute: {}", e);
                    return Ok(None);
                }
            },
            Err(e) => {
                tracing::error!("Failed to get version attribute: {}", e);
                return Ok(None);
            }
        };
        
        let info = GPUInfo {
            device_name: device_name.clone(),
            total_memory,
            allocated_memory: memory_allocated,
            reserved_memory: memory_reserved,
            free_memory: total_memory - memory_allocated,
            cuda_version: version,
            compute_capability: "N/A".to_string(),
        };
        
        tracing::info!("Successfully got GPU info for device: {}", device_name);
        Ok(Some(info))
    })
}

fn get_gpu_memory_info() -> Result<Option<GPUInfo>> {
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.total,memory.used,memory.free,name", "--format=csv,nounits,noheader"])
        .output()?;
    
    if !output.status.success() {
        tracing::warn!("Failed to get GPU info from nvidia-smi");
        return Ok(None);
    }
    
    let output_str = String::from_utf8(output.stdout)?;
    let lines: Vec<&str> = output_str.trim().split('\n').collect();
    
    if lines.is_empty() {
        tracing::warn!("No GPU found");
        return Ok(None);
    }
    
    // Get first GPU info
    let line = lines[0];
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    
    if parts.len() != 4 {
        tracing::warn!("Unexpected nvidia-smi output format");
        return Ok(None);
    }
    
    let total_memory = parts[0].parse::<f64>()?;
    let used_memory = parts[1].parse::<f64>()?;
    let free_memory = parts[2].parse::<f64>()?;
    let device_name = parts[3].to_string();
    
    Ok(Some(GPUInfo {
        device_name,
        total_memory,
        allocated_memory: used_memory,
        reserved_memory: 0.0,  // nvidia-smi doesn't provide this
        free_memory,
        cuda_version: "N/A".to_string(),  // We'll keep this from PyTorch
        compute_capability: "N/A".to_string(),
    }))
}

fn check_gpu_available() -> bool {
    Python::with_gil(|py| {
        if let Ok(torch) = py.import("torch") {
            if let Ok(cuda) = torch.getattr("cuda") {
                if let Ok(is_available) = cuda.getattr("is_available") {
                    if let Ok(result) = is_available.call0() {
                        if let Ok(available) = result.extract::<bool>() {
                            return available;
                        }
                    }
                }
            }
        }
        false
    })
}

async fn fetch_public_ip() -> Result<String> {
    let client = Client::new();
    let response = client.get("https://api.ipify.org")
        .send()
        .await?;
    let ip = response.text().await?;
    Ok(ip)
}

async fn health_check() -> &'static str {
    "OK"
}

#[derive(Serialize)]
struct ClientStatus {
    client_id: Uuid,
    ip_address: String,
    port: u16,
    status: String,
    last_heartbeat: DateTime<Utc>,
    loaded_models: Vec<String>,
} 