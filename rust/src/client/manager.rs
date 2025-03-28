use crate::protocol::types::{ClientCapabilities, GPUClient, GPUInfo, HeartbeatUpdate, PredictionRequest, PredictionResponse, ModelType};
use anyhow::Result;
use chrono::Utc;
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
    routing::post,
    extract::{State, Json},
};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::process::Command;
use std::process::Child;

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
        })
    }

    pub async fn start(&mut self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);
        
        self.register().await?;
        
        // Start the prediction server
        let manager = Arc::new(self.clone());
        let app = Router::new()
            .route("/predict", post(predict))
            .with_state(manager);
            
        let addr = (Ipv4Addr::new(127, 0, 0, 1), self.port);
        tracing::info!("Starting prediction server on {}:{}", addr.0, addr.1);
        
        // Start both the heartbeat loop and the prediction server
        tokio::select! {
            _ = self.heartbeat_loop() => {}
            _ = axum::serve(tokio::net::TcpListener::bind(addr).await?, app) => {}
        }
        
        Ok(())
    }

    async fn register(&self) -> Result<()> {
        let gpu_info = get_gpu_info()?;
        let capabilities = ClientCapabilities {
            models: vec!["stable_diffusion".to_string()],
            gpu_available: gpu_info.is_some(),
        };

        // Use direct IP and port instead of ngrok URL
        let client = GPUClient {
            client_id: self.client_id,
            ip_address: format!("{}:{}", self.ip_addr, self.port),
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
            loaded_models: vec!["stable_diffusion".to_string()],
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
        while self.running.load(Ordering::SeqCst) {
            self.send_heartbeat().await?;
            time::sleep(Duration::from_secs(10)).await;
        }
        Ok(())
    }

    async fn send_heartbeat(&self) -> Result<()> {
        let update = HeartbeatUpdate {
            client_id: self.client_id,
            loaded_models: vec![],
            status: "online".to_string(),
            last_heartbeat: Utc::now(),
            ip_address: Some(format!("{}:{}", self.ip_addr, self.port)),
        };

        self.session
            .post(&format!("{}/heartbeat/{}", self.server_url, self.client_id))
            .json(&update)
            .send()
            .await?;

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

    pub async fn handle_prediction_request(&self, request: PredictionRequest) -> Result<PredictionResponse, Box<dyn std::error::Error>> {
        // Lock the mutex to get access to model_manager
        let mut model_manager = self.model_manager.lock().await;
        
        // Initialize if not already initialized
        if !model_manager.is_initialized() {
            tracing::info!("Initializing Python module...");
            if let Err(e) = model_manager.initialize().await {
                tracing::error!("Failed to initialize Python module: {}", e);
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

        match request.model_type {
            ModelType::CovidXRay => {
                if let Some(image_url) = request.image_url {
                    // Handle COVID X-Ray prediction
                    tracing::info!("Loading COVID X-Ray model...");
                    match model_manager.load_covid_model(&request.model_cid).await {
                        Ok(_) => {
                            tracing::info!("COVID X-Ray model loaded successfully");
                            
                            // Get device info from Python
                            let device_info = match model_manager.get_device_info().await {
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
                            match model_manager.process_xray(&image_url).await {
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
                    let _model = match model_manager.get_model(&request.model_cid).await {
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
                    let result = match model_manager.generate_image(
                        &prompt,
                        inference_steps,
                        guidance_scale,
                    ).await {
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
        }
    }
}

// Move predict function outside impl block
async fn predict(
    State(manager): State<Arc<GPUClientManager>>,
    Json(request): Json<PredictionRequest>,
) -> Json<PredictionResponse> {
    match manager.handle_prediction_request(request).await {
        Ok(response) => Json(response),
        Err(e) => Json(PredictionResponse {
            success: false,
            prompt: None,
            generation_time_ms: None,
            parameters: None,
            timestamp: None,
            image_base64: None,
            error: Some(e.to_string()),
        }),
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