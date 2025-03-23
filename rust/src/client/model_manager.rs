use std::collections::HashMap;
use anyhow::Result;
use tracing;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct ModelManager {
    loaded_models: HashMap<String, LoadedModel>,
    python_module: Arc<Mutex<Option<Py<PyAny>>>>,
    initialized: bool,
}

#[derive(Debug)]
pub struct LoadedModel {
    pub model_type: String,
    pub device: String,
    pub model_cid: String,
}

#[derive(Debug)]
pub struct GenerationResult {
    pub prompt: String,
    pub generation_time_ms: f64,
    pub parameters: HashMap<String, f32>,
    pub timestamp: String,
    pub generated_image: String,
}

#[derive(Debug)]
pub struct CovidResult {
    pub prediction: String,
    pub confidence: String,
    pub probabilities: HashMap<String, String>,
    pub prediction_time_ms: f64,
    pub timestamp: String,
    pub source_type: String,
    pub original_source: String,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            loaded_models: HashMap::new(),
            python_module: Arc::new(Mutex::new(None)),
            initialized: false,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub async fn initialize(&mut self) -> Result<()> {
        let mut module = self.python_module.lock().await;
        
        Python::with_gil(|py| {
            tracing::info!("Loading Python GPU module...");
            
            // First check if CUDA is available in Python
            let torch = match py.import("torch") {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!("Failed to import torch: {}", e);
                    return Err(anyhow::anyhow!("Failed to import torch: {}", e));
                }
            };
            
            let cuda_available: bool = match torch.getattr("cuda") {
                Ok(cuda) => match cuda.getattr("is_available") {
                    Ok(is_available) => match is_available.call0() {
                        Ok(result) => match result.extract() {
                            Ok(available) => available,
                            Err(e) => {
                                tracing::error!("Failed to extract CUDA availability: {}", e);
                                return Err(anyhow::anyhow!("Failed to extract CUDA availability: {}", e));
                            }
                        },
                        Err(e) => {
                            tracing::error!("Failed to call is_available(): {}", e);
                            return Err(anyhow::anyhow!("Failed to call is_available(): {}", e));
                        }
                    },
                    Err(e) => {
                        tracing::error!("Failed to get is_available attribute: {}", e);
                        return Err(anyhow::anyhow!("Failed to get is_available attribute: {}", e));
                    }
                },
                Err(e) => {
                    tracing::error!("Failed to get cuda attribute: {}", e);
                    return Err(anyhow::anyhow!("Failed to get cuda attribute: {}", e));
                }
            };
            
            if !cuda_available {
                tracing::warn!("CUDA is not available in Python environment");
            } else {
                tracing::info!("CUDA is available in Python environment");
                let cuda_version: String = match torch.getattr("version") {
                    Ok(version) => match version.getattr("cuda") {
                        Ok(cuda) => match cuda.extract::<String>() {
                            Ok(version) => version,
                            Err(e) => {
                                tracing::error!("Failed to extract CUDA version: {}", e);
                                return Err(anyhow::anyhow!("Failed to extract CUDA version: {}", e));
                            }
                        },
                        Err(e) => {
                            tracing::error!("Failed to get cuda attribute from version: {}", e);
                            return Err(anyhow::anyhow!("Failed to get cuda attribute from version: {}", e));
                        }
                    },
                    Err(e) => {
                        tracing::error!("Failed to get version attribute: {}", e);
                        return Err(anyhow::anyhow!("Failed to get version attribute: {}", e));
                    }
                };
                tracing::info!("CUDA version: {}", cuda_version);
            }
            
            // Load the Python file
            tracing::info!("Loading Python module from file...");
            
            // // Add the protocol directory to Python's path (parent of current dir)
            // let protocol_dir = std::env::current_dir()?.parent().unwrap().to_path_buf();
            // let protocol_dir_str = protocol_dir.to_str().unwrap().replace('\\', "/");
            
            // // Create a simpler Python path setup
            // let setup_code = format!(
            //     "import sys; sys.path.insert(0, r'{}')",
            //     protocol_dir_str
            // );
            
            // tracing::info!("Adding Python path: {}", setup_code);
            
            // if let Err(e) = py.eval(&setup_code, None, None) {
            //     tracing::error!("Failed to add protocol directory to Python path: {}", e);
            //     return Err(anyhow::anyhow!("Failed to add protocol directory to Python path: {}", e));
            // }
            
            // // Verify the path was added
            // let verify_code = "print(sys.path[0])";
            // if let Err(e) = py.eval(verify_code, None, None) {
            //     tracing::error!("Failed to verify Python path: {}", e);
            //     return Err(anyhow::anyhow!("Failed to verify Python path: {}", e));
            // }
            
            let gpu_load = match PyModule::from_code(
                py,
                include_str!("../../../gpu_loadrust.py"),
                "gpu_loadrust.py",
                "gpu_loadrust",
            ) {
                Ok(module) => {
                    tracing::info!("Successfully loaded Python module");
                    module
                },
                Err(e) => {
                    tracing::error!("Failed to load Python module: {}", e);
                    return Err(anyhow::anyhow!("Failed to load Python module: {}", e));
                }
            };

            // Get the GPUModelLoader class
            tracing::info!("Getting GPUModelLoader class...");
            let gpu_loader_class = match gpu_load.getattr("GPUModelLoader") {
                Ok(class) => {
                    tracing::info!("Successfully got GPUModelLoader class");
                    class
                },
                Err(e) => {
                    tracing::error!("Failed to get GPUModelLoader class: {}", e);
                    return Err(anyhow::anyhow!("Failed to get GPUModelLoader class: {}", e));
                }
            };
            
            // Create a new instance of GPUModelLoader
            tracing::info!("Creating new instance of GPUModelLoader...");
            let loader_instance = match gpu_loader_class.call0() {
                Ok(instance) => {
                    tracing::info!("Successfully created GPUModelLoader instance");
                    instance
                },
                Err(e) => {
                    tracing::error!("Failed to create GPUModelLoader instance: {}", e);
                    return Err(anyhow::anyhow!("Failed to create GPUModelLoader instance: {}", e));
                }
            };
            
            // Verify the instance is valid
            tracing::info!("Verifying GPUModelLoader instance...");
            if let Err(e) = loader_instance.getattr("device") {
                tracing::error!("Failed to verify GPUModelLoader instance: {}", e);
                return Err(anyhow::anyhow!("Failed to verify GPUModelLoader instance: {}", e));
            }
            
            *module = Some(loader_instance.into());
            
            tracing::info!("Python GPU module loaded successfully");
            Ok::<(), anyhow::Error>(())
        })?;

        self.initialized = true;
        Ok(())
    }

    pub async fn get_model(&mut self, model_cid: &str) -> Result<LoadedModel> {
        let module = self.python_module.lock().await;
        
        if module.is_none() {
            return Err(anyhow::anyhow!("Python module not initialized"));
        }

        Python::with_gil(|py| {
            let gpu_loader = module.as_ref().unwrap().as_ref(py);
            
            // Check if model is already loaded
            let status = gpu_loader.call_method0("model_status")?;
            let status_dict = status.downcast::<PyDict>().map_err(|e| anyhow::anyhow!("Failed to downcast status to dict: {}", e))?;
            
            let models_loaded: Vec<String> = status_dict
                .get_item("models_loaded")
                .ok_or_else(|| anyhow::anyhow!("models_loaded key not found in status dict"))?
                .extract()?;
            
            if !models_loaded.contains(&"stable_diffusion".to_string()) {
                tracing::info!("Loading Stable Diffusion model: {}", model_cid);
                let success: bool = gpu_loader
                    .call_method1("load_stable_diffusion", (model_cid,))?
                    .extract()?;
                
                if !success {
                    return Err(anyhow::anyhow!("Failed to load Stable Diffusion model"));
                }
            }

            let device: String = status_dict
                .get_item("device")
                .ok_or_else(|| anyhow::anyhow!("device key not found in status dict"))?
                .extract()?;
            
            Ok(LoadedModel {
                device,
                model_type: "stable_diffusion".to_string(),
                model_cid: model_cid.to_string(),
            })
        })
    }

    pub async fn generate_image(
        &self,
        prompt: &str,
        inference_steps: i32,
        guidance_scale: f32,
    ) -> Result<GenerationResult> {
        let module = self.python_module.lock().await;
        
        Python::with_gil(|py| {
            if let Some(module) = module.as_ref() {
                // Create inference parameters dictionary
                let inference_params = PyDict::new(py);
                inference_params.set_item("num_inference_steps", inference_steps)?;
                inference_params.set_item("guidance_scale", guidance_scale)?;
                
                // Import asyncio
                let asyncio = py.import("asyncio")?;
                
                // Get the coroutine from the async function
                let coro = module.call_method(py, "generate_image", (prompt, inference_params), None)?;
                
                // Create a new event loop for this thread
                let new_loop = asyncio.call_method1("new_event_loop", ())?;
                asyncio.call_method1("set_event_loop", (new_loop,))?;
                
                // Run the coroutine in the new event loop
                let result = match new_loop.call_method1("run_until_complete", (coro,)) {
                    Ok(result) => result,
                    Err(e) => {
                        // Close the event loop before returning error
                        let _ = new_loop.call_method0("close");
                        return Err(anyhow::anyhow!("Failed to run coroutine: {}", e));
                    }
                };
                
                // Close the event loop
                if let Err(e) = new_loop.call_method0("close") {
                    tracing::warn!("Failed to close event loop: {}", e);
                }
                
                // Convert result to dictionary
                let result_dict = result
                    .downcast::<PyDict>()
                    .map_err(|e| anyhow::anyhow!("Failed to downcast result to dict: {}", e))?;
                
                // Extract all fields from the result dictionary
                let prompt: String = result_dict
                    .get_item("prompt")
                    .ok_or_else(|| anyhow::anyhow!("prompt key not found in result"))?
                    .extract()?;
                
                let generation_time_ms: f64 = result_dict
                    .get_item("generation_time_ms")
                    .ok_or_else(|| anyhow::anyhow!("generation_time_ms key not found in result"))?
                    .extract()?;
                
                let timestamp: String = result_dict
                    .get_item("timestamp")
                    .ok_or_else(|| anyhow::anyhow!("timestamp key not found in result"))?
                    .extract()?;
                
                let parameters: HashMap<String, f32> = result_dict
                    .get_item("parameters")
                    .ok_or_else(|| anyhow::anyhow!("parameters key not found in result"))?
                    .extract()?;
                
                let generated_image: String = result_dict
                    .get_item("generated_image")
                    .ok_or_else(|| anyhow::anyhow!("generated_image key not found in result"))?
                    .extract()?;
                
                Ok(GenerationResult {
                    prompt,
                    generation_time_ms,
                    parameters,
                    timestamp,
                    generated_image,
                })
            } else {
                Err(anyhow::anyhow!("Python module not initialized"))
            }
        })
    }

    pub async fn get_device_info(&self) -> Result<String> {
        let module = self.python_module.lock().await;
        
        Python::with_gil(|py| {
            if let Some(module) = module.as_ref() {
                let device_info = module
                    .call_method0(py, "get_device_info")?
                    .extract::<String>(py)?;
                Ok(device_info)
            } else {
                Ok("Not initialized".to_string())
            }
        })
    }

    pub fn unload_model(&mut self, model_cid: &str) -> Result<()> {
        if self.loaded_models.remove(model_cid).is_some() {
            tracing::info!("Model unloaded: {}", model_cid);
        }
        Ok(())
    }

    pub async fn load_covid_model(&self, model_cid: &str) -> Result<()> {
        let module = self.python_module.lock().await;
        
        Python::with_gil(|py| {
            if let Some(module) = module.as_ref() {
                let success: bool = module
                    .call_method1(py, "load_covid_model", (model_cid,))?
                    .extract(py)?;
                
                if !success {
                    return Err(anyhow::anyhow!("Failed to load COVID-19 model"));
                }
                
                Ok(())
            } else {
                Err(anyhow::anyhow!("Python module not initialized"))
            }
        })
    }

    pub async fn process_xray(&self, image_url: &str) -> Result<CovidResult> {
        let module = self.python_module.lock().await;
        
        Python::with_gil(|py| {
            if let Some(module) = module.as_ref() {
                // Import asyncio
                let asyncio = py.import("asyncio")?;
                
                // Get the coroutine from the async function
                let coro = module.call_method(py, "process_xray_async", (image_url,), None)?;
                
                // Create a new event loop for this thread
                let new_loop = asyncio.call_method1("new_event_loop", ())?;
                asyncio.call_method1("set_event_loop", (new_loop,))?;
                
                // Run the coroutine in the new event loop
                let result = match new_loop.call_method1("run_until_complete", (coro,)) {
                    Ok(result) => result,
                    Err(e) => {
                        // Close the event loop before returning error
                        let _ = new_loop.call_method0("close");
                        return Err(anyhow::anyhow!("Failed to run coroutine: {}", e));
                    }
                };
                
                // Close the event loop
                if let Err(e) = new_loop.call_method0("close") {
                    tracing::warn!("Failed to close event loop: {}", e);
                }
                
                // Convert result to dictionary
                let result_dict = result
                    .downcast::<PyDict>()
                    .map_err(|e| anyhow::anyhow!("Failed to downcast result to dict: {}", e))?;
                
                // Extract all fields from the result dictionary
                let prediction: String = result_dict
                    .get_item("prediction")
                    .ok_or_else(|| anyhow::anyhow!("prediction key not found in result"))?
                    .extract()?;
                
                let confidence: String = result_dict
                    .get_item("confidence")
                    .ok_or_else(|| anyhow::anyhow!("confidence key not found in result"))?
                    .extract()?;
                
                let probabilities: HashMap<String, String> = result_dict
                    .get_item("probabilities")
                    .ok_or_else(|| anyhow::anyhow!("probabilities key not found in result"))?
                    .extract()?;
                
                let prediction_time_ms: f64 = result_dict
                    .get_item("prediction_time_ms")
                    .ok_or_else(|| anyhow::anyhow!("prediction_time_ms key not found in result"))?
                    .extract()?;
                
                let timestamp: String = result_dict
                    .get_item("timestamp")
                    .ok_or_else(|| anyhow::anyhow!("timestamp key not found in result"))?
                    .extract()?;
                
                let source_type: String = result_dict
                    .get_item("source_type")
                    .ok_or_else(|| anyhow::anyhow!("source_type key not found in result"))?
                    .extract()?;
                
                let original_source: String = result_dict
                    .get_item("original_source")
                    .ok_or_else(|| anyhow::anyhow!("original_source key not found in result"))?
                    .extract()?;
                
                Ok(CovidResult {
                    prediction,
                    confidence,
                    probabilities,
                    prediction_time_ms,
                    timestamp,
                    source_type,
                    original_source,
                })
            } else {
                Err(anyhow::anyhow!("Python module not initialized"))
            }
        })
    }
} 