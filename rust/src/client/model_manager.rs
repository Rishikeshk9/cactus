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
            let gpu_load = match PyModule::from_code(
                py,
                include_str!("../../../gpu_load.py"),
                "gpu_load.py",
                "gpu_load",
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
            let models_loaded: Vec<String> = status.getattr("models_loaded")?.extract()?;
            
            if !models_loaded.contains(&"stable_diffusion".to_string()) {
                tracing::info!("Loading Stable Diffusion model: {}", model_cid);
                let success: bool = gpu_loader
                    .call_method1("load_stable_diffusion", (model_cid,))?
                    .extract()?;
                
                if !success {
                    return Err(anyhow::anyhow!("Failed to load Stable Diffusion model"));
                }
            }

            let device: String = status.getattr("device")?.extract()?;
            
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
    ) -> Result<String> {
        let module = self.python_module.lock().await;
        
        Python::with_gil(|py| {
            if let Some(module) = module.as_ref() {
                let kwargs = PyDict::new(py);
                kwargs.set_item("prompt", prompt)?;
                kwargs.set_item("num_inference_steps", inference_steps)?;
                kwargs.set_item("guidance_scale", guidance_scale)?;
                
                let result = module
                    .call_method(py, "generate_image", (), Some(kwargs))?
                    .extract::<String>(py)?;
                Ok(result)
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
} 