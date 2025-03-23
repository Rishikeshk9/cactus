print(">>> gpu_loadrust.py Loading...")
import sys
import os

# Add the directory of this file (i.e., protocol/) to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os
import torch
import requests
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from datetime import datetime
import json
from io import BytesIO
import torchvision.models as models
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import aiohttp
from typing import Optional, Literal
from diffusers import StableDiffusionPipeline
import base64
from fastapi.middleware.cors import CORSMiddleware
import argparse
from pathlib import Path


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your Next.js app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    model_cid: str
    model_type: Literal["covid_xray", "stable_diffusion"]
    image_url: Optional[str] = None
    prompt: Optional[str] = None
    callback_url: Optional[str] = None
    # Stable Diffusion parameters
    num_inference_steps: Optional[int] = 50  # Default value, lower = faster but lower quality
    guidance_scale: Optional[float] = 7.5    # Default value, controls how closely to follow the prompt
    quality_preset: Optional[Literal["fast", "balanced", "quality"]] = "balanced"
    
    class Config:
        use_enum_values = True

    @property
    def get_inference_params(self):
        """Get inference parameters based on quality preset or custom values"""
        if self.quality_preset == "fast":
            return {
                "num_inference_steps": self.num_inference_steps or 20,
                "guidance_scale": self.guidance_scale or 7.0
            }
        elif self.quality_preset == "quality":
            return {
                "num_inference_steps": self.num_inference_steps or 100,
                "guidance_scale": self.guidance_scale or 8.5
            }
        else:  # balanced
            return {
                "num_inference_steps": self.num_inference_steps or 50,
                "guidance_scale": self.guidance_scale or 7.5
            }

class ModelConfig:
    def __init__(self, model_type: str, config: dict):
        self.model_type = model_type
        self.config = config
        
    @property
    def input_size(self):
        return self.config.get("input_size", [3, 224, 224])
    
    @property
    def output_classes(self):
        return self.config.get("output_classes", 3)
    
    @property
    def model_id(self):
        return self.config.get("model_id", "")

class CovidXrayModel(nn.Module):
    def __init__(self):
        super(CovidXrayModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 26 * 26, 512)
        self.fc2 = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 26 * 26)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class GPUModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUModelLoader, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.models = {}
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.port = 8000  # Default port
            self.client_manager = None
            self.initialized = True
            self.model_weights_dir = Path("model_weights")
            self.models_config_dir = Path("models_config")
            self.model_weights_dir.mkdir(exist_ok=True)
            self.models_config_dir.mkdir(exist_ok=True)
            
            # Define COVID-19 X-ray image transforms
            self.covid_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])  # Single channel normalization
            ])
            
            # Define COVID-19 classes
            self.covid_classes = ["normal", "pneumonia", "covid-19"]

    def get_device_info(self):
        """Get information about the current device."""
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        return "CPU"

    def get_loaded_models(self):
        """Get list of currently loaded models"""
        return list(self.models.keys())

    def model_status(self):
        """Get status of loaded models and device"""
        return {
            "models_loaded": list(self.models.keys()),
            "device": str(self.device),
            "device_type": self.get_device_info()
        }

    def load_model_config(self, model_type: str) -> ModelConfig:
        """Load model configuration from JSON file"""
        try:
            config_path = os.path.join("models_config", "models.json")
            with open(config_path, 'r') as f:
                configs = json.load(f)
                if model_type not in configs:
                    raise ValueError(f"Configuration for model type {model_type} not found")
                return ModelConfig(model_type, configs[model_type])
        except Exception as e:
            print(f"Error loading model config: {str(e)}")
            raise

    def load_stable_diffusion(self, model_cid: str) -> bool:
        """Load Stable Diffusion model from local files or download from Hugging Face"""
        try:
            config = self.load_model_config("stable_diffusion")
            model_dir = os.path.join("model_weights", "stable_diffusion")
            os.makedirs(model_dir, exist_ok=True)
            
            # If model_cid starts with "hf:", get the actual model ID
            model_id = model_cid[3:] if model_cid.startswith("hf:") else model_cid
            
            # Check current loaded model
            current_model = self.models.get('stable_diffusion')
            
            # Function to get model path in cache
            def get_cached_model_path(model_id):
                # Convert model ID to cache directory format (replace / with --)
                cache_dir = model_id.replace('/', '--')
                return os.path.join(model_dir, "models--" + cache_dir)
            
            # Get the cache path for requested model
            model_cache_path = get_cached_model_path(model_id)
            
            # Check if we need to switch models
            if current_model is not None:
                current_model_id = getattr(current_model.config, '_name_or_path', None)
                if current_model_id == model_id:
                    print(f"Model {model_id} is already loaded in memory")
                    return True
                else:
                    print(f"Unloading current model {current_model_id} to load {model_id}")
                    self.unload_models('stable_diffusion')
            
            print(f"Loading/Downloading model from Hugging Face: {model_id}")
            
            # Check if model exists in cache
            if os.path.exists(model_cache_path):
                print(f"Found model in cache: {model_cache_path}")
            else:
                print(f"Model not found in cache, downloading: {model_id}")
            
            try:
                # Special handling for different model types
                pipe = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=self.torch_dtype,
                        use_safetensors=True,
                        cache_dir=model_dir
                    )
                    
                
                # Move model to device and enable optimizations
                pipe = pipe.to(self.device)
                if self.device == "cuda":
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                    
                    # Additional memory optimizations for large models
                    if "xl" in model_id.lower():
                        print("Enabling additional optimizations for XL model...")
                        pipe.enable_model_cpu_offload()
                
                self.models['stable_diffusion'] = pipe
                print(f"Stable Diffusion model {model_id} loaded successfully")
                return True
                
            except Exception as load_error:
                print(f"Error during model loading: {str(load_error)}")
                # Clean up any partially downloaded files
                if os.path.exists(model_cache_path):
                    import shutil
                    print(f"Cleaning up failed model download: {model_cache_path}")
                    shutil.rmtree(model_cache_path, ignore_errors=True)
                raise
            
        except Exception as e:
            print(f"Error loading Stable Diffusion model: {str(e)}")
            return False

    async def generate_image(self, prompt: str, inference_params: dict) -> dict:
        """Generate image using Stable Diffusion"""
        try:
            if 'stable_diffusion' not in self.models:
                raise Exception("Stable Diffusion model not loaded")

            # Start timing
            start_time = datetime.now()
            print(f"Image Generation Started")
            # Generate image with specified parameters
            pipe = self.models['stable_diffusion']
            
            # Safety check for pipeline
            if not hasattr(pipe, '__call__'):
                raise Exception("Invalid pipeline object")
                
            try:
                # Convert device to string for torch.autocast
                device_str = str(self.device)
                with torch.autocast(device_str):
                    output = pipe(
                            prompt,
                            num_inference_steps=inference_params["num_inference_steps"],
                            guidance_scale=inference_params["guidance_scale"]
                        )    
                    
                    # Handle different output formats
                    if output is None:
                        raise Exception("Pipeline returned None")
                    elif isinstance(output, dict) and "images" in output:
                        image = output["images"][0]
                    elif hasattr(output, "images"):
                        image = output.images[0]
                    else:
                        raise Exception(f"Unexpected pipeline output format: {type(output)}")
                    
            except Exception as pipe_error:
                print(f"Pipeline error: {str(pipe_error)}. Attempting recovery...")
                 
                self.unload_models('stable_diffusion')
                torch.cuda.empty_cache()  # Clear CUDA cache
                raise Exception(f"Pipeline error: {str(pipe_error)}")
            
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds() * 1000

            # Convert PIL image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            # Print the base64 image string to logs (truncated for readability)
            if len(img_str) > 100:
                print(f"Generated image base64 (truncated): {img_str[:50]}...{img_str[-50:]}")
            else:
                print(f"Generated image base64: {img_str}")
            result = {
                "generated_image": img_str,
                "prompt": prompt,
                "generation_time_ms": round(generation_time, 2),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "parameters": {
                    "num_inference_steps": inference_params["num_inference_steps"],
                    "guidance_scale": inference_params["guidance_scale"]
                }
            }

            return result

        except Exception as e:
            print(f"Error generating image: {str(e)}")
            # Re-raise the exception with more details
            raise Exception(f"Image generation failed: {str(e)}")

    def download_model_from_pinata(self, cid, model_type):
        """Download model weights from Pinata using CID"""
        try:
            # Pinata Gateway URL
            gateway_url = f"https://green-partial-ermine-438.mypinata.cloud/ipfs/{cid}"
            
            # Create model type specific directory
            model_dir = os.path.join("model_weights", model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model weights locally
            model_path = os.path.join(model_dir, f"model_{cid}.pth")
            
            if not os.path.exists(model_path):
                print(f"Downloading model from {gateway_url}")
                # Download the file
                response = requests.get(gateway_url, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            return model_path
        except Exception as e:
            print(f"Error downloading model from Pinata: {str(e)}")
            return None

    def load_covid_model(self, model_cid):
        """Load COVID-19 X-ray classification model"""
        try:
           
            model = models.densenet121(pretrained=True)

            # Download and load weights
            weights_path = self.download_model_from_pinata(model_cid, "covid_xray")
            if not weights_path:
                raise Exception("Failed to download model weights from Pinata")
            
            config_path = os.path.join("models_config", "models.json")
            with open(config_path, 'r') as f:
                model_config = json.load(f)["xray"]

             # Adapt input layer for grayscale if needed
            if model_config["input_size"][0] == 1:
                # Replace first conv layer to accept single channel
                model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Modify classifier for the number of classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, model_config["output_classes"])
            
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            print("Model loaded successfully")
            self.models['covid_xray'] = model
            return True
        except Exception as e:
            print(f"Error loading COVID-19 model: {str(e)}")
            return False

    def load_image_from_url(self, image_url):
        """Load image from URL"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error loading image from URL: {str(e)}")
            return None

    def process_xray(self, image_source):
        """
        Process X-ray image for COVID-19 detection
        Args:
            image_source: Can be either a local file path or a URL
        """
        try:
            if 'covid_xray' not in self.models:
                raise Exception("COVID-19 model not loaded")

            # Load and preprocess the image
            if image_source.startswith(('http://', 'https://')):
                image = self.load_image_from_url(image_source)
                if image is None:
                    raise Exception("Failed to load image from URL")
            else:
                if not os.path.exists(image_source):
                    raise Exception(f"Image file not found: {image_source}")
                image = Image.open(image_source)

            # Convert to grayscale
            image = image.convert('L')

            # Preprocess image
            input_tensor = self.covid_transforms(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # Get prediction with timing
            start_time = datetime.now()
            with torch.no_grad():
                output = self.models['covid_xray'](input_batch)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            end_time = datetime.now()
            prediction_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds

            # Generate result metadata
            result = {
                "prediction": self.covid_classes[predicted_class],
                "confidence": f"{confidence * 100:.2f}%",
                "probabilities": {
                    class_name: f"{prob * 100:.2f}%"
                    for class_name, prob in zip(self.covid_classes, probabilities[0].tolist())
                },
                "prediction_time_ms": round(prediction_time, 2),  # Round to 2 decimal places
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "source_type": "url" if image_source.startswith(('http://', 'https://')) else "local_file",
                "original_source": image_source
            }

            return result

        except Exception as e:
            print(f"Error processing X-ray image: {str(e)}")
            return None

    def unload_models(self, model_type: Optional[str] = None):
        """
        Unload models from GPU memory
        Args:
            model_type: Specific model type to unload. If None, unloads all models.
        """
        try:
            if model_type:
                if model_type in self.models:
                    del self.models[model_type]
                    print(f"Unloaded {model_type} model")
                else:
                    print(f"Model {model_type} not loaded")
            else:
                # Create a list of keys to avoid dictionary modification during iteration
                model_names = list(self.models.keys())
                for model_name in model_names:
                    del self.models[model_name]
                print("Unloaded all models")
            
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"Error unloading models: {str(e)}")
            return False

    async def send_callback(self, callback_url: str, result: dict):
        """Send prediction results to callback URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=result) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Error sending callback: {str(e)}")
            return False

    async def process_xray_async(self, image_source: str, callback_url: Optional[str] = None):
        """
        Asynchronous version of process_xray that supports callbacks
        """
        result = self.process_xray(image_source)
        
        if result and callback_url:
            await self.send_callback(callback_url, result)
        
        return result

# Create singleton instance
gpu_loader = GPUModelLoader()

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        if request.model_type == "stable_diffusion":
            if not request.prompt:
                raise HTTPException(status_code=400, detail="Prompt is required for Stable Diffusion")
                
            # Load model if not already loaded
            if 'stable_diffusion' not in gpu_loader.models:
                success = gpu_loader.load_stable_diffusion(request.model_cid)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to load Stable Diffusion model")
            
            # Get inference parameters and generate image
            inference_params = request.get_inference_params
            result = await gpu_loader.generate_image(request.prompt, inference_params)
            
        else:  # covid_xray
            if not request.image_url:
                raise HTTPException(status_code=400, detail="Image URL is required for X-ray analysis")
                
            # Load model if not already loaded
            if 'covid_xray' not in gpu_loader.models:
                success = gpu_loader.load_covid_model(request.model_cid)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to load model")

            # Process the image
            result = await gpu_loader.process_xray_async(
                request.image_url,
                request.callback_url
            )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to process request")

        # Send callback if provided
        if result and request.callback_url:
            await gpu_loader.send_callback(request.callback_url, result)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def model_status():
    """Check loaded models and their status"""
    return {
        "models_loaded": list(gpu_loader.models.keys()),
        "device": gpu_loader.device,
        "device_type": gpu_loader.get_device_info()
    }

@app.post("/unload")
async def unload_models(model_type: Optional[str] = None):
    """
    Unload models from GPU memory
    Args:
        model_type: Optional model type to unload ("covid_xray" or "stable_diffusion").
                   If not provided, unloads all models.
    """
    try:
        if model_type and model_type not in ["covid_xray", "stable_diffusion"]:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        success = gpu_loader.unload_models(model_type)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to unload models")
        
        return {
            "status": "success",
            "message": f"Successfully unloaded {'all models' if not model_type else model_type}",
            "remaining_models": list(gpu_loader.models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GPU Load Server')
    parser.add_argument('--gateway', type=str, default='localhost',
                      help='IP address of the GPU server (default: localhost)')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to run the GPU load server on (default: 8000)')
    args = parser.parse_args()
    
    # Set the server URL based on the gateway argument
    server_url = f"http://{args.gateway}"
    print(f"Connecting to GPU server at: {server_url}")
    print(f"Starting GPU load server on port: {args.port}")
    
    
