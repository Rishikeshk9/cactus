from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import logging
import aiohttp
import threading
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GPUClient(BaseModel):
    client_id: str
    ip_address: str
    port: int
    gpu_info: Dict
    loaded_models: List[str]
    last_heartbeat: str
    status: str
    capabilities: Dict

    def get_last_heartbeat(self) -> datetime:
        return datetime.fromisoformat(self.last_heartbeat)

class PredictionRequest(BaseModel):
    model_type: str
    model_cid: str
    prompt: str
    inference_steps: int = 30
    guidance_scale: float = 7.5
    callback_url: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded image data
    image_url: Optional[str] = None    # URL of the image

class ClientRegistry:
    def __init__(self):
        self.clients: Dict[str, GPUClient] = {}
        self.heartbeat_timeout = 3  # seconds
        self._lock = asyncio.Lock()  # Lock for client registration/updates
        self._search_lock = asyncio.Lock()  # Separate lock for client search
        self._cleanup_lock = asyncio.Lock()  # Lock for cleanup operations
        logger.info("Initialized ClientRegistry")

    async def register_client(self, client: GPUClient):
        async with self._lock:
            logger.info(f"Registering new client: {client.client_id}")
            logger.debug(f"Client details: {client.dict()}")
            
            # If client already exists, update its information
            if client.client_id in self.clients:
                logger.info(f"Client {client.client_id} already exists, updating information")
                existing_client = self.clients[client.client_id]
                for key, value in client.dict().items():
                    setattr(existing_client, key, value)
                self.clients[client.client_id] = existing_client
            else:
                self.clients[client.client_id] = client
                
            logger.info(f"Successfully registered/updated client: {client.client_id} at {client.ip_address}:{client.port}")
            logger.info(f"Total clients: {len(self.clients)}")

    async def update_client(self, client_id: str, update_data: Dict) -> bool:
        try:
            async with self._lock:
                logger.info(f"Updating client: {client_id}")
                logger.debug(f"Update data: {update_data}")
                
                if client_id in self.clients:
                    client = self.clients[client_id]
                    for key, value in update_data.items():
                        setattr(client, key, value)
                    logger.info(f"Successfully updated client: {client_id}")
                    return True
                else:
                    # Create new client from update data
                    new_client = GPUClient(
                        client_id=client_id,
                        ip_address=update_data.get("ip_address", "unknown"),
                        port=update_data.get("port", 8000),
                        gpu_info=update_data.get("gpu_info", {}),
                        loaded_models=update_data.get("loaded_models", []),
                        last_heartbeat=update_data.get("last_heartbeat", datetime.now().isoformat()),
                        status=update_data.get("status", "active"),
                        capabilities=update_data.get("capabilities", {})
                    )
                    await self.register_client(new_client)
                    return True

        except Exception as e:
            logger.error(f"Error updating client: {str(e)}")
            return False

    async def remove_client(self, client_id: str):
        async with self._cleanup_lock:
            logger.info(f"Removing client: {client_id}")
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Successfully removed client: {client_id}")
                logger.info(f"Remaining clients: {len(self.clients)}")
            else:
                logger.warning(f"Client not found for removal: {client_id}")

    async def get_active_clients(self) -> List[GPUClient]:
        """Get list of active clients without modifying the registry"""
        current_time = datetime.now()
        active_clients = []
        
        # Create a copy of the clients dictionary to avoid modification during iteration
        clients_copy = dict(self.clients)
        
        for client_id, client in clients_copy.items():
            try:
                time_diff = (current_time - client.get_last_heartbeat()).seconds
                if time_diff < self.heartbeat_timeout:
                    active_clients.append(client)
            except Exception as e:
                logger.error(f"Error processing client {client_id}: {str(e)}")
        
        return active_clients

    async def cleanup_inactive_clients(self):
        """Separate method to clean up inactive clients"""
        async with self._cleanup_lock:
            current_time = datetime.now()
            clients_to_remove = []
            
            # Create a copy of the clients dictionary
            clients_copy = dict(self.clients)
            
            for client_id, client in clients_copy.items():
                try:
                    time_diff = (current_time - client.get_last_heartbeat()).seconds
                    if time_diff >= self.heartbeat_timeout:
                        clients_to_remove.append(client_id)
                        logger.info(f"Marking client {client_id} for removal due to timeout")
                except Exception as e:
                    logger.error(f"Error checking client {client_id} for cleanup: {str(e)}")
                    clients_to_remove.append(client_id)
            
            # Remove inactive clients
            for client_id in clients_to_remove:
                await self.remove_client(client_id)
            
            logger.info(f"Cleanup completed. Removed {len(clients_to_remove)} inactive clients")

    async def get_client_by_id(self, client_id: str) -> Optional[GPUClient]:
        async with self._lock:
            if client_id in self.clients:
                logger.info(f"Found client: {client_id}")
                return self.clients[client_id]
            logger.warning(f"Client not found: {client_id}")
            return None

    async def find_best_client(self, model_type: str) -> Optional[GPUClient]:
        """Find the best available client for the requested model type"""
        try:
            async with self._search_lock:
                logger.info(f"Starting client search for model: {model_type}")
                active_clients = await self.get_active_clients()
                logger.info(f"Found {len(active_clients)} active clients")
                
                if not active_clients:
                    logger.warning("No active clients found")
                    return None
                
                # First try to find a client that already has the model loaded
                for client in active_clients:
                    logger.debug(f"Checking client {client.client_id} with models: {client.loaded_models}")
                    if model_type in client.loaded_models:
                        logger.info(f"Found client {client.client_id} with model {model_type} already loaded")
                        return client
                
                # If no client has the model loaded, find any client with GPU capabilities
                for client in active_clients:
                    if client.status == "active":
                        # Check if client has GPU capabilities
                        gpu_info = client.gpu_info
                        if gpu_info and gpu_info.get("device_name") and gpu_info.get("total_memory", 0) > 0:
                            logger.info(f"Selected client {client.client_id} with GPU: {gpu_info.get('device_name')}")
                            return client
                
                logger.warning("No suitable client with GPU capabilities found")
                return None
                
        except Exception as e:
            logger.error(f"Error in find_best_client: {str(e)}")
            return None

    async def print_clients_table(self):
        """Print a formatted table of all connected clients"""
        async with self._lock:
            current_time = datetime.now()
            logger.info("\n=== Connected Clients Table ===")
            logger.info(f"Total Clients: {len(self.clients)}")
            logger.info(f"Current Time: {current_time}")
            logger.info("-" * 100)
            logger.info(f"{'Client ID':<36} {'IP Address':<15} {'Port':<6} {'Status':<8} {'Last Heartbeat':<20} {'Models':<30}")
            logger.info("-" * 100)
            
            for client_id, client in self.clients.items():
                try:
                    time_diff = (current_time - client.get_last_heartbeat()).seconds
                    status = "active" if time_diff < self.heartbeat_timeout else "inactive"
                    models_str = ", ".join(client.loaded_models[:2]) + ("..." if len(client.loaded_models) > 2 else "")
                    logger.info(f"{client_id:<36} {client.ip_address:<15} {client.port:<6} {status:<8} {client.last_heartbeat:<20} {models_str:<30}")
                except Exception as e:
                    logger.error(f"Error formatting client {client_id}: {str(e)}")
            logger.info("-" * 100 + "\n")

# Create global registry
registry = ClientRegistry()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received {request.method} request to {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/register")
async def register_client(client: GPUClient):
    logger.info(f"Received registration request from client: {client.client_id}")
    try:
        await registry.register_client(client)
        return {"status": "success", "message": f"Client {client.client_id} registered"}
    except Exception as e:
        logger.error(f"Error registering client: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/heartbeat/{client_id}")
async def client_heartbeat(client_id: str, update_data: Dict, request: Request):
    logger.info(f"Received heartbeat from client: {client_id}")
    try:
        # Add client_id to update_data if not present
        if "client_id" not in update_data:
            update_data["client_id"] = client_id
            
        # Ensure we have a last_heartbeat timestamp
        if "last_heartbeat" not in update_data:
            update_data["last_heartbeat"] = datetime.now().isoformat()
            
        # Add client IP and port from request
        client_host = request.client.host if request.client else "unknown"
        client_port = request.client.port if request.client else 8000
        
        # Add required fields if missing
        update_data.setdefault("ip_address", client_host)
        update_data.setdefault("port", client_port)
        update_data.setdefault("gpu_info", {})
        update_data.setdefault("capabilities", {})
        
        # Try to update the client
        success = await registry.update_client(client_id, update_data)
        if success:
            # Print the updated clients table
            await registry.print_clients_table()
            return {"status": "success", "message": "Heartbeat received"}
        
        # If update failed, try to register the client
        try:
            new_client = GPUClient(
                client_id=client_id,
                ip_address=client_host,
                port=client_port,
                gpu_info=update_data.get("gpu_info", {}),
                loaded_models=update_data.get("loaded_models", []),
                last_heartbeat=update_data.get("last_heartbeat", datetime.now().isoformat()),
                status=update_data.get("status", "active"),
                capabilities=update_data.get("capabilities", {})
            )
            await registry.register_client(new_client)
            # Print the updated clients table after registration
            await registry.print_clients_table()
            return {"status": "success", "message": "Client re-registered"}
        except Exception as e:
            logger.error(f"Error re-registering client: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        logger.error(f"Error processing heartbeat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clients")
async def get_clients():
    logger.info("Received request for client list")
    try:
        # First get the list of active clients
        active_clients = await registry.get_active_clients()
        logger.info(f"Returning {len(active_clients)} active clients")
        
        # Then run cleanup in the background
        asyncio.create_task(registry.cleanup_inactive_clients())
        
        return {
            "active_clients": active_clients,
            "total_clients": len(registry.clients)
        }
    except Exception as e:
        logger.error(f"Error getting clients: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clients/{client_id}")
async def get_client(client_id: str):
    logger.info(f"Received request for client: {client_id}")
    try:
        client = await registry.get_client_by_id(client_id)
        if client:
            return client
        raise HTTPException(status_code=404, detail="Client not found")
    except Exception as e:
        logger.error(f"Error getting client: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clients/{client_id}")
async def remove_client(client_id: str):
    logger.info(f"Received request to remove client: {client_id}")
    try:
        await registry.remove_client(client_id)
        return {"status": "success", "message": f"Client {client_id} removed"}
    except Exception as e:
        logger.error(f"Error removing client: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: Request):
    """Handle prediction requests and route them to appropriate clients"""
    try:
        # Get raw request data
        request_data = await request.json()
        logger.info(f"Received prediction request data: {request_data}")
        
        # Extract required fields with defaults
        model_type = request_data.get("model_type")
        if not model_type:
            raise HTTPException(status_code=400, detail="model_type is required")
            
        model_cid = request_data.get("model_cid", "")
        prompt = request_data.get("prompt", "")
        inference_steps = request_data.get("inference_steps", 30)
        guidance_scale = request_data.get("guidance_scale", 7.5)
        callback_url = request_data.get("callback_url")
        image_data = request_data.get("image_data")
        image_url = request_data.get("image_url")
        
        logger.info(f"Processing request for model: {model_type}")
        
        # Find the best available client with timeout
        try:
            client = await asyncio.wait_for(registry.find_best_client(model_type), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("Timeout while finding best client")
            raise HTTPException(status_code=503, detail="Timeout while finding suitable client")
            
        if not client:
            logger.error("No suitable client found")
            raise HTTPException(status_code=503, detail="No suitable client available")
        
        # Forward the request to the selected client
        client_url = f"http://{client.ip_address}:{client.port}/predict"
        logger.info(f"Forwarding request to client: {client_url}")
        
        # Prepare the request data
        forward_data = {
            "model_type": model_type,
            "model_cid": model_cid,
            "prompt": prompt,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "callback_url": callback_url,
            "image_data": image_data,
            "image_url": image_url
        }
        
        # If we have image data, ensure it's properly formatted
        if forward_data.get("image_data"):
            # Ensure the base64 data is properly formatted
            if not forward_data["image_data"].startswith("data:image/"):
                forward_data["image_data"] = f"data:image/jpeg;base64,{forward_data['image_data']}"
        
        logger.info(f"Forwarding data to client: {forward_data}")
        
        # Add timeout to the client request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(client_url, json=forward_data, timeout=30.0) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Successfully received response from client {client.client_id}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Client returned error: {response.status} - {error_text}")
                        # If client returns error, try to find another client
                        logger.info("Attempting to find another client...")
                        client = await registry.find_best_client(model_type)
                        if client and client.client_id != client.client_id:
                            return await predict(request)  # Retry with new client
                        raise HTTPException(status_code=response.status, detail=error_text)
            except asyncio.TimeoutError:
                logger.error(f"Timeout while waiting for client {client.client_id} response")
                raise HTTPException(status_code=504, detail="Client request timeout")
                
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON in request")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # If request fails, try to find another client
        logger.info("Attempting to find another client...")
        client = await registry.find_best_client(model_type)
        if client and client.client_id != client.client_id:
            return await predict(request)  # Retry with new client
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug") 