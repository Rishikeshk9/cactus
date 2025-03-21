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

class ClientRegistry:
    def __init__(self):
        self.clients: Dict[str, GPUClient] = {}
        self.heartbeat_timeout = 30  # seconds
        self._lock = threading.Lock()  # Add thread lock for synchronization
        logger.info("Initialized ClientRegistry")

    def register_client(self, client: GPUClient):
        with self._lock:
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

    def update_client(self, client_id: str, update_data: Dict):
        with self._lock:
            logger.info(f"Updating client: {client_id}")
            logger.debug(f"Update data: {update_data}")
            
            if client_id in self.clients:
                client = self.clients[client_id]
                for key, value in update_data.items():
                    setattr(client, key, value)
                logger.info(f"Successfully updated client: {client_id}")
                return True
            else:
                # If client not found, try to re-register with the update data
                logger.warning(f"Client {client_id} not found, attempting to re-register")
                try:
                    # Create a new client object with the update data
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
                    self.register_client(new_client)
                    return True
                except Exception as e:
                    logger.error(f"Failed to re-register client {client_id}: {str(e)}")
                    return False

    def remove_client(self, client_id: str):
        with self._lock:
            logger.info(f"Removing client: {client_id}")
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Successfully removed client: {client_id}")
                logger.info(f"Remaining clients: {len(self.clients)}")
            else:
                logger.warning(f"Client not found for removal: {client_id}")

    def get_active_clients(self) -> List[GPUClient]:
        with self._lock:
            current_time = datetime.now()
            active_clients = []
            logger.info(f"Checking active clients at {current_time}")
            
            # Create a copy of the clients dictionary to avoid modification during iteration
            clients_copy = dict(self.clients)
            
            for client_id, client in clients_copy.items():
                time_diff = (current_time - client.get_last_heartbeat()).seconds
                logger.debug(f"Client {client_id} last heartbeat: {time_diff} seconds ago")
                if time_diff < self.heartbeat_timeout:
                    active_clients.append(client)
                else:
                    logger.info(f"Client {client_id} timed out, removing...")
                    self.remove_client(client_id)
            
            logger.info(f"Found {len(active_clients)} active clients")
            return active_clients

    def get_client_by_id(self, client_id: str) -> Optional[GPUClient]:
        with self._lock:
            if client_id in self.clients:
                logger.info(f"Found client: {client_id}")
                return self.clients[client_id]
            logger.warning(f"Client not found: {client_id}")
            return None

    def find_best_client(self, model_type: str) -> Optional[GPUClient]:
        """Find the best available client for the requested model type"""
        with self._lock:
            active_clients = self.get_active_clients()
            logger.info(f"Looking for client with model: {model_type}")
            logger.info(f"Active clients: {[c.client_id for c in active_clients]}")
            
            # First try to find a client that already has the model loaded
            for client in active_clients:
                if model_type in client.loaded_models:
                    logger.info(f"Found client {client.client_id} with model {model_type} already loaded")
                    return client
            
            # If no client has the model loaded, find the one with the most free memory
            best_client = None
            max_free_memory = 0
            
            for client in active_clients:
                if client.status == "active":
                    free_memory = client.gpu_info.get("free_memory", 0)
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_client = client
            
            if best_client:
                logger.info(f"Selected client {best_client.client_id} with {max_free_memory}GB free memory")
            else:
                logger.warning("No suitable client found")
            
            return best_client

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
        registry.register_client(client)
        return {"status": "success", "message": f"Client {client.client_id} registered"}
    except Exception as e:
        logger.error(f"Error registering client: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/heartbeat/{client_id}")
async def client_heartbeat(client_id: str, update_data: Dict):
    logger.info(f"Received heartbeat from client: {client_id}")
    try:
        # Add client_id to update_data if not present
        if "client_id" not in update_data:
            update_data["client_id"] = client_id
            
        if registry.update_client(client_id, update_data):
            return {"status": "success", "message": "Heartbeat received"}
        raise HTTPException(status_code=404, detail="Client not found")
    except Exception as e:
        logger.error(f"Error processing heartbeat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clients")
async def get_clients():
    logger.info("Received request for client list")
    try:
        active_clients = registry.get_active_clients()
        logger.info(f"Returning {len(active_clients)} active clients")
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
        client = registry.get_client_by_id(client_id)
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
        registry.remove_client(client_id)
        return {"status": "success", "message": f"Client {client_id} removed"}
    except Exception as e:
        logger.error(f"Error removing client: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Handle prediction requests and route them to appropriate clients"""
    logger.info(f"Received prediction request for model: {request.model_type}")
    
    # Find the best available client
    client = registry.find_best_client(request.model_type)
    if not client:
        raise HTTPException(status_code=503, detail="No suitable client available")
    
    # Forward the request to the selected client
    client_url = f"http://{client.ip_address}:{client.port}/predict"
    logger.info(f"Forwarding request to client: {client_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(client_url, json=request.dict()) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully received response from client {client.client_id}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Client returned error: {response.status} - {error_text}")
                    raise HTTPException(status_code=response.status, detail=error_text)
    except Exception as e:
        logger.error(f"Error forwarding request to client: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug") 