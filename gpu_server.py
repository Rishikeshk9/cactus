from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import logging

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
    last_heartbeat: str  # Changed from datetime to str
    status: str  # "active", "inactive", "busy"
    capabilities: Dict  # VRAM, CUDA version, etc.

    def get_last_heartbeat(self) -> datetime:
        """Convert string heartbeat to datetime"""
        return datetime.fromisoformat(self.last_heartbeat)

class ClientRegistry:
    def __init__(self):
        self.clients: Dict[str, GPUClient] = {}
        self.heartbeat_timeout = 30  # seconds
        logger.info("Initialized ClientRegistry")

    def register_client(self, client: GPUClient):
        logger.info(f"Registering new client: {client.client_id}")
        logger.debug(f"Client details: {client.dict()}")
        self.clients[client.client_id] = client
        logger.info(f"Successfully registered client: {client.client_id} at {client.ip_address}:{client.port}")
        logger.info(f"Total clients: {len(self.clients)}")

    def update_client(self, client_id: str, update_data: Dict):
        logger.info(f"Updating client: {client_id}")
        logger.debug(f"Update data: {update_data}")
        if client_id in self.clients:
            client = self.clients[client_id]
            for key, value in update_data.items():
                setattr(client, key, value)
            logger.info(f"Successfully updated client: {client_id}")
            return True
        logger.warning(f"Client not found: {client_id}")
        return False

    def remove_client(self, client_id: str):
        logger.info(f"Removing client: {client_id}")
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Successfully removed client: {client_id}")
            logger.info(f"Remaining clients: {len(self.clients)}")
        else:
            logger.warning(f"Client not found for removal: {client_id}")

    def get_active_clients(self) -> List[GPUClient]:
        current_time = datetime.now()
        active_clients = []
        logger.info(f"Checking active clients at {current_time}")
        for client_id, client in self.clients.items():
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
        if client_id in self.clients:
            logger.info(f"Found client: {client_id}")
            return self.clients[client_id]
        logger.warning(f"Client not found: {client_id}")
        return None

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

if __name__ == "__main__":
    logger.info("Starting server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug") 