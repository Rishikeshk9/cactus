import aiohttp
import asyncio
import uuid
import socket
import torch
from datetime import datetime
from typing import Dict, Optional
import logging
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class GPUClientManager:
    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())
        self.ip_address = self._get_local_ip()
        self.port = 8000  # Default port for gpu_load.py
        self.session: Optional[aiohttp.ClientSession] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.is_running = False
        logger.info(f"Initialized GPUClientManager with server URL: {self.server_url}")
        logger.info(f"Client ID: {self.client_id}")
        logger.info(f"IP Address: {self.ip_address}")

    def _get_local_ip(self) -> str:
        try:
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            logger.info(f"Found local IP: {ip}")
            return ip
        except Exception as e:
            logger.error(f"Error getting local IP: {e}, using localhost")
            return "127.0.0.1"

    def _get_gpu_info(self) -> Dict:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved_memory = torch.cuda.memory_reserved(device) / 1024**3  # GB
            free_memory = total_memory - reserved_memory  # GB
            
            info = {
                "device_name": torch.cuda.get_device_name(device),
                "total_memory": total_memory,
                "allocated_memory": allocated_memory,
                "reserved_memory": reserved_memory,
                "free_memory": free_memory,
                "cuda_version": torch.version.cuda,
                "compute_capability": f"{torch.cuda.get_device_capability(device)[0]}.{torch.cuda.get_device_capability(device)[1]}"
            }
            logger.info(f"GPU Info: {info}")
            return info
        logger.warning("No GPU available, using CPU")
        return {
            "device_name": "CPU",
            "total_memory": 0,
            "allocated_memory": 0,
            "reserved_memory": 0,
            "free_memory": 0,
            "cuda_version": "N/A",
            "compute_capability": "N/A"
        }

    async def register(self, loaded_models: list):
        """Register with the server"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            logger.info("Created new aiohttp session")

        client_data = {
            "client_id": self.client_id,
            "ip_address": self.ip_address,
            "port": self.port,
            "gpu_info": self._get_gpu_info(),
            "loaded_models": loaded_models,
            "last_heartbeat": datetime.now().isoformat(),
            "status": "active",
            "capabilities": {
                "max_batch_size": 1,
                "supported_models": ["stable_diffusion", "stable_diffusion_xl"],
                "max_resolution": 1024
            }
        }

        logger.info(f"Attempting to register with server at {self.server_url}/register")
        logger.debug(f"Registration data: {client_data}")

        try:
            async with self.session.post(f"{self.server_url}/register", json=client_data) as response:
                if response.status == 200:
                    logger.info(f"Successfully registered with server. Client ID: {self.client_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to register with server. Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error registering with server: {str(e)}")
            return False

    async def send_heartbeat(self, loaded_models: list, status: str = "active"):
        """Send heartbeat to server"""
        if not self.session:
            logger.warning("No active session for heartbeat")
            return False

        update_data = {
            "loaded_models": loaded_models,
            "status": status,
            "last_heartbeat": datetime.now().isoformat()
        }

        try:
            logger.debug(f"Sending heartbeat to {self.server_url}/heartbeat/{self.client_id}")
            async with self.session.post(
                f"{self.server_url}/heartbeat/{self.client_id}",
                json=update_data
            ) as response:
                if response.status == 200:
                    logger.debug("Heartbeat sent successfully")
                    return True
                else:
                    logger.error(f"Heartbeat failed with status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error sending heartbeat: {str(e)}")
            return False

    async def heartbeat_loop(self, loaded_models_callback):
        """Maintain heartbeat with server"""
        self.is_running = True
        logger.info("Starting heartbeat loop")
        while self.is_running:
            try:
                loaded_models = loaded_models_callback()
                await self.send_heartbeat(loaded_models)
                await asyncio.sleep(10)  # Send heartbeat every 10 seconds
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def start(self, loaded_models_callback, server_url: str = None):
        """Start the client manager"""
        logger.info("Starting client manager...")
        if server_url:
            self.server_url = server_url
            logger.info(f"Using provided server URL: {self.server_url}")
        
        if await self.register(loaded_models_callback()):
            self.heartbeat_task = asyncio.create_task(
                self.heartbeat_loop(loaded_models_callback)
            )
            logger.info("Client manager started successfully")
            return True
        logger.error("Failed to start client manager")
        return False

    async def stop(self):
        """Stop the client manager"""
        logger.info("Stopping client manager...")
        self.is_running = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            logger.info("Heartbeat task cancelled")
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Session closed")
        logger.info("Client manager stopped") 