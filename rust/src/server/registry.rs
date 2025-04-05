use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;
use chrono::{DateTime, Utc};
use thiserror::Error;
use uuid::Uuid;

use crate::protocol::types::{GPUClient, HeartbeatUpdate};

#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("client not found")]
    ClientNotFound,
}

pub struct ClientRegistry {
    clients: RwLock<HashMap<Uuid, GPUClient>>,
    heartbeat_timeout: Duration,
}

impl ClientRegistry {
    pub fn new() -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            heartbeat_timeout: Duration::from_secs(3),
        }
    }

    pub async fn register_client(&self, client: GPUClient) -> Result<(), RegistryError> {
        let mut clients = self.clients.write().unwrap();
        clients.insert(client.client_id, client);
        Ok(())
    }

    pub async fn update_client(&self, update: HeartbeatUpdate) -> Result<(), RegistryError> {
        let mut clients = self.clients.write().unwrap();
        if let Some(client) = clients.get_mut(&update.client_id) {
            client.loaded_models = update.loaded_models;
            client.status = update.status;
            client.last_heartbeat = update.last_heartbeat;
            client.capabilities = update.capabilities;
            client.gpu_info = update.gpu_info;
            if let Some(ip) = update.ip_address {
                client.ip_address = ip;
            }
            Ok(())
        } else {
            Err(RegistryError::ClientNotFound)
        }
    }

    pub async fn find_best_client(&self, model_name: &str) -> Option<GPUClient> {
        let mut clients = self.clients.write().unwrap();
        let now = Utc::now();

        // Clean up inactive clients first
        clients.retain(|_, client| {
            let is_active = now.signed_duration_since(client.last_heartbeat) < chrono::Duration::seconds(30);
            if !is_active {
                tracing::info!("Removing inactive client: {}", client.client_id);
            }
            is_active
        });

        // Find clients with GPU and required model
        let mut available_clients: Vec<_> = clients
            .values()
            .filter(|client| {
                // First check if client is online
                if client.status != "online" {
                    tracing::warn!("Client {} is not online (status: {})", client.client_id, client.status);
                    return false;
                }

                // Check if client has GPU
                if client.gpu_info.total_memory == 0.0 {
                    tracing::warn!("Client {} has no GPU memory", client.client_id);
                    return false;
                }

                // Check if model is already loaded
                if client.loaded_models.contains(&model_name.to_string()) {
                    tracing::info!("Client {} already has model {} loaded", client.client_id, model_name);
                    return true;
                }

                // Check if client can load the model
                let has_enough_memory = match model_name {
                    "stable_diffusion" => client.gpu_info.total_memory >= 8.0, // SD typically needs 8GB+ VRAM
                    _ => true,
                };

                if !has_enough_memory {
                    tracing::warn!(
                        "Client {} has insufficient GPU memory ({}GB) for model {}",
                        client.client_id,
                        client.gpu_info.total_memory,
                        model_name
                    );
                    return false;
                }

                tracing::info!(
                    "Client {} can load model {} (GPU: {}GB VRAM, Status: {})",
                    client.client_id,
                    model_name,
                    client.gpu_info.total_memory,
                    client.status
                );
                true
            })
            .cloned()
            .collect();

        // Sort by GPU memory (highest first) and free memory (highest first)
        available_clients.sort_by(|a, b| {
            // First compare by total GPU memory
            let memory_cmp = b.gpu_info.total_memory.partial_cmp(&a.gpu_info.total_memory)
                .unwrap_or(std::cmp::Ordering::Equal);
            
            if memory_cmp != std::cmp::Ordering::Equal {
                memory_cmp
            } else {
                // If total memory is equal, compare by free memory
                b.gpu_info.free_memory.partial_cmp(&a.gpu_info.free_memory)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Take the best client and load the model if needed
        if let Some(mut client) = available_clients.first().cloned() {
            if !client.loaded_models.contains(&model_name.to_string()) {
                tracing::info!(
                    "Loading model {} on client {} (GPU: {}GB VRAM, Free: {}GB)",
                    model_name,
                    client.client_id,
                    client.gpu_info.total_memory,
                    client.gpu_info.free_memory
                );
                // Add model to loaded models
                client.loaded_models.push(model_name.to_string());
                // Update client in registry
                if let Some(existing_client) = clients.get_mut(&client.client_id) {
                    *existing_client = client.clone();
                }
            }
            Some(client)
        } else {
            tracing::warn!("No suitable online GPU clients found for model {}", model_name);
            None
        }
    }

    pub async fn get_active_clients(&self) -> Vec<GPUClient> {
        let clients = self.clients.read().unwrap();
        let now = Utc::now();
        
        clients
            .values()
            .filter(|client| {
                (now - client.last_heartbeat).num_seconds() < self.heartbeat_timeout.as_secs() as i64
            })
            .cloned()
            .collect()
    }

    pub async fn cleanup_inactive_clients(&self) {
        let mut clients = self.clients.write().unwrap();
        let now = Utc::now();
        
        clients.retain(|_, client| {
            (now - client.last_heartbeat).num_seconds() < self.heartbeat_timeout.as_secs() as i64
        });
    }

    pub async fn print_clients_table(&self) {
        // Clean up inactive clients before printing
        self.cleanup_inactive_clients().await;
        
        let clients = self.get_active_clients().await;
        println!("\nConnected Clients:");
        println!("{:<36} {:<15} {:<10} {:<20} {:<10}", 
            "Client ID", "IP Address", "Port", "Status", "Models");
        println!("{}", "-".repeat(101));

        for client in clients {
            println!("{:<36} {:<15} {:<10} {:<20} {:<10}",
                client.client_id,
                client.ip_address,
                client.port,
                client.status,
                client.loaded_models.join(", "));
        }
        println!();
    }
} 