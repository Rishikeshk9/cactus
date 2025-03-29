use axum::{
    routing::{post, get},
    Router,
    extract::{Path, Json, State},
    http::{StatusCode, Method},
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use uuid::Uuid;
use tower_http::cors::{CorsLayer, Any};
use tracing;
use std::time::Duration;
use std::net::SocketAddr;
use chrono::Utc;

use crate::protocol::types::{
    GPUClient, PredictionRequest, HeartbeatUpdate, ServerResponse, 
    PredictionError, ModelType
};
use crate::server::registry::{ClientRegistry, RegistryError};

pub mod registry;

pub struct AppState {
    pub registry: Arc<ClientRegistry>,
}

pub struct Server {
    host: String,
    port: u16,
    registry: Arc<ClientRegistry>,
}

impl Server {
    pub fn new(host: String, port: u16) -> anyhow::Result<Self> {
        Ok(Self {
            host,
            port,
            registry: Arc::new(ClientRegistry::new()),
        })
    }

    pub async fn run(&self) -> anyhow::Result<()> {
        let registry = self.registry.clone();

        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers(Any)
            .max_age(Duration::from_secs(3600));

        let app = Router::new()
            .route("/register", post(register_client))
            .route("/heartbeat/:client_id", post(update_client))
            .route("/predict", post(predict))
            .route("/clients", get(list_clients))
            .layer(cors)
            .with_state(registry);

            let addr: SocketAddr = format!("0.0.0.0:{}", self.port).parse()?;
            tracing::info!("Starting server on {}", addr);

        axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
        Ok(())
    }
}

async fn register_client(
    State(registry): State<Arc<ClientRegistry>>,
    Json(client): Json<GPUClient>,
) -> Json<ServerResponse> {
    match registry.register_client(client).await {
        Ok(_) => Json(ServerResponse {
            status: "success".to_string(),
            message: "Client registered successfully".to_string(),
        }),
        Err(e) => Json(ServerResponse {
            status: "error".to_string(),
            message: format!("Failed to register client: {}", e),
        }),
    }
}

async fn update_client(
    State(registry): State<Arc<ClientRegistry>>,
    Json(update): Json<HeartbeatUpdate>,
) -> Json<ServerResponse> {
    match registry.update_client(update).await {
        Ok(_) => Json(ServerResponse {
            status: "success".to_string(),
            message: "Client updated successfully".to_string(),
        }),
        Err(e) => Json(ServerResponse {
            status: "error".to_string(),
            message: format!("Failed to update client: {}", e),
        }),
    }
}

async fn list_clients(State(registry): State<Arc<ClientRegistry>>) -> Json<Vec<GPUClient>> {
    tracing::info!("Received request to list clients");
    let clients = registry.get_active_clients().await;
    tracing::info!("Found {} active clients", clients.len());
    Json(clients)
}

async fn client_heartbeat(
    State(registry): State<Arc<ClientRegistry>>,
    Path(_client_id): Path<Uuid>,
    Json(update): Json<HeartbeatUpdate>,
) -> Result<Json<ServerResponse>, StatusCode> {
    match registry.update_client(update).await {
        Ok(_) => {
            registry.print_clients_table().await;
            Ok(Json(ServerResponse {
                status: "success".to_string(),
                message: "Heartbeat updated".to_string(),
            }))
        }
        Err(RegistryError::ClientNotFound) => {
            Ok(Json(ServerResponse {
                status: "error".to_string(),
                message: "Client not found".to_string(),
            }))
        }
    }
}

async fn get_clients(
    State(registry): State<Arc<ClientRegistry>>,
) -> Result<Json<Vec<GPUClient>>, StatusCode> {
    let clients = registry.get_active_clients().await;
    Ok(Json(clients))
}

#[axum::debug_handler]
async fn predict(
    State(registry): State<Arc<ClientRegistry>>,
    Json(request): Json<PredictionRequest>,
) -> Response {
    tracing::info!(
        "Received prediction request for model type {:?}",
        request.model_type
    );

    // Validate request based on model type
    match request.model_type {
        ModelType::CovidXRay => {
            if request.image_url.is_none() {
                return PredictionError {
                    status: StatusCode::BAD_REQUEST,
                    message: "image_url is required for COVID X-Ray model".to_string(),
                }.into_response();
            }
        }
        ModelType::StableDiffusion => {
            if request.prompt.is_none() || request.quality_preset.is_none() {
                return PredictionError {
                    status: StatusCode::BAD_REQUEST,
                    message: "prompt and quality_preset are required for Stable Diffusion model".to_string(),
                }.into_response();
            }
            if request.prompt.as_ref().unwrap().is_empty() {
                return PredictionError {
                    status: StatusCode::BAD_REQUEST,
                    message: "Empty prompt".to_string(),
                }.into_response();
            }
        }
    }

    // Find best available client based on model type
    let model_name = match request.model_type {
        ModelType::CovidXRay => "covid_xray",
        ModelType::StableDiffusion => "stable_diffusion",
    };

    let client = match registry.find_best_client(model_name).await {
        Some(client) => {
            tracing::info!(
                "Found client {} at {}:{} for model type {}",
                client.client_id,
                client.ip_address,
                client.port,
                model_name
            );
            client
        }
        None => {
            tracing::error!("No available client found for model type {}", model_name);
            return PredictionError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: format!("No available client found for model type {}", model_name),
            }.into_response();
        }
    };

    // Send prediction request to client
    match client.send_prediction_request(request).await {
        Ok(response) => {
            tracing::info!(
                "Received prediction response from client {}",
                client.client_id
            );
            Json(response).into_response()
        }
        Err(e) => {
            tracing::error!(
                "Error getting prediction from client {}: {}",
                client.client_id,
                e
            );
            // Check if client is still active
            if let Err(e) = registry.update_client(HeartbeatUpdate {
                client_id: client.client_id,
                status: "error".to_string(),
                loaded_models: client.loaded_models.clone(),
                last_heartbeat: Utc::now(),
                ip_address: Some(client.ip_address.clone()),
                capabilities: client.capabilities.clone(),
            }).await {
                tracing::warn!("Failed to update client status: {}", e);
            }
            PredictionError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: e.to_string(),
            }.into_response()
        }
    }
} 