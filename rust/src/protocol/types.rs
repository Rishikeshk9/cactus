use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::error::Error;
use reqwest::Client;
use std::fmt;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use tracing;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    pub device_name: String,
    pub total_memory: f64,
    pub allocated_memory: f64,
    pub reserved_memory: f64,
    pub free_memory: f64,
    pub cuda_version: String,
    pub compute_capability: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    pub models: Vec<String>,
    pub gpu_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUClient {
    pub client_id: Uuid,
    pub ip_address: String,
    pub port: u16,
    pub gpu_info: GPUInfo,
    pub loaded_models: Vec<String>,
    pub last_heartbeat: DateTime<Utc>,
    pub status: String,
    pub capabilities: ClientCapabilities,
}

#[derive(Debug)]
pub enum ClientError {
    RequestFailed(String),
    InvalidResponse(String),
    ServerError(String),
    InvalidRequest(String),
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClientError::RequestFailed(msg) => write!(f, "Request failed: {}", msg),
            ClientError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            ClientError::ServerError(msg) => write!(f, "Server error: {}", msg),
            ClientError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
        }
    }
}

impl Error for ClientError {}

// Make ClientError Send and Sync
unsafe impl Send for ClientError {}
unsafe impl Sync for ClientError {}

impl GPUClient {
    pub async fn send_prediction_request(&self, request: PredictionRequest) -> Result<PredictionResponse, ClientError> {
        let client = Client::new();
        
        // Check if this is a domain name (contains dots) or an IP address
        let url = if self.ip_address.contains('.') && !self.ip_address.chars().all(|c| c.is_digit(10) || c == '.') {
            // For domain names, use https and don't append port
            format!("http://{}/predict", self.ip_address)
        } else {
            // For IP addresses, use http and append port
            format!("http://{}:{}/predict", self.ip_address, self.port)
        };
        
        tracing::info!("Forwarding prediction request to client at URL: {}", url);
        
        // Validate request based on model type
        match request.model_type {
            ModelType::CovidXRay => {
                if request.image_url.is_none() {
                    return Err(ClientError::InvalidRequest("image_url is required for COVID X-Ray model".to_string()));
                }
            }
            ModelType::StableDiffusion => {
                if request.prompt.is_none() || request.quality_preset.is_none() {
                    return Err(ClientError::InvalidRequest("prompt and quality_preset are required for Stable Diffusion model".to_string()));
                }
            }
        }
        
        let response = client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ClientError::RequestFailed(e.to_string()))?;
            
        if !response.status().is_success() {
            return Err(ClientError::ServerError(format!("Client returned error: {}", response.status())));
        }
        
        let prediction_response: PredictionResponse = response.json().await
            .map_err(|e| ClientError::InvalidResponse(e.to_string()))?;
        Ok(prediction_response)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum QualityPreset {
    Fast,
    Balanced,
    Quality,
}

impl QualityPreset {
    pub fn get_inference_steps(&self) -> u32 {
        match self {
            QualityPreset::Fast => 20,
            QualityPreset::Balanced => 30,
            QualityPreset::Quality => 50,
        }
    }

    pub fn get_guidance_scale(&self) -> f32 {
        match self {
            QualityPreset::Fast => 7.5,
            QualityPreset::Balanced => 8.5,
            QualityPreset::Quality => 9.5,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ModelType {
    #[serde(rename = "covid_xray")]
    CovidXRay,
    #[serde(rename = "stable_diffusion")]
    StableDiffusion,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PredictionRequest {
    pub model_type: ModelType,
    pub model_cid: String,
    // Fields for COVID X-Ray
    pub image_url: Option<String>,
    // Fields for Stable Diffusion
    pub prompt: Option<String>,
    pub quality_preset: Option<QualityPreset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatUpdate {
    pub client_id: Uuid,
    pub loaded_models: Vec<String>,
    pub status: String,
    pub last_heartbeat: DateTime<Utc>,
    pub ip_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerResponse {
    pub status: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PredictionResponse {
    pub success: bool,
    pub prompt: Option<String>,
    pub generation_time_ms: Option<f64>,
    pub parameters: Option<HashMap<String, f32>>,
    pub timestamp: Option<String>,
    pub image_base64: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug)]
pub struct PredictionError {
    pub status: StatusCode,
    pub message: String,
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.status, self.message)
    }
}

impl Error for PredictionError {}

// Make PredictionError Send and Sync
unsafe impl Send for PredictionError {}
unsafe impl Sync for PredictionError {}

impl IntoResponse for PredictionError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": self.message,
            "status": self.status.as_u16()
        });
        
        (self.status, axum::Json(body)).into_response()
    }
} 