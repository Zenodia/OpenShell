//! HTTP health endpoints using Axum.

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use serde::Serialize;
use std::sync::Arc;

use crate::ServerState;

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Service status.
    pub status: &'static str,

    /// Service version.
    pub version: &'static str,

    /// Uptime in seconds.
    pub uptime_seconds: u64,
}

/// Simple health check - returns 200 OK.
async fn health() -> impl IntoResponse {
    StatusCode::OK
}

/// Kubernetes liveness probe.
async fn healthz() -> impl IntoResponse {
    StatusCode::OK
}

/// Kubernetes readiness probe with detailed status.
async fn readyz(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let response = HealthResponse {
        status: "healthy",
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: state.uptime_seconds(),
    };

    (StatusCode::OK, Json(response))
}

/// Create the health router.
pub fn health_router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .with_state(state)
}
