//! gRPC service implementation.

use navigator_core::proto::{
    HealthRequest, HealthResponse, ServiceStatus, navigator_server::Navigator,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::ServerState;

/// Navigator gRPC service implementation.
#[derive(Debug, Clone)]
pub struct NavigatorService {
    state: Arc<ServerState>,
}

impl NavigatorService {
    /// Create a new Navigator service.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(state: Arc<ServerState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl Navigator for NavigatorService {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            status: ServiceStatus::Healthy.into(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: self.state.uptime_seconds(),
        }))
    }
}
