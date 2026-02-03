//! Navigator Sandbox library.
//!
//! This crate provides process sandboxing and monitoring capabilities.

mod policy;
mod process;
mod proxy;
mod sandbox;

use miette::{IntoDiagnostic, Result};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{error, info};

use crate::policy::NetworkMode;
use crate::policy::SandboxPolicy;
use crate::proxy::ProxyHandle;
pub use process::{ProcessHandle, ProcessStatus};

/// Run a command in the sandbox.
///
/// # Errors
///
/// Returns an error if the command fails to start or encounters a fatal error.
pub async fn run_sandbox(
    command: Vec<String>,
    workdir: Option<String>,
    timeout_secs: u64,
    interactive: bool,
    policy_path: Option<String>,
    _health_check: bool,
    _health_port: u16,
) -> Result<i32> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| miette::miette!("No command specified"))?;

    let policy_path = policy_path.or_else(|| std::env::var("NAVIGATOR_SANDBOX_POLICY").ok());
    let policy_path = policy_path.ok_or_else(|| {
        miette::miette!("Sandbox policy is required. Provide --policy or NAVIGATOR_SANDBOX_POLICY")
    })?;
    info!(policy_path = policy_path, "Loading sandbox policy");
    let policy = SandboxPolicy::from_path(std::path::Path::new(&policy_path))?;

    let _proxy = if matches!(policy.network.mode, NetworkMode::Proxy) {
        let proxy_policy = policy.network.proxy.as_ref().ok_or_else(|| {
            miette::miette!("Network mode is set to proxy but no proxy configuration was provided")
        })?;
        Some(ProxyHandle::start(proxy_policy).await?)
    } else {
        None
    };

    let mut handle = ProcessHandle::spawn(program, args, workdir.as_deref(), interactive, &policy)?;

    info!(pid = handle.pid(), "Process started");

    // Wait for process with optional timeout
    let result = if timeout_secs > 0 {
        if let Ok(result) = timeout(Duration::from_secs(timeout_secs), handle.wait()).await {
            result
        } else {
            error!("Process timed out, killing");
            handle.kill()?;
            return Ok(124); // Standard timeout exit code
        }
    } else {
        handle.wait().await
    };

    let status = result.into_diagnostic()?;

    info!(exit_code = status.code(), "Process exited");

    Ok(status.code())
}
