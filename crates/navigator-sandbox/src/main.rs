//! Navigator Sandbox - process sandbox and monitor.

use clap::Parser;
use miette::{IntoDiagnostic, Result};
use tracing::info;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

use navigator_sandbox::run_sandbox;

/// Navigator Sandbox - process isolation and monitoring.
#[derive(Parser, Debug)]
#[command(name = "navigator-sandbox")]
#[command(about = "Process sandbox and monitor", long_about = None)]
struct Args {
    /// Command to execute in the sandbox.
    #[arg(trailing_var_arg = true, required = true)]
    command: Vec<String>,

    /// Working directory for the sandboxed process.
    #[arg(long, short)]
    workdir: Option<String>,

    /// Timeout in seconds (0 = no timeout).
    #[arg(long, short, default_value = "0")]
    timeout: u64,

    /// Run in interactive mode (inherit process group for terminal control).
    #[arg(long, short = 'i')]
    interactive: bool,

    /// Path to YAML policy file for sandbox configuration.
    #[arg(long, env = "NAVIGATOR_SANDBOX_POLICY", required = true)]
    policy: Option<String>,

    /// Log level (trace, debug, info, warn, error).
    #[arg(long, default_value = "warn", env = "NAVIGATOR_LOG_LEVEL")]
    log_level: String,

    /// Enable health check endpoint.
    #[arg(long)]
    health_check: bool,

    /// Port for health check endpoint.
    #[arg(long, default_value = "8080")]
    health_port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/var/log/navigator.log")
        .into_diagnostic()?;
    let (file_writer, _file_guard) = tracing_appender::non_blocking(file);

    let stdout_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&args.log_level));
    let file_filter = EnvFilter::new("info");

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stdout)
                .with_filter(stdout_filter),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(file_writer)
                .with_ansi(false)
                .with_filter(file_filter),
        )
        .init();

    info!(command = ?args.command, "Starting sandbox");

    let exit_code = run_sandbox(
        args.command,
        args.workdir,
        args.timeout,
        args.interactive,
        args.policy,
        args.health_check,
        args.health_port,
    )
    .await?;

    std::process::exit(exit_code);
}
