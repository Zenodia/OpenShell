//! CLI command implementations.

use indicatif::{ProgressBar, ProgressStyle};
use miette::{IntoDiagnostic, Result};
use navigator_core::proto::{HealthRequest, navigator_client::NavigatorClient};
use owo_colors::OwoColorize;
use std::time::Duration;

/// Check server health.
#[allow(clippy::significant_drop_tightening)]
pub async fn health(server: &str) -> Result<()> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .into_diagnostic()?,
    );
    spinner.set_message("Checking server health...");
    spinner.enable_steady_tick(Duration::from_millis(100));

    let mut client = NavigatorClient::connect(server.to_string())
        .await
        .into_diagnostic()?;

    let response = client.health(HealthRequest {}).await.into_diagnostic()?;
    let health = response.into_inner();

    spinner.finish_and_clear();

    let (status_icon, status_text) = match health.status {
        1 => ("●".green().to_string(), "healthy"),
        2 => ("●".yellow().to_string(), "degraded"),
        _ => ("●".red().to_string(), "unhealthy"),
    };

    println!("{status_icon} Server is {status_text}");
    println!("  {} {}", "Version:".dimmed(), health.version);
    println!("  {} {}s", "Uptime:".dimmed(), health.uptime_seconds);

    Ok(())
}

/// Show server status.
pub async fn status(server: &str) -> Result<()> {
    println!("{}", "Server Status".bold().cyan());
    println!();
    println!("  {} {}", "Server:".dimmed(), server);

    // Try to connect and get health
    match NavigatorClient::connect(server.to_string()).await {
        Ok(mut client) => match client.health(HealthRequest {}).await {
            Ok(response) => {
                let health = response.into_inner();
                println!("  {} {}", "Status:".dimmed(), "Connected".green());
                println!("  {} {}", "Version:".dimmed(), health.version);
                println!("  {} {}s", "Uptime:".dimmed(), health.uptime_seconds);
            }
            Err(e) => {
                println!("  {} {}", "Status:".dimmed(), "Error".red());
                println!("  {} {}", "Error:".dimmed(), e);
            }
        },
        Err(e) => {
            println!("  {} {}", "Status:".dimmed(), "Disconnected".red());
            println!("  {} {}", "Error:".dimmed(), e);
        }
    }

    Ok(())
}
