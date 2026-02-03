//! Navigator CLI - command-line interface for Navigator.

use clap::{Parser, Subcommand};
use miette::Result;
use owo_colors::OwoColorize;

use navigator_cli::run;

/// Navigator CLI - agent execution and management.
#[derive(Parser, Debug)]
#[command(name = "navigator")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Increase verbosity (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Server address to connect to.
    #[arg(
        long,
        short,
        default_value = "http://127.0.0.1:50051",
        global = true,
        env = "NAVIGATOR_SERVER"
    )]
    server: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Check server health.
    Health,

    /// Show server status and information.
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging based on verbosity
    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .init();

    match cli.command {
        Some(Commands::Health) => {
            run::health(&cli.server).await?;
        }
        Some(Commands::Status) => {
            run::status(&cli.server).await?;
        }
        None => {
            println!(
                "{} {}",
                "Navigator".bold().cyan(),
                env!("CARGO_PKG_VERSION").dimmed()
            );
            println!();
            println!("Run {} for usage information.", "--help".green());
        }
    }

    Ok(())
}
