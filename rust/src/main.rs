use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod protocol;
mod server;
mod client;

use server::Server;
use client::manager::GPUClientManager;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the server
    Server {
        /// Host to bind to
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind to
        #[arg(short, long, default_value = "8001")]
        port: u16,
    },
    /// Run the client
    Client {
        /// Server URL
        #[arg(short, long, default_value = "http://127.0.0.1:8001")]
        server_url: String,
        /// Port to bind to
        #[arg(short, long, default_value = "8002")]
        port: u16,
        /// Public IP address of the client
        #[arg(short = 'i', long)]
        public_ip: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Server { host, port } => {
            let server = Server::new(host, port)?;
            server.run().await?;
        }
        Commands::Client { server_url, port, public_ip } => {
            let mut client = GPUClientManager::new(server_url, port, public_ip)?;
            let client_clone = client.clone();

            // Handle shutdown signals
            tokio::spawn(async move {
                let ctrl_c = async {
                    tokio::signal::ctrl_c()
                        .await
                        .expect("Failed to listen for ctrl+c");
                };

                #[cfg(unix)]
                let terminate = async {
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("Failed to install SIGTERM handler")
                        .recv()
                        .await;
                };

                #[cfg(not(unix))]
                let terminate = std::future::pending::<()>();

                tokio::select! {
                    _ = ctrl_c => {},
                    _ = terminate => {},
                }

                client_clone.stop();
            });

            if let Err(e) = client.start().await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }

    Ok(())
} 