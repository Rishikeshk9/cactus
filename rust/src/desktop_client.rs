#![windows_subsystem = "windows"]

use eframe::egui::{self, RichText, Color32, TextFormat, Align, Layout};
use eframe::IconData;
use image::io::Reader as ImageReader;
use image::ImageFormat;
use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::io::{BufRead, BufReader, Cursor};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::fmt;

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(windows)]
const DETACHED_PROCESS: u32 = 0x00000008;

// Embed the icon data directly into the executable
const ICON_DATA: &[u8] = include_bytes!("../assets/cactus.ico");

#[derive(Debug, Clone)]
struct GPUInfo {
    device_name: String,
    total_memory: f64,
    allocated_memory: f64,
    reserved_memory: f64,
    cuda_version: String,
    cuda_driver_version: String,
    compute_capability: String,
    power_usage: String,
    temperature: String,
    utilization: String,
}

// Configuration structure that can be saved/loaded
#[derive(Serialize, Deserialize, Clone)]
struct ClientConfig {
    server_url: String,
    public_ip: String,
    port: String,
    // Add any other configuration parameters here
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            server_url: "http://3.110.255.211:8001".to_string(),
            public_ip: "".to_string(),
            port: "8002".to_string(),
        }
    }
}

#[derive(Clone)]
struct LogEntry {
    timestamp: Option<String>,
    level: Option<String>,
    module: Option<String>,
    message: String,
    is_error: bool,
}

#[derive(Clone)]
struct LogParser;

impl LogParser {
    fn parse_line(line: &str) -> LogEntry {
        if line.starts_with("ERROR: ") {
            return LogEntry {
                timestamp: None,
                level: Some("ERROR".to_string()),
                module: None,
                message: line["ERROR: ".len()..].to_string(),
                is_error: true,
            };
        }

        // Try to parse structured log line
        if line.contains("[2m") && line.contains("[0m") {
            let mut timestamp = None;
            let mut level = None;
            let mut module = None;
            let mut message = String::new();

            // Extract timestamp
            if let Some(timestamp_end) = line.find("Z") {
                if let Some(timestamp_start) = line.find("[2m") {
                    timestamp = Some(line[timestamp_start + 3..timestamp_end].trim().to_string());
                }
            }

            // Extract level
            if line.contains("[32m INFO") {
                level = Some("INFO".to_string());
            }

            // Extract module and message
            let parts: Vec<&str> = line.split("[0m").collect();
            for part in parts {
                let clean_part = part
                    .trim()
                    .trim_start_matches('□')
                    .trim_start_matches("[2m")
                    .trim_start_matches("[32m")
                    .trim_start_matches("INFO")
                    .trim();

                if !clean_part.is_empty() {
                    if clean_part.contains("::") && module.is_none() {
                        let module_parts: Vec<&str> = clean_part.split_whitespace().collect();
                        if !module_parts.is_empty() {
                            module = Some(module_parts[0].to_string());
                            if module_parts.len() > 1 {
                                message = module_parts[1..].join(" ");
                            }
                        }
                    } else {
                        if !message.is_empty() {
                            message.push(' ');
                        }
                        message.push_str(clean_part);
                    }
                }
            }

            LogEntry {
                timestamp,
                level,
                module,
                message,
                is_error: false,
            }
        } else {
            // Regular log line
            LogEntry {
                timestamp: None,
                level: None,
                module: None,
                message: line.trim_start_matches('□').trim().to_string(),
                is_error: false,
            }
        }
    }
}

#[derive(Clone, PartialEq)]
enum ClientStatus {
    Offline,
    Online,
    Busy,
    Error(String),
}

impl fmt::Display for ClientStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClientStatus::Offline => write!(f, "OFFLINE"),
            ClientStatus::Online => write!(f, "ONLINE"),
            ClientStatus::Busy => write!(f, "BUSY"),
            ClientStatus::Error(msg) => write!(f, "ERROR: {}", msg),
        }
    }
}

impl ClientStatus {
    fn color(&self) -> Color32 {
        match self {
            ClientStatus::Offline => Color32::from_gray(150),
            ClientStatus::Online => Color32::from_rgb(100, 255, 100),
            ClientStatus::Busy => Color32::from_rgb(255, 180, 50),
            ClientStatus::Error(_) => Color32::from_rgb(255, 100, 100),
        }
    }

    fn text(&self) -> String {
        match self {
            ClientStatus::Offline => "● OFFLINE".to_string(),
            ClientStatus::Online => "● ONLINE".to_string(),
            ClientStatus::Busy => "● BUSY".to_string(),
            ClientStatus::Error(msg) => format!("● ERROR: {}", msg),
        }
    }
}

struct CactusClientApp {
    config: ClientConfig,
    config_path: PathBuf,
    client_process: Option<Child>,
    status: ClientStatus,
    log_entries: Arc<Mutex<Vec<LogEntry>>>,
    should_stop: Arc<Mutex<bool>>,
    gpu_info: Option<GPUInfo>,
    last_gpu_update: std::time::Instant,
    log_parser: LogParser,
    last_activity: std::time::Instant,
}

impl CactusClientApp {
    fn new() -> Self {
        // Create config directory if it doesn't exist
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("cactus-client");
        
        fs::create_dir_all(&config_dir).ok();
        
        let config_path = config_dir.join("config.json");
        
        // Load config if it exists
        let config = if config_path.exists() {
            match fs::read_to_string(&config_path) {
                Ok(json) => match serde_json::from_str(&json) {
                    Ok(cfg) => cfg,
                    Err(_) => ClientConfig::default(),
                },
                Err(_) => ClientConfig::default(),
            }
        } else {
            ClientConfig::default()
        };

        let mut app = Self {
            config,
            config_path,
            client_process: None,
            status: ClientStatus::Offline,
            log_entries: Arc::new(Mutex::new(vec![LogEntry {
                timestamp: None,
                level: None,
                module: None,
                message: "Welcome to Cactus Protocol Client".to_string(),
                is_error: false,
            }])),
            should_stop: Arc::new(Mutex::new(false)),
            gpu_info: None,
            last_gpu_update: std::time::Instant::now(),
            log_parser: LogParser,
            last_activity: std::time::Instant::now(),
        };
        
        app.update_gpu_info();
        app
    }
    
    fn save_config(&self) -> Result<(), String> {
        match serde_json::to_string_pretty(&self.config) {
            Ok(json) => match fs::write(&self.config_path, json) {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Failed to write config: {}", e)),
            },
            Err(e) => Err(format!("Failed to serialize config: {}", e)),
        }
    }
    
    fn append_log(&self, text: &str) {
        if let Ok(mut logs) = self.log_entries.lock() {
            for line in text.lines() {
                logs.push(LogParser::parse_line(line));
            }
        }
    }
    
    fn update_status(&mut self) {
        // Update status based on log messages and process state
        if let Ok(logs) = self.log_entries.lock() {
            if let Some(last_log) = logs.last() {
                // Check for error messages
                if last_log.is_error {
                    self.status = ClientStatus::Error(last_log.message.clone());
                    return;
                }

                // Check for specific status messages
                if last_log.message.contains("Client registered successfully") {
                    self.status = ClientStatus::Online;
                    self.last_activity = std::time::Instant::now();
                } else if last_log.message.contains("Processing request") 
                    || last_log.message.contains("Generating") 
                    || last_log.message.contains("Loading model") {
                    self.status = ClientStatus::Busy;
                    self.last_activity = std::time::Instant::now();
                }
            }
        }

        // Check if client is running
        match &mut self.client_process {
            Some(process) => {
                match process.try_wait() {
                    Ok(Some(_)) => {
                        // Process has exited
                        self.status = ClientStatus::Offline;
                        self.client_process = None;
                    },
                    Ok(None) => {
                        // Process is still running
                        // If no activity for more than 30 seconds, check if still responsive
                        if self.last_activity.elapsed() > Duration::from_secs(30) 
                            && self.status != ClientStatus::Offline {
                            self.status = ClientStatus::Online;
                        }
                    },
                    Err(_) => {
                        // Error checking process status
                        self.status = ClientStatus::Error("Failed to check process status".to_string());
                        self.client_process = None;
                    }
                }
            },
            None => {
                self.status = ClientStatus::Offline;
            }
        }
    }

    fn start_client(&mut self) -> Result<(), String> {
        if self.client_process.is_some() {
            return Err("Client is already running".to_string());
        }
        
        if self.config.public_ip.trim().is_empty() {
            return Err("Public IP is required".to_string());
        }
        
        // Save the current configuration
        self.save_config().ok();
        
        let log_line = format!(
            "Starting client with:\nServer URL: {}\nPublic IP: {}\nPort: {}\n",
            self.config.server_url, self.config.public_ip, self.config.port
        );
        self.append_log(&log_line);
        
        // Find the path to cactus.exe
        let executable_path = std::env::current_exe().unwrap_or_default();
        let executable_dir = executable_path.parent().unwrap_or_else(|| std::path::Path::new(""));
        let cactus_path = executable_dir.join("cactus.exe");
        
        // Create the command to run the client
        let command_path = if cactus_path.exists() {
            self.append_log(&format!("Using cactus.exe from: {}\n", cactus_path.display()));
            cactus_path
        } else {
            self.append_log("Cactus.exe not found in the same directory, trying PATH...\n");
            std::path::PathBuf::from("cactus")
        };
        
        // Reset the stop flag
        *self.should_stop.lock().unwrap() = false;
        
        // Create the command with stdout and stderr piped
        let mut command = Command::new(command_path);
        
        // On Windows, set the CREATE_NO_WINDOW flag
        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);
        
        command
            .arg("client")
            .arg("--server-url")
            .arg(&self.config.server_url)
            .arg("--public-ip")
            .arg(&self.config.public_ip)
            .arg("--port")
            .arg(&self.config.port)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
            
        // Start the process
        match command.spawn() {
            Ok(mut child) => {
                // Get the stdout and stderr handles
                let stdout = child.stdout.take().expect("Failed to capture stdout");
                let stderr = child.stderr.take().expect("Failed to capture stderr");
                
                // Clone the log_entries Arc for the threads
                let log_entries_stdout = self.log_entries.clone();
                let log_entries_stderr = self.log_entries.clone();
                let should_stop_stdout = self.should_stop.clone();
                let should_stop_stderr = self.should_stop.clone();
                
                // Spawn a thread to read stdout
                thread::spawn(move || {
                    let reader = BufReader::new(stdout);
                    for line in reader.lines() {
                        if *should_stop_stdout.lock().unwrap() {
                            break;
                        }
                        if let Ok(line) = line {
                            if let Ok(mut logs) = log_entries_stdout.lock() {
                                logs.push(LogParser::parse_line(&line));
                            }
                        }
                    }
                });
                
                // Spawn a thread to read stderr
                thread::spawn(move || {
                    let reader = BufReader::new(stderr);
                    for line in reader.lines() {
                        if *should_stop_stderr.lock().unwrap() {
                            break;
                        }
                        if let Ok(line) = line {
                            if let Ok(mut logs) = log_entries_stderr.lock() {
                                logs.push(LogParser::parse_line(&line));
                            }
                        }
                    }
                });
                
                // Store the child process
                self.client_process = Some(child);
                self.status = ClientStatus::Online;
                self.last_activity = std::time::Instant::now();
                Ok(())
            },
            Err(e) => {
                let error_msg = format!("Failed to start client: {}", e);
                self.append_log(&format!("ERROR: {}\n", error_msg));
                self.status = ClientStatus::Error(error_msg.clone());
                Err(error_msg)
            }
        }
    }
    
    fn stop_client(&mut self) {
        // Set the stop flag to signal the reader threads to stop
        *self.should_stop.lock().unwrap() = true;
        
        if let Some(mut child) = self.client_process.take() {
            match child.kill() {
                Ok(_) => {
                    self.status = ClientStatus::Offline;
                    self.append_log("Client stopped\n");
                },
                Err(e) => {
                    self.status = ClientStatus::Error(format!("Failed to stop client: {}", e));
                    self.append_log(&format!("ERROR: {}\n", self.status));
                }
            }
        }
    }

    fn update_gpu_info(&mut self) {
        // Try nvidia-smi first with extended metrics
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap,power.draw,temperature.gpu,utilization.gpu", "--format=csv,nounits,noheader"])
            .creation_flags(CREATE_NO_WINDOW)
            .output();

        match output {
            Ok(output) if output.status.success() => {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    let lines: Vec<&str> = output_str.trim().split('\n').collect();
                    if !lines.is_empty() {
                        let parts: Vec<&str> = lines[0].split(',').map(|s| s.trim()).collect();
                        if parts.len() >= 9 {
                            let device_name = parts[0].to_string();
                            let total_memory = parts[1].parse::<f64>().unwrap_or(0.0);
                            let used_memory = parts[2].parse::<f64>().unwrap_or(0.0);
                            let free_memory = parts[3].parse::<f64>().unwrap_or(0.0);
                            let driver_version = parts[4].to_string();
                            let compute_capability = parts[5].to_string();
                            let power_usage = format!("{} W", parts[6]);
                            let temperature = format!("{} °C", parts[7]);
                            let utilization = format!("{}%", parts[8]);

                            self.gpu_info = Some(GPUInfo {
                                device_name,
                                total_memory,
                                allocated_memory: used_memory,
                                reserved_memory: 0.0,
                                cuda_version: "N/A".to_string(),
                                cuda_driver_version: driver_version,
                                compute_capability,
                                power_usage,
                                temperature,
                                utilization,
                            });
                            return;
                        }
                    }
                }
            }
            _ => {}
        }

        // If nvidia-smi fails, try PyTorch with extended information
        let output = Command::new("python")
            .args(&["-c", r#"
import torch
import subprocess

def get_nvidia_smi_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,power.draw,temperature.gpu,utilization.gpu', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split(',')
        return None
    except:
        return None

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    smi_info = get_nvidia_smi_info()
    
    print(f"{torch.cuda.get_device_name(device)}")  # Device name
    print(f"{torch.cuda.get_device_properties(device).total_memory / 1024**2}")  # Total memory
    print(f"{torch.cuda.memory_allocated(device) / 1024**2}")  # Allocated memory
    print(f"{torch.cuda.memory_reserved(device) / 1024**2}")  # Reserved memory
    print(f"{torch.version.cuda}")  # CUDA version
    
    if smi_info and len(smi_info) >= 4:
        print(f"{smi_info[0].strip()}")  # Driver version
        print(f"{smi_info[1].strip()} W")  # Power usage
        print(f"{smi_info[2].strip()} °C")  # Temperature
        print(f"{smi_info[3].strip()}%")  # Utilization
    else:
        print("N/A\nN/A\nN/A\nN/A")
    
    print(f"{torch.cuda.get_device_capability(device)[0]}.{torch.cuda.get_device_capability(device)[1]}")  # Compute capability
else:
    print("No GPU\n0\n0\n0\nN/A\nN/A\nN/A\nN/A\nN/A\nN/A")
"#])
            .creation_flags(CREATE_NO_WINDOW)
            .output();

        if let Ok(output) = output {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                let lines: Vec<&str> = output_str.trim().split('\n').collect();
                if lines.len() >= 10 {
                    let device_name = lines[0].to_string();
                    let total_memory = lines[1].parse::<f64>().unwrap_or(0.0);
                    let allocated_memory = lines[2].parse::<f64>().unwrap_or(0.0);
                    let reserved_memory = lines[3].parse::<f64>().unwrap_or(0.0);
                    let cuda_version = lines[4].to_string();
                    let cuda_driver_version = lines[5].to_string();
                    let power_usage = lines[6].to_string();
                    let temperature = lines[7].to_string();
                    let utilization = lines[8].to_string();
                    let compute_capability = lines[9].to_string();

                    self.gpu_info = Some(GPUInfo {
                        device_name,
                        total_memory,
                        allocated_memory,
                        reserved_memory,
                        cuda_version,
                        cuda_driver_version,
                        compute_capability,
                        power_usage,
                        temperature,
                        utilization,
                    });
                    return;
                }
            }
        }

        // If both methods fail, set GPU info to None
        self.gpu_info = None;
    }
}

// Add Drop implementation to handle cleanup when the app closes
impl Drop for CactusClientApp {
    fn drop(&mut self) {
        // Stop the client if it's running
        self.stop_client();
        
        // Extra safety: wait a short moment to ensure the process is terminated
        if let Some(mut child) = self.client_process.take() {
            // Try to kill the process again just to be sure
            let _ = child.kill();
            // Wait for the process to exit
            let _ = child.wait();
        }
    }
}

impl eframe::App for CactusClientApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update status
        self.update_status();

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Cactus Protocol Client");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // Status indicator
                    let status_text = RichText::new(self.status.text())
                        .color(self.status.color())
                        .monospace();
                    ui.label(status_text);
                });
            });
        });

        // Check if it's time to update GPU info (every 500ms)
        if self.last_gpu_update.elapsed() >= Duration::from_millis(500) {
            self.update_gpu_info();
            self.last_gpu_update = std::time::Instant::now();
        }

        // Request a repaint every 100ms to update the log display and ensure smooth GPU info updates
        ctx.request_repaint_after(Duration::from_millis(100));
        
        egui::CentralPanel::default().show(ctx, |ui| {
         
            
            ui.add_space(10.0);

            // Top section with two columns
            ui.horizontal(|ui| {
                // Left column: Configuration
                ui.vertical(|ui| {
                    ui.group(|ui| {
                        ui.heading("Configuration");
                        
                        ui.add_space(5.0);
                        
                        ui.horizontal(|ui| {
                            ui.label("Server URL:");
                            ui.text_edit_singleline(&mut self.config.server_url);
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Public IP:");
                            ui.text_edit_singleline(&mut self.config.public_ip);
                        });
                        
                        ui.add_space(2.0);
                        ui.label("Your IP must be static so jobs can reach you");
                        
                        ui.horizontal(|ui| {
                            ui.label("Port:");
                            ui.text_edit_singleline(&mut self.config.port);
                        });
                        
                        ui.add_space(10.0);
                        
                        // Save config button
                        if ui.button("Save Configuration").clicked() {
                            if let Err(e) = self.save_config() {
                                self.status = ClientStatus::Error(e);
                            } else {
                                self.status = ClientStatus::Online;
                            }
                        }
                    });
                });

                ui.add_space(10.0);

                // Right column: System Info
                ui.vertical(|ui| {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.heading("System Information");
                            
                            if ui.button("🔄").clicked() {
                                self.update_gpu_info();
                                self.last_gpu_update = std::time::Instant::now();
                            }
                        });
                        
                        ui.add_space(5.0);
                        
                        match &self.gpu_info {
                            Some(gpu) => {
                                ui.label(format!("GPU: {}", gpu.device_name));
                                ui.label(format!("Memory: {:.1}GB used / {:.1}GB total", 
                                    gpu.allocated_memory / 1024.0, 
                                    gpu.total_memory / 1024.0));
                                if gpu.reserved_memory > 0.0 {
                                    ui.label(format!("Reserved: {:.1}GB", gpu.reserved_memory / 1024.0));
                                }
                                ui.label(format!("CUDA: {} (Driver: {})", 
                                    gpu.cuda_version, 
                                    gpu.cuda_driver_version));
                                ui.label(format!("Compute: {}", gpu.compute_capability));
                                ui.horizontal(|ui| {
                                    ui.label(format!("{}   ", gpu.temperature));
                                    ui.label(format!("{}   ", gpu.power_usage));
                                    ui.label(format!("Load: {}", gpu.utilization));
                                });
                            }
                            None => {
                                ui.colored_label(ui.style().visuals.error_fg_color, "No GPU detected");
                                ui.label("System will run in CPU-only mode");
                            }
                        }
                    });
                });
            });
            
            ui.add_space(10.0);
            
            // Control section
            ui.group(|ui| {
                ui.heading("Client Control");
                
                ui.horizontal(|ui| {
                    if self.client_process.is_some() {
                        if ui.button("Stop Client").clicked() {
                            self.stop_client();
                        }
                    } else {
                        if ui.button("Start Client").clicked() {
                            if let Err(e) = self.start_client() {
                                self.status = ClientStatus::Error(e);
                            }
                        }
                    }
                    
                    ui.label(&self.status.text());
                });
            });
            
            ui.add_space(10.0);
            
            // Log section
            ui.group(|ui| {
                ui.heading("Logs");
                
                let text_height = ui.available_height() - 40.0;
                
                egui::ScrollArea::vertical()
                    .max_height(text_height)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if let Ok(logs) = self.log_entries.lock() {
                            for entry in logs.iter() {
                                ui.horizontal(|ui| {
                                    // Timestamp in gray
                                    if let Some(ts) = &entry.timestamp {
                                        ui.label(RichText::new(format!("[{}]", ts)).color(Color32::from_gray(180)));
                                        ui.add_space(4.0);
                                    }

                                    // Log level with appropriate color
                                    if let Some(level) = &entry.level {
                                        let level_color = if entry.is_error {
                                            Color32::from_rgb(255, 100, 100)
                                        } else {
                                            Color32::from_rgb(100, 255, 100)
                                        };
                                        ui.label(RichText::new(level).color(level_color));
                                        ui.add_space(4.0);
                                    }

                                    // Module name in blue
                                    if let Some(module) = &entry.module {
                                        ui.label(RichText::new(module).color(Color32::from_rgb(100, 180, 255)));
                                        ui.add_space(4.0);
                                    }

                                    // Message in default color
                                    let message_color = if entry.is_error {
                                        Color32::from_rgb(255, 100, 100)
                                    } else {
                                        ui.style().visuals.text_color()
                                    };
                                    ui.label(RichText::new(&entry.message).color(message_color));
                                });
                            }
                        }
                    });
                
                if ui.button("Clear Logs").clicked() {
                    if let Ok(mut logs) = self.log_entries.lock() {
                        logs.clear();
                        logs.push(LogEntry {
                            timestamp: None,
                            level: None,
                            module: None,
                            message: "Logs cleared".to_string(),
                            is_error: false,
                        });
                    }
                }
            });
        });

        // Request repaint for real-time updates
        ctx.request_repaint_after(Duration::from_millis(500));
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(800.0, 600.0)),
        min_window_size: Some(egui::vec2(600.0, 400.0)),
        icon_data: load_icon(),
        ..Default::default()
    };
    eframe::run_native(
        "Cactus Desktop Client",
        options,
        Box::new(|_cc| Box::new(CactusClientApp::new())),
    )
}

fn load_icon() -> Option<IconData> {
    // Try to load the icon from the embedded data
    if let Ok(icon) = image::load_from_memory_with_format(ICON_DATA, image::ImageFormat::Ico) {
        let rgba = icon.to_rgba8();
        let (width, height) = rgba.dimensions();
        let rgba_data = rgba.into_raw();
        
        // Convert to PNG format
        let mut png_data = Vec::new();
        let mut encoder = image::codecs::png::PngEncoder::new(&mut png_data);
        if encoder.write_image(&rgba_data, width, height, image::ColorType::Rgba8).is_ok() {
            return IconData::try_from_png_bytes(&png_data).ok();
        }
    }
    None
}
