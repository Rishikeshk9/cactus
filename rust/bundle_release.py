import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

def run_command(cmd, cwd=None, env=None):
    try:
        subprocess.run(cmd, check=True, cwd=cwd, capture_output=True, text=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}:")
        print(e.stderr)
        return False

def check_rust_installation():
    print("Checking for Rust installation...")
    if not run_command(["rustc", "--version"]):
        print("\nERROR: Rust is not installed or not in PATH")
        print("Please install Rust:")
        print("1. Visit https://rustup.rs/")
        print("2. On Windows, download and run rustup-init.exe from https://win.rustup.rs/")
        print("3. After installation, close and reopen your terminal")
        print("\nInstallation failed. Please install Rust and try again.")
        sys.exit(1)
    print("✓ Rust installation found!")

def create_venv():
    print("Creating virtual environment...")
    if os.path.exists("venv"):
        shutil.rmtree("venv")
    if not run_command([sys.executable, "-m", "venv", "venv"]):
        print("Failed to create virtual environment")
        sys.exit(1)
    print("✓ Virtual environment created")

def get_python_cmd():
    if platform.system() == "Windows":
        return str(Path("venv") / "Scripts" / "python.exe")
    return str(Path("venv") / "bin" / "python")

def check_pytorch_cuda():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        print(f"Found existing PyTorch installation:")
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {cuda_available}")
        print(f"✓ CUDA version: {cuda_version}")
        return True if cuda_available else False
    except ImportError:
        print("PyTorch not found, will install...")
        return None
    except Exception as e:
        print(f"Error checking PyTorch: {e}")
        return None

def install_packages():
    python_cmd = get_python_cmd()
    
    print("Upgrading pip...")
    run_command([python_cmd, "-m", "pip", "install", "--upgrade", "pip"])
    print("✓ Pip upgraded")

    # First verify Python environment
    print("Verifying Python environment...")
    if not run_command([python_cmd, "-c", "import sys; print(sys.executable)"]):
        print("ERROR: Python environment is not properly set up")
        sys.exit(1)
    print("✓ Python environment verified")

    # Check existing PyTorch installation
    print("\nChecking PyTorch installation...")
    pytorch_status = check_pytorch_cuda()
    
    if pytorch_status is None:
        # Clear pip cache first
        print("Clearing pip cache...")
        run_command([python_cmd, "-m", "pip", "cache", "purge"])
        print("✓ Pip cache cleared")

        # Install PyTorch with CUDA support
        print("Installing PyTorch with CUDA support...")
        torch_install_result = run_command([
            python_cmd,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--verbose",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu121",
            "torch",
            "torchvision",
            "torchaudio"
        ])

        if not torch_install_result:
            print("Failed to install PyTorch with CUDA support")
            run_command([python_cmd, "-m", "pip", "debug"])
            sys.exit(1)
            
        # Verify CUDA is available after fresh install
        pytorch_status = check_pytorch_cuda()
        
    if not pytorch_status:
        print("\nERROR: CUDA verification failed")
        print("Please ensure:")
        print("1. You have an NVIDIA GPU")
        print("2. NVIDIA drivers are up to date")
        print("3. CUDA 12.1 or later is installed")
        sys.exit(1)
    
    print("✓ CUDA support verified")

    # Install other packages one by one with Rust requirement warning
    other_packages = [
        ("transformers", "4.38.2", "This package requires Rust compilation and may take several minutes..."),
        ("diffusers", "0.27.0", "Installing diffusers..."),
        ("pillow", "10.2.0", "Installing Pillow for image processing..."),
        ("numpy", "1.26.4", "Installing NumPy..."),
        ("accelerate", "0.27.2", "Installing Accelerate for better performance...")
    ]

    print("\nInstalling additional packages...")
    print("Note: Some packages require Rust compilation which may take several minutes")
    
    for package, version, message in other_packages:
        print(f"\n{message}")
        print(f"Installing {package}=={version}...")
        
        # Use multiple workers for faster compilation
        env = os.environ.copy()
        if package == "transformers":
            # Use more workers for Rust compilation
            env["RUSTFLAGS"] = "-C codegen-units=8"
            # Use cargo parallel compilation
            env["CARGO_BUILD_JOBS"] = str(os.cpu_count() or 4)
        
        # Build the command list properly
        cmd = [
            python_cmd,
            "-m",
            "pip",
            "install",
            "--no-cache-dir"
        ]
        
        # Only add --no-deps for packages other than transformers
        if package != "transformers":
            cmd.append("--no-deps")
            
        cmd.append(f"{package}=={version}")
        
        if not run_command(cmd, env=env):
            print(f"\nERROR: Failed to install {package}")
            if package == "transformers":
                print("\nFor transformers package failure:")
                print("1. Ensure Rust is properly installed")
                print("2. Try running 'rustc --version' to verify Rust installation")
                print("3. If Rust is missing, install it from https://rustup.rs/")
                print("\nNote: Compilation can take 5-10 minutes depending on your system")
            sys.exit(1)
        print(f"✓ {package} installed successfully")

    print("\n✓ All packages installed successfully!")

def copy_python_env():
    release_dir = Path("target") / "release"
    python_dir = release_dir / "python"
    
    print("\nCopying Python environment...")
    if python_dir.exists():
        shutil.rmtree(python_dir)
    
    # Create necessary directories
    os.makedirs(python_dir, exist_ok=True)
    
    venv_dir = Path("venv")
    if platform.system() == "Windows":
        # Copy Python DLLs and exe
        print("Copying Python runtime files...")
        for file in (venv_dir / "Scripts").glob("python*.dll"):
            shutil.copy2(file, python_dir)
        shutil.copy2(venv_dir / "Scripts" / "python.exe", python_dir)
        
        # Copy site-packages
        site_packages_src = venv_dir / "Lib" / "site-packages"
        site_packages_dst = python_dir / "site-packages"
    else:
        # Copy Python binary and libs
        lib_dir = venv_dir / "lib"
        python_version = next(lib_dir.glob("python*"))
        shutil.copytree(python_version, python_dir / "lib" / python_version.name)
        
        # Copy site-packages
        site_packages_src = python_version / "site-packages"
        site_packages_dst = python_dir / "site-packages"
    
    print("Copying site-packages...")
    shutil.copytree(site_packages_src, site_packages_dst)
    print("✓ Python environment copied")

def copy_models_config():
    print("\nCopying models configuration...")
    release_dir = Path("target") / "release"
    models_config_dir = release_dir / "models_config"
    
    if not models_config_dir.exists():
        os.makedirs(models_config_dir)
    
    # Copy gpu_loadrust.py if it exists
    if os.path.exists("models_config/gpu_loadrust.py"):
        shutil.copy2("models_config/gpu_loadrust.py", models_config_dir)
        print("✓ Models configuration copied")
    else:
        print("Warning: models_config/gpu_loadrust.py not found")

def main():
    print("\n=== Starting DeAI Client Bundling Process ===\n")
    try:
        check_rust_installation()
        create_venv()
        install_packages()
        copy_python_env()
        copy_models_config()
        print("\n=== Bundling completed successfully! ===")
        print("\nYou can now run the client using run_client.bat")
    except KeyboardInterrupt:
        print("\n\nBundling process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during bundling process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 