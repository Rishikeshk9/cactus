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

def get_wheel_dir():
    return Path("wheels_cache")

def check_cached_wheel(package, version):
    wheel_dir = get_wheel_dir()
    if wheel_dir.exists():
        # Look for transformers wheel file matching the version
        for wheel in wheel_dir.glob(f"{package}-{version}*.whl"):
            return wheel
    return None

def cache_wheel(python_cmd, package, version):
    wheel_dir = get_wheel_dir()
    wheel_dir.mkdir(exist_ok=True)
    
    print(f"Building and caching wheel for {package}=={version}")
    print("This may take several minutes but only needs to be done once...")
    
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C codegen-units=8"
    env["CARGO_BUILD_JOBS"] = str(os.cpu_count() or 4)
    
    # Build wheel
    if not run_command([
        python_cmd,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        str(wheel_dir),
        f"{package}=={version}"
    ], env=env):
        return None
        
    # Return the cached wheel path
    return check_cached_wheel(package, version)

def check_package_version(python_cmd, package):
    try:
        result = subprocess.run(
            [python_cmd, "-c", f"import {package}; print({package}.__version__)"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None

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
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
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
        print("3. CUDA 11.8 or later is installed")
        sys.exit(1)
    
    print("✓ CUDA support verified")

    # Install other packages one by one with Rust requirement warning
    other_packages = [
        ("transformers", "4.38.2", "This package requires Rust compilation and may take several minutes..."),
        ("diffusers", "0.27.0", "Installing diffusers..."),
        ("numpy", "1.26.4", "Installing NumPy..."),
        ("pillow", None, "Installing Pillow for image processing..."),
        ("accelerate", "1.5.2", "Installing Accelerate for better performance...")
    ]

    print("\nChecking and installing additional packages...")
    print("Note: Some packages require Rust compilation which may take several minutes")
    
    for package, version, message in other_packages:
        # Special handling for Pillow which has a different import name
        import_name = "PIL" if package == "pillow" else package
        
        # Check if package is already installed
        current_version = check_package_version(python_cmd, import_name)
        if version is None:
            if current_version:
                print(f"\n✓ {package} {current_version} is already installed")
                continue
        elif current_version == version:
            print(f"\n✓ {package} {version} is already installed")
            continue
        elif current_version:
            print(f"\nUpdating {package} from version {current_version} to {version}")
            # For numpy, we need to uninstall first to avoid version conflicts
            if package == "numpy":
                print(f"Uninstalling existing numpy version {current_version}...")
                if not run_command([
                    python_cmd,
                    "-m",
                    "pip",
                    "uninstall",
                    "-y",
                    "numpy"
                ]):
                    print("Failed to uninstall numpy")
                    sys.exit(1)
                print("✓ Existing numpy uninstalled")
        else:
            print(f"\n{message}")
            if version:
                print(f"Installing {package}=={version}...")
            else:
                print(f"Installing {package}...")

        if package == "transformers":
            # Check for cached wheel
            wheel_path = check_cached_wheel(package, version)
            if not wheel_path:
                # Build and cache wheel if not found
                wheel_path = cache_wheel(python_cmd, package, version)
                if not wheel_path:
                    print("Failed to build transformers wheel")
                    sys.exit(1)
            
            # Install from cached wheel
            print(f"Installing {package} from cached wheel...")
            if not run_command([
                python_cmd,
                "-m",
                "pip",
                "install",
                str(wheel_path)
            ]):
                print("Failed to install from cached wheel")
                sys.exit(1)
            print(f"✓ {package} installed successfully")
            continue
        
        # For other packages, install normally
        cmd = [
            python_cmd,
            "-m",
            "pip",
            "install",
            "--no-cache-dir"
        ]
        
        # Only use --no-deps for packages that don't need their dependencies
        if package not in ["pillow", "transformers", "numpy"]:
            cmd.append("--no-deps")
            
        if version:
            cmd.append(f"{package}=={version}")
        else:
            cmd.append(package)
        
        if not run_command(cmd):
            print(f"\nERROR: Failed to install {package}")
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