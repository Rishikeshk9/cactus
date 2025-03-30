import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

def run_command(cmd, cwd=None):
    try:
        subprocess.run(cmd, check=True, cwd=cwd, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}:")
        print(e.stderr)
        return False

def create_venv():
    print("Creating virtual environment...")
    if os.path.exists("venv"):
        shutil.rmtree("venv")
    if not run_command([sys.executable, "-m", "venv", "venv"]):
        print("Failed to create virtual environment")
        sys.exit(1)

def get_python_cmd():
    if platform.system() == "Windows":
        return str(Path("venv") / "Scripts" / "python.exe")
    return str(Path("venv") / "bin" / "python")

def install_packages():
    python_cmd = get_python_cmd()
    print("Upgrading pip...")
    run_command([python_cmd, "-m", "pip", "install", "--upgrade", "pip"])

    # First verify Python environment
    print("Verifying Python environment...")
    if not run_command([python_cmd, "-c", "import sys; print(sys.executable)"]):
        print("ERROR: Python environment is not properly set up")
        sys.exit(1)

    # Clear pip cache first
    print("Clearing pip cache...")
    run_command([python_cmd, "-m", "pip", "cache", "purge"])

    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA support...")
    torch_install_result = run_command([
        python_cmd,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--verbose",  # Added verbose flag to see more details
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu121",
        "torch",
        "torchvision",
        "torchaudio"
    ])

    if not torch_install_result:
        print("Failed to install PyTorch with CUDA support")
        # Print pip debug info
        run_command([python_cmd, "-m", "pip", "debug"])
        sys.exit(1)

    # Verify the installation immediately after
    print("Verifying PyTorch installation...")
    verify_cmd = [
        python_cmd,
        "-c",
        "import torch; print(f'PyTorch version: {torch.__version__}'); "
        "print(f'CUDA available: {torch.cuda.is_available()}'); "
        "print(f'CUDA version: {torch.version.cuda}')"
    ]
    
    if not run_command(verify_cmd):
        print("ERROR: PyTorch verification failed")
        print("Trying to diagnose the issue...")
        # Try to get more information about the Python environment
        run_command([python_cmd, "-c", "import sys; print('Python path:', sys.path)"])
        run_command([python_cmd, "-m", "pip", "list"])
        sys.exit(1)

    # Install other packages one by one
    other_packages = [
        "transformers==4.38.2",
        "diffusers==0.27.0",
        "pillow==10.2.0",
        "numpy==1.26.4",
        "accelerate==0.27.2"
    ]

    for package in other_packages:
        print(f"Installing {package}...")
        if not run_command([
            python_cmd,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            package
        ]):
            print(f"Failed to install {package}")
            sys.exit(1)

    print("All packages installed successfully!")

def copy_python_env():
    release_dir = Path("target") / "release"
    python_dir = release_dir / "python"
    
    print("Copying Python environment...")
    if python_dir.exists():
        shutil.rmtree(python_dir)
    
    # Create necessary directories
    os.makedirs(python_dir, exist_ok=True)
    
    venv_dir = Path("venv")
    if platform.system() == "Windows":
        # Copy Python DLLs and exe
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
    
    shutil.copytree(site_packages_src, site_packages_dst)

def copy_models_config():
    print("Copying models configuration...")
    release_dir = Path("target") / "release"
    models_config_dir = release_dir / "models_config"
    
    if not models_config_dir.exists():
        os.makedirs(models_config_dir)
    
    # Copy gpu_loadrust.py if it exists
    if os.path.exists("models_config/gpu_loadrust.py"):
        shutil.copy2("models_config/gpu_loadrust.py", models_config_dir)
    else:
        print("Warning: models_config/gpu_loadrust.py not found")

def main():
    print("Starting bundling process...")
    create_venv()
    install_packages()
    copy_python_env()
    copy_models_config()
    print("Bundling complete!")

if __name__ == "__main__":
    main() 