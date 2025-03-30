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

    # Try to detect CUDA availability
    cuda_available = False
    try:
        # First try to install torch CPU to check CUDA
        print("Installing PyTorch (CPU) to check CUDA...")
        if run_command([
            python_cmd, 
            "-m", 
            "pip", 
            "install", 
            "--no-cache-dir",
            "torch",  # Latest stable version
            "torchvision",
            "torchaudio"
        ]):
            import torch
            cuda_available = torch.cuda.is_available()
    except:
        pass

    print(f"CUDA detected: {cuda_available}")
    
    # If CUDA is available, reinstall PyTorch with CUDA support
    if cuda_available:
        print("Reinstalling PyTorch with CUDA support...")
        run_command([
            python_cmd,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torch",
            "torchvision",
            "torchaudio"
        ])
        if not run_command([
            python_cmd,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu121",
            "torch",  # Latest stable version with CUDA
            "torchvision",
            "torchaudio"
        ]):
            print("Failed to install PyTorch with CUDA support, falling back to CPU version...")
            cuda_available = False

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