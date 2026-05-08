"""
GTX 1660 Ti Optimized Training Setup Script

Automated setup and installation script for GTX 1660 Ti training framework.
Handles environment detection, dependency installation, and verification.

Author: AI Assistant
Created: 2024
"""

import os
import sys
import subprocess
import platform
import importlib
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GTX1660TiSetup:
    """Automated setup for GTX 1660 Ti training environment."""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.cuda_available = None
        self.cuda_version = None
        self.gpu_name = None
        
    def run_setup(self):
        """Run complete setup process."""
        print("🚀 GTX 1660 Ti Training Framework Setup")
        print("=" * 50)
        
        # Check system requirements
        if not self._check_system_requirements():
            return False
        
        # Detect GPU and CUDA
        self._detect_gpu_environment()
        
        # Install dependencies
        if not self._install_dependencies():
            return False
        
        # Verify installation
        if not self._verify_installation():
            return False
        
        # Create directories
        self._create_directories()
        
        # Run basic tests
        self._run_basic_tests()
        
        print("\n✅ Setup completed successfully!")
        print("🎯 You can now start training models on your GTX 1660 Ti!")
        print("\n📚 Next steps:")
        print("1. Check examples in ./training/examples/")
        print("2. Run benchmark: python training/benchmark.py --quick")
        print("3. Start training: python training/examples/resnet_training.py --demo")
        
        return True
    
    def _check_system_requirements(self) -> bool:
        """Check system requirements."""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if self.python_version < (3, 8):
            print(f"❌ Python {self.python_version.major}.{self.python_version.minor} is not supported")
            print("   Please install Python 3.8 or higher")
            return False
        
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
        # Check platform
        if self.platform not in ["Windows", "Linux", "Darwin"]:
            print(f"⚠️ Platform {self.platform} may not be fully supported")
        
        print(f"✅ Platform: {self.platform}")
        
        # Check available disk space
        current_dir = Path.cwd()
        free_space = shutil.disk_usage(current_dir).free / (1024**3)  # GB
        if free_space < 5:
            print(f"⚠️ Low disk space: {free_space:.1f} GB available")
        else:
            print(f"✅ Disk space: {free_space:.1f} GB available")
        
        return True
    
    def _detect_gpu_environment(self):
        """Detect GPU and CUDA environment."""
        print("\n🎮 Detecting GPU environment...")
        
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
            
            if self.cuda_available:
                self.cuda_version = torch.version.cuda
                self.gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                print(f"✅ CUDA available: {self.cuda_version}")
                print(f"✅ GPU: {self.gpu_name}")
                print(f"✅ VRAM: {gpu_memory:.1f} GB")
                
                # Check if it's GTX 1660 Ti
                if "1660" in self.gpu_name.upper():
                    print("🎯 GTX 1660 Ti detected - Perfect!")
                else:
                    print(f"⚠️ {self.gpu_name} detected - Framework optimized for GTX 1660 Ti")
            else:
                print("❌ CUDA not available")
                print("⚠️ Training will be very slow on CPU")
                
        except ImportError:
            print("❌ PyTorch not installed - will install during setup")
            self.cuda_available = False
    
    def _install_dependencies(self) -> bool:
        """Install required dependencies."""
        print("\n📦 Installing dependencies...")
        
        # Determine PyTorch version based on CUDA availability
        if self.cuda_available and self.cuda_version:
            if self.cuda_version.startswith("11.8"):
                pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            elif self.cuda_version.startswith("12.1"):
                pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            else:
                pytorch_cmd = "pip install torch torchvision torchaudio"
        else:
            # CPU-only version - use cpu-only index
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        
        # Install PyTorch
        print("🔥 Installing PyTorch...")
        if not self._run_command(pytorch_cmd):
            print("❌ Failed to install PyTorch")
            return False
        
        # Install other dependencies
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            print("📚 Installing other dependencies...")
            if not self._run_command(f"pip install -r {requirements_file}"):
                print("❌ Failed to install dependencies")
                return False
        
        return True
    
    def _verify_installation(self) -> bool:
        """Verify installation."""
        print("\n🔍 Verifying installation...")
        
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__}")
            
            # Test CUDA
            if torch.cuda.is_available():
                print(f"✅ CUDA {torch.version.cuda}")
                print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
                
                # Test basic CUDA operation
                try:
                    x = torch.randn(10, 10).cuda()
                    y = torch.matmul(x, x)
                    print("✅ CUDA operations working")
                except Exception as e:
                    print(f"❌ CUDA operation failed: {e}")
                    return False
            else:
                print("⚠️ CUDA not available - CPU only")
            
            # Test key dependencies
            dependencies = [
                ("numpy", "numpy"),
                ("pandas", "pandas"),
                ("tqdm", "tqdm"),
                ("transformers", "transformers"),
                ("peft", "peft"),
                ("bitsandbytes", "bitsandbytes")
            ]
            
            for name, import_name in dependencies:
                try:
                    importlib.import_module(import_name)
                    print(f"✅ {name}")
                except ImportError:
                    print(f"⚠️ {name} not available (optional)")
            
            return True
            
        except ImportError as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def _create_directories(self):
        """Create necessary directories."""
        print("\n📁 Creating directories...")
        
        directories = [
            "checkpoints",
            "benchmark_results",
            "logs",
            "data",
            "models"
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created {directory}/")
    
    def _run_basic_tests(self):
        """Run basic functionality tests."""
        print("\n🧪 Running basic tests...")
        
        try:
            import torch
            from gtx1660_optimizer import GTX1660Config, MemoryMonitor
            
            # Test configuration
            config = GTX1660Config()
            print("✅ GTX1660Config working")
            
            # Test memory monitor
            if torch.cuda.is_available():
                monitor = MemoryMonitor()
                info = monitor.get_memory_info()
                print(f"✅ Memory monitor working: {info['allocated_gb']:.2f}GB used")
            else:
                print("⚠️ Memory monitor skipped (no CUDA)")
            
            # Test basic model creation
            class TestModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 1)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = TestModel()
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Test forward pass
            x = torch.randn(5, 10)
            if torch.cuda.is_available():
                x = x.cuda()
            
            with torch.no_grad():
                output = model(x)
            
            print("✅ Basic model test passed")
            
        except Exception as e:
            print(f"⚠️ Basic test failed: {e}")
    
    def _run_command(self, command: str) -> bool:
        """Run shell command and return success status."""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return True
            else:
                print(f"❌ Command failed: {command}")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ Command timed out: {command}")
            return False
        except Exception as e:
            print(f"❌ Command error: {e}")
            return False


def main():
    """Main setup function."""
    setup = GTX1660TiSetup()
    
    try:
        success = setup.run_setup()
        
        if success:
            print("\n🎉 Setup completed successfully!")
            print("📖 Check README.md for usage instructions")
        else:
            print("\n❌ Setup failed. Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
