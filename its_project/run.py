#!/usr/bin/env python3
"""
Intelligent Trading System - Unified Launcher
============================================

This is the main entry point for the ITS application.
Use this script to launch different components of the trading system.

Usage:
    python run.py gui                    # Launch GUI application
    python run.py gui --demo             # Launch GUI in demo mode
    python run.py gui --async           # Launch GUI with async integration
    python run.py gui --realtime        # Launch GUI with real-time storage
    python run.py backend                # Start backend services
    python run.py data                    # Run data processing
    python run.py test                    # Run system tests
    python run.py check                   # System health check
    python run.py setup                   # Initial setup
"""

import sys
import os
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure logs directory exists
logs_dir = PROJECT_ROOT / 'logs'
logs_dir.mkdir(exist_ok=True)

# Configure logging using project module
try:
    from common.logging import configure_logging
    configure_logging(logging.INFO)
except ImportError:
    # Fallback logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / 'system.log', mode='a')
        ]
    )

logger = logging.getLogger(__name__)


class ITSLauncher:
    """Unified launcher for ITS components."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.gui_dir = self.project_root / 'gui'
        self.logs_dir = self.project_root / 'logs'
        self.data_dir = self.project_root / 'data'
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories."""
        for directory in [self.logs_dir, self.data_dir]:
            directory.mkdir(exist_ok=True)
    
    def run_gui(self, mode: str = 'production', demo: bool = False) -> int:
        """Launch GUI application."""
        logger.info(f"Starting GUI in {mode} mode (demo={demo})")
        
        # Determine which GUI to launch
        if mode == 'realtime':
            gui_script = self.gui_dir / 'realtime_main_window.py'
        elif mode == 'async':
            gui_script = self.gui_dir / 'async_main_window.py'
        elif mode == 'production':
            gui_script = self.gui_dir / 'production_main_window.py'
        elif mode == 'integrated':
            gui_script = self.gui_dir / 'integrated_main_window.py'
        else:
            gui_script = self.gui_dir / 'main_window.py'
        
        if not gui_script.exists():
            logger.error(f"GUI script not found: {gui_script}")
            return 1
        
        # Prepare command with proper Python path
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root) + os.pathsep + env.get('PYTHONPATH', '')
        
        cmd = [sys.executable, str(gui_script)]
        if demo:
            cmd.append('--demo')
        
        # Change to GUI directory
        original_cwd = os.getcwd()
        os.chdir(self.gui_dir)
        
        try:
            # Launch GUI with proper environment
            process = subprocess.Popen(cmd, env=env)
            return process.wait()
        except Exception as e:
            logger.error(f"Failed to start GUI: {e}")
            return 1
        finally:
            os.chdir(original_cwd)
    
    def run_backend(self) -> int:
        """Start backend services."""
        logger.info("Starting backend services")
        
        # This would start backend services
        # For now, just log that backend would start
        logger.info("Backend services placeholder - implement as needed")
        return 0
    
    def run_data_processing(self) -> int:
        """Run data processing pipeline."""
        logger.info("Starting data processing")
        
        # This would run data processing
        logger.info("Data processing placeholder - implement as needed")
        return 0
    
    def run_tests(self, test_type: str = 'all') -> int:
        """Run system tests."""
        logger.info(f"Running {test_type} tests")
        
        # Test commands
        test_commands = {
            'all': ['python', '-m', 'pytest', 'tests/', '-v'],
            'unit': ['python', '-m', 'pytest', 'tests/unit/', '-v'],
            'integration': ['python', '-m', 'pytest', 'tests/integration/', '-v'],
            'gui': ['python', '-m', 'pytest', 'tests/gui/', '-v']
        }
        
        cmd = test_commands.get(test_type, test_commands['all'])
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return 1
    
    def run_health_check(self) -> int:
        """Run system health check."""
        logger.info("Running system health check")
        
        # Import health check
        try:
            from scripts.health_check import SystemHealthChecker
            checker = SystemHealthChecker()
            return checker.run_all_checks()
        except ImportError:
            logger.warning("Health check module not found, running basic checks")
            return self._basic_health_check()
    
    def _basic_health_check(self) -> int:
        """Basic health check without external dependencies."""
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
        
        # Check directories
        required_dirs = ['gui', 'storage', 'models', 'features', 'decision']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                issues.append(f"Missing directory: {dir_name}")
        
        # Check key files
        key_files = [
            'gui/main_window.py',
            'storage/data_loader.py',
            'models/base.py',
            'decision/signal_generator.py'
        ]
        
        for file_path in key_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                issues.append(f"Missing file: {file_path}")
        
        # Report results
        if issues:
            logger.error("Health check failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return 1
        else:
            logger.info("Basic health check passed")
            return 0
    
    def run_setup(self) -> int:
        """Run initial setup."""
        logger.info("Running initial setup")
        
        # Create directories
        directories = [
            'logs',
            'data/parquet',
            'data/raw',
            'models/saved',
            'cache',
            'tests/unit',
            'tests/integration',
            'tests/gui',
            'scripts'
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Create basic config files
        self._create_basic_configs()
        
        # Test setup
        logger.info("Testing setup...")
        test_result = self._test_setup()
        
        if test_result == 0:
            logger.info("✅ Initial setup completed successfully!")
            logger.info("You can now run: python run.py gui")
        else:
            logger.warning("⚠️ Setup completed with warnings")
        
        return test_result
    
    def _test_setup(self) -> int:
        """Test that setup was successful."""
        issues = []
        
        # Check GUI files exist
        gui_files = [
            'gui/main_window.py',
            'gui/production_main_window.py',
            'gui/async_main_window.py',
            'gui/realtime_main_window.py',
            'gui/integrated_main_window.py'
        ]
        
        for gui_file in gui_files:
            full_path = self.project_root / gui_file
            if not full_path.exists():
                issues.append(f"Missing GUI file: {gui_file}")
        
        # Check common module
        common_files = ['common/__init__.py', 'common/config.py', 'common/logging.py']
        for common_file in common_files:
            full_path = self.project_root / common_file
            if not full_path.exists():
                issues.append(f"Missing common file: {common_file}")
        
        if issues:
            logger.warning("Setup test found issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return 1
        else:
            logger.info("✅ All critical files present")
            return 0
    
    def _create_basic_configs(self):
        """Create basic configuration files."""
        # Create requirements.txt
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            requirements_content = """
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# GUI
PyQt6>=6.0.0

# Storage
pyarrow>=6.0.0
psycopg2-binary>=2.9.0
asyncpg>=0.24.0

# ML
torch>=1.9.0
tensorflow>=2.6.0

# Testing
pytest>=6.2.0
pytest-qt>=4.0.0

# Development
black>=21.0.0
flake8>=3.9.0
"""
            requirements_file.write_text(requirements_content.strip())
            logger.info("Created requirements.txt")
        
        # Create basic config
        config_file = self.project_root / 'config.yaml'
        if not config_file.exists():
            config_content = """
# ITS Configuration
database:
  timescale_dsn: "postgresql://user:pass@localhost/its"
  parquet_path: "data/parquet"

trading:
  initial_balance: 10000.0
  commission_rate: 0.001
  max_position_size: 0.1

models:
  default_model: "GRU-LSTM"
  model_weights: [0.4, 0.3, 0.3]

logging:
  level: "INFO"
  file: "logs/system.log"
"""
            config_file.write_text(config_content.strip())
            logger.info("Created config.yaml")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Trading System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py setup                    # Initial setup
  python run.py gui                      # Launch production GUI
  python run.py gui --mode async         # Launch async GUI
  python run.py gui --demo               # Launch GUI in demo mode
  python run.py check                    # System health check
  python run.py test --type gui          # Run GUI tests
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands', metavar='COMMAND')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Run initial setup and create directories')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI application')
    gui_parser.add_argument('--mode', choices=['basic', 'integrated', 'production', 'async', 'realtime'],
                           default='production', help='GUI mode (default: production)')
    gui_parser.add_argument('--demo', action='store_true', help='Run in demo mode with sample data')
    
    # Backend command
    backend_parser = subparsers.add_parser('backend', help='Start backend services')
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Run data processing pipeline')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    test_parser.add_argument('--type', choices=['all', 'unit', 'integration', 'gui'],
                            default='all', help='Test type (default: all)')
    
    # Health check command
    check_parser = subparsers.add_parser('check', help='Run system health check')
    
    # Parse arguments
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        logger.error(f"Argument error: {e}")
        parser.print_help()
        return 1
    
    if not args.command:
        print("Error: No command specified")
        parser.print_help()
        return 1
    
    # Create launcher
    launcher = ITSLauncher()
    
    # Execute command
    try:
        if args.command == 'gui':
            return launcher.run_gui(mode=args.mode, demo=args.demo)
        elif args.command == 'backend':
            return launcher.run_backend()
        elif args.command == 'data':
            return launcher.run_data_processing()
        elif args.command == 'test':
            return launcher.run_tests(test_type=args.type)
        elif args.command == 'check':
            return launcher.run_health_check()
        elif args.command == 'setup':
            return launcher.run_setup()
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
