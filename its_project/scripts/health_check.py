#!/usr/bin/env python3
"""
ITS System Health Check Utility
==============================

Comprehensive health checking for all ITS system components.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class SystemHealthChecker:
    """Comprehensive system health checker."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.results = []
        self.issues = []
        
    def run_all_checks(self) -> int:
        """Run all health checks."""
        logger.info("Starting comprehensive system health check")
        
        # Core checks
        self.check_python_environment()
        self.check_project_structure()
        self.check_dependencies()
        self.check_gui_components()
        self.check_backend_components()
        self.check_storage_components()
        self.check_model_components()
        self.check_configuration()
        
        # Report results
        return self.generate_report()
    
    def check_python_environment(self) -> bool:
        """Check Python environment."""
        logger.info("Checking Python environment...")
        
        issues = []
        
        # Python version
        if sys.version_info < (3, 8):
            issues.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check if running in virtual environment
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            issues.append("Virtual environment recommended but not detected")
        
        # Check key modules
        required_modules = ['numpy', 'pandas', 'asyncio']
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                issues.append(f"Required module not found: {module}")
        
        self._add_result("Python Environment", len(issues) == 0, issues)
        return len(issues) == 0
    
    def check_project_structure(self) -> bool:
        """Check project structure."""
        logger.info("Checking project structure...")
        
        issues = []
        
        # Required directories
        required_dirs = [
            'gui', 'storage', 'models', 'features', 'decision', 
            'execution', 'metalearning', 'backtesting', 'data_layer', 'common'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                issues.append(f"Missing directory: {dir_name}")
            elif not dir_path.is_dir():
                issues.append(f"Path is not directory: {dir_name}")
        
        # Required files
        required_files = [
            'run.py',
            'gui/__init__.py',
            'gui/main_window.py',
            'storage/data_loader.py',
            'models/base.py',
            'decision/signal_generator.py',
            'execution/trader.py'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                issues.append(f"Missing file: {file_path}")
            elif not full_path.is_file():
                issues.append(f"Path is not file: {file_path}")
        
        self._add_result("Project Structure", len(issues) == 0, issues)
        return len(issues) == 0
    
    def check_dependencies(self) -> bool:
        """Check Python dependencies."""
        logger.info("Checking dependencies...")
        
        issues = []
        
        # Core dependencies
        core_deps = [
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('PyQt6', 'PyQt6.QtCore'),
            ('pyarrow', 'pyarrow'),
            ('sklearn', 'sklearn')
        ]
        
        for dep_name, import_name in core_deps:
            try:
                importlib.import_module(import_name)
            except ImportError:
                issues.append(f"Missing dependency: {dep_name}")
        
        # Optional dependencies
        optional_deps = [
            ('torch', 'torch'),
            ('tensorflow', 'tensorflow'),
            ('psycopg2', 'psycopg2'),
            ('asyncpg', 'asyncpg')
        ]
        
        for dep_name, import_name in optional_deps:
            try:
                importlib.import_module(import_name)
            except ImportError:
                issues.append(f"Optional dependency missing: {dep_name} (may affect functionality)")
        
        self._add_result("Dependencies", len([i for i in issues if 'Optional' not in i]) == 0, issues)
        return len([i for i in issues if 'Optional' not in i]) == 0
    
    def check_gui_components(self) -> bool:
        """Check GUI components."""
        logger.info("Checking GUI components...")
        
        issues = []
        
        # GUI files
        gui_files = [
            'gui/main_window.py',
            'gui/components/__init__.py',
            'gui/components/sidebar.py',
            'gui/components/main_chart.py',
            'gui/components/signal_panel.py',
            'gui/components/model_panel.py',
            'gui/components/bottom_panel.py',
            'gui/styles/dark_theme.py'
        ]
        
        for file_path in gui_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                issues.append(f"Missing GUI component: {file_path}")
        
        # GUI integration files
        gui_integration_files = [
            'gui/data_adapters.py',
            'gui/format_validators.py',
            'gui/backend_bridge.py',
            'gui/real_backend_bridge.py',
            'gui/realtime_backend_bridge.py',
            'gui/async_integration.py'
        ]
        
        for file_path in gui_integration_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                issues.append(f"Missing GUI integration: {file_path}")
        
        # Try to import GUI modules
        try:
            sys.path.insert(0, str(self.project_root / 'gui'))
            import main_window
            import components.sidebar
            import components.main_chart
        except ImportError as e:
            issues.append(f"GUI import error: {e}")
        finally:
            if str(self.project_root / 'gui') in sys.path:
                sys.path.remove(str(self.project_root / 'gui'))
        
        self._add_result("GUI Components", len(issues) == 0, issues)
        return len(issues) == 0
    
    def check_backend_components(self) -> bool:
        """Check backend components."""
        logger.info("Checking backend components...")
        
        issues = []
        
        # Backend files
        backend_files = [
            'decision/signal_generator.py',
            'decision/decision.py',
            'execution/trader.py',
            'execution/base.py',
            'meta/model_selector.py',
            'features/feature_builder.py',
            'features/synchronizer.py'
        ]
        
        for file_path in backend_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                issues.append(f"Missing backend component: {file_path}")
        
        # Try to import backend modules
        try:
            from decision.signal_generator import SignalGenerator
            from execution.trader import PaperTrader
            from meta.model_selector import ModelSelector
        except ImportError as e:
            issues.append(f"Backend import error: {e}")
        
        self._add_result("Backend Components", len(issues) == 0, issues)
        return len(issues) == 0
    
    def check_storage_components(self) -> bool:
        """Check storage components."""
        logger.info("Checking storage components...")
        
        issues = []
        
        # Storage files
        storage_files = [
            'storage/data_loader.py',
            'storage/parquet_store.py',
            'storage/timescale_client.py'
        ]
        
        for file_path in storage_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                issues.append(f"Missing storage component: {file_path}")
        
        # Try to import storage modules
        try:
            from storage.data_loader import DataLoader
        except ImportError as e:
            issues.append(f"Storage import error: {e}")
        
        # Check data directories
        data_dirs = ['data', 'data/parquet', 'data/raw']
        for dir_path in data_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                issues.append(f"Missing data directory: {dir_path}")
        
        self._add_result("Storage Components", len(issues) == 0, issues)
        return len(issues) == 0
    
    def check_model_components(self) -> bool:
        """Check model components."""
        logger.info("Checking model components...")
        
        issues = []
        
        # Model files
        model_files = [
            'models/base.py',
            'models/gru_model.py',
            'models/cnn_lob_model.py',
            'models/boosting_model.py'
        ]
        
        for file_path in model_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                issues.append(f"Missing model component: {file_path}")
        
        # Try to import model modules
        try:
            from models.base import BaseModel
            from models.gru_model import GRUModel
        except ImportError as e:
            issues.append(f"Model import error: {e}")
        
        # Check models directory
        models_dir = self.project_root / 'models' / 'saved'
        if not models_dir.exists():
            issues.append("Models saved directory missing (create with: mkdir -p models/saved)")
        
        self._add_result("Model Components", len(issues) == 0, issues)
        return len(issues) == 0
    
    def check_configuration(self) -> bool:
        """Check configuration."""
        logger.info("Checking configuration...")
        
        issues = []
        
        # Config files
        config_files = [
            'config.yaml',
            'requirements.txt',
            '.env'
        ]
        
        for file_path in config_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                issues.append(f"Missing config file: {file_path}")
        
        # Check logs directory
        logs_dir = self.project_root / 'logs'
        if not logs_dir.exists():
            issues.append("Logs directory missing")
        
        # Check cache directory
        cache_dir = self.project_root / 'cache'
        if not cache_dir.exists():
            issues.append("Cache directory missing")
        
        self._add_result("Configuration", len(issues) == 0, issues)
        return len(issues) == 0
    
    def _add_result(self, category: str, passed: bool, issues: List[str]):
        """Add check result."""
        self.results.append({
            'category': category,
            'passed': passed,
            'issues': issues
        })
        
        if not passed and issues:
            self.issues.extend([f"{category}: {issue}" for issue in issues])
    
    def generate_report(self) -> int:
        """Generate health check report."""
        print("\n" + "="*60)
        print("ITS SYSTEM HEALTH CHECK REPORT")
        print("="*60)
        
        passed_count = 0
        total_count = len(self.results)
        
        for result in self.results:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"\n{result['category']}: {status}")
            
            if result['issues']:
                for issue in result['issues']:
                    print(f"  - {issue}")
            else:
                print("  All checks passed")
            
            if result['passed']:
                passed_count += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {passed_count}/{total_count} checks passed")
        
        if passed_count == total_count:
            print("All systems healthy! Ready to run.")
            return_code = 0
        else:
            print(f"Found {len(self.issues)} issues that need attention.")
            print("Run 'python run.py setup' to fix missing directories.")
            return_code = 1
        
        print("="*60)
        return return_code


def main():
    """Main entry point."""
    checker = SystemHealthChecker()
    return checker.run_all_checks()


if __name__ == '__main__':
    sys.exit(main())
