#!/usr/bin/env python3
"""
ITS System Launcher - Advanced Launching Utility
===============================================

Advanced launcher with additional features for system management,
monitoring, and deployment.
"""

import sys
import os
import subprocess
import time
import signal
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import psutil

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class LaunchConfig:
    """Configuration for system launch."""
    component: str
    mode: str
    demo: bool = False
    monitor: bool = False
    log_level: str = "INFO"
    timeout: Optional[int] = None
    env_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}


class ProcessManager:
    """Manage running processes."""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.configs: Dict[str, LaunchConfig] = {}
        
    def start_process(self, name: str, config: LaunchConfig) -> bool:
        """Start a process with configuration."""
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(config.env_vars)
            
            # Prepare command
            if config.component == 'gui':
                cmd = self._get_gui_command(config)
            elif config.component == 'backend':
                cmd = self._get_backend_command(config)
            elif config.component == 'data':
                cmd = self._get_data_command(config)
            else:
                logger.error(f"Unknown component: {config.component}")
                return False
            
            # Start process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[name] = process
            self.configs[name] = config
            
            logger.info(f"Started process {name}: {' '.join(cmd)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start process {name}: {e}")
            return False
    
    def _get_gui_command(self, config: LaunchConfig) -> List[str]:
        """Get GUI launch command."""
        gui_dir = PROJECT_ROOT / 'gui'
        
        # Determine GUI script
        if config.mode == 'realtime':
            script = 'realtime_main_window.py'
        elif config.mode == 'async':
            script = 'async_main_window.py'
        elif config.mode == 'production':
            script = 'production_main_window.py'
        elif config.mode == 'integrated':
            script = 'integrated_main_window.py'
        else:
            script = 'main_window.py'
        
        cmd = [sys.executable, str(gui_dir / script)]
        
        if config.demo:
            cmd.append('--demo')
        
        return cmd
    
    def _get_backend_command(self, config: LaunchConfig) -> List[str]:
        """Get backend launch command."""
        return [sys.executable, '-m', 'its_project.backend.main']
    
    def _get_data_command(self, config: LaunchConfig) -> List[str]:
        """Get data processing command."""
        return [sys.executable, '-m', 'its_project.data_pipeline.main']
    
    def stop_process(self, name: str) -> bool:
        """Stop a process."""
        if name not in self.processes:
            logger.warning(f"Process {name} not found")
            return False
        
        process = self.processes[name]
        
        try:
            # Try graceful termination
            process.terminate()
            
            # Wait for termination
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                process.kill()
                process.wait()
            
            del self.processes[name]
            del self.configs[name]
            
            logger.info(f"Stopped process {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop process {name}: {e}")
            return False
    
    def is_running(self, name: str) -> bool:
        """Check if process is running."""
        if name not in self.processes:
            return False
        
        process = self.processes[name]
        return process.poll() is None
    
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all processes."""
        status = {}
        
        for name, process in self.processes.items():
            config = self.configs[name]
            
            # Get process info
            try:
                psutil_process = psutil.Process(process.pid)
                cpu_percent = psutil_process.cpu_percent()
                memory_info = psutil_process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
            except:
                cpu_percent = 0
                memory_mb = 0
            
            status[name] = {
                'running': self.is_running(name),
                'component': config.component,
                'mode': config.mode,
                'demo': config.demo,
                'pid': process.pid,
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'start_time': time.time()
            }
        
        return status
    
    def stop_all(self):
        """Stop all processes."""
        for name in list(self.processes.keys()):
            self.stop_process(name)


class SystemMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[Dict] = []
        self.max_history = 1000
    
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent,
                    'network_io': psutil.net_io_counters()._asdict()
                }
                
                # Add to history
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def get_current_metrics(self) -> Dict:
        """Get current system metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict()
        }
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict]:
        """Get metrics history."""
        return self.metrics_history[-limit:]


class AdvancedLauncher:
    """Advanced system launcher with monitoring and management."""
    
    def __init__(self):
        self.process_manager = ProcessManager()
        self.system_monitor = SystemMonitor()
        self.launch_history: List[Dict] = []
        
    def launch_system(self, configs: List[LaunchConfig]) -> bool:
        """Launch multiple system components."""
        logger.info(f"Launching {len(configs)} system components")
        
        # Start monitoring
        self.system_monitor.start_monitoring()
        
        # Launch processes
        success_count = 0
        for i, config in enumerate(configs):
            process_name = f"{config.component}_{i}"
            
            if self.process_manager.start_process(process_name, config):
                success_count += 1
                
                # Record launch
                self.launch_history.append({
                    'timestamp': time.time(),
                    'component': config.component,
                    'mode': config.mode,
                    'process_name': process_name,
                    'success': True
                })
            else:
                self.launch_history.append({
                    'timestamp': time.time(),
                    'component': config.component,
                    'mode': config.mode,
                    'process_name': process_name,
                    'success': False
                })
        
        logger.info(f"Successfully launched {success_count}/{len(configs)} components")
        return success_count == len(configs)
    
    def stop_system(self):
        """Stop all system components."""
        logger.info("Stopping system components")
        
        # Stop monitoring
        self.system_monitor.stop_monitoring()
        
        # Stop processes
        self.process_manager.stop_all()
        
        logger.info("System stopped")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            'processes': self.process_manager.get_status(),
            'system_metrics': self.system_monitor.get_current_metrics(),
            'launch_history': self.launch_history[-10:],  # Last 10 launches
            'uptime': time.time() - (self.launch_history[0]['timestamp'] if self.launch_history else 0)
        }
    
    def launch_preset(self, preset_name: str) -> bool:
        """Launch a predefined system preset."""
        presets = {
            'basic': [
                LaunchConfig('gui', 'basic', demo=True)
            ],
            'development': [
                LaunchConfig('gui', 'integrated', demo=False),
                LaunchConfig('backend', 'development')
            ],
            'production': [
                LaunchConfig('gui', 'production', demo=False, monitor=True),
                LaunchConfig('backend', 'production')
            ],
            'full': [
                LaunchConfig('gui', 'realtime', demo=False, monitor=True),
                LaunchConfig('backend', 'production'),
                LaunchConfig('data', 'production')
            ]
        }
        
        if preset_name not in presets:
            logger.error(f"Unknown preset: {preset_name}")
            return False
        
        configs = presets[preset_name]
        logger.info(f"Launching preset: {preset_name}")
        
        return self.launch_system(configs)
    
    def save_status(self, filepath: str):
        """Save system status to file."""
        status = self.get_system_status()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2, default=str)
            
            logger.info(f"Status saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    def load_status(self, filepath: str) -> Dict:
        """Load system status from file."""
        try:
            with open(filepath, 'r') as f:
                status = json.load(f)
            
            logger.info(f"Status loaded from {filepath}")
            return status
            
        except Exception as e:
            logger.error(f"Failed to load status: {e}")
            return {}


def create_preset_configs() -> Dict[str, List[LaunchConfig]]:
    """Create preset configurations."""
    return {
        'demo': [
            LaunchConfig('gui', 'basic', demo=True, monitor=False)
        ],
        'development': [
            LaunchConfig('gui', 'integrated', demo=False, monitor=True),
            LaunchConfig('backend', 'development', monitor=True)
        ],
        'testing': [
            LaunchConfig('gui', 'production', demo=True, monitor=True),
            LaunchConfig('backend', 'testing', monitor=True)
        ],
        'production': [
            LaunchConfig('gui', 'realtime', demo=False, monitor=True),
            LaunchConfig('backend', 'production', monitor=True),
            LaunchConfig('data', 'production', monitor=True)
        ]
    }


def main():
    """Main entry point for advanced launcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ITS Advanced System Launcher")
    parser.add_argument('--preset', choices=['demo', 'development', 'testing', 'production'],
                       help='Launch preset configuration')
    parser.add_argument('--component', choices=['gui', 'backend', 'data'],
                       help='Component to launch')
    parser.add_argument('--mode', choices=['basic', 'integrated', 'production', 'async', 'realtime'],
                       help='Launch mode')
    parser.add_argument('--demo', action='store_true', help='Demo mode')
    parser.add_argument('--monitor', action='store_true', help='Enable monitoring')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--stop', action='store_true', help='Stop all processes')
    parser.add_argument('--save-status', help='Save status to file')
    parser.add_argument('--load-status', help='Load status from file')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    launcher = AdvancedLauncher()
    
    try:
        if args.stop:
            launcher.stop_system()
            return 0
        
        if args.status:
            status = launcher.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            return 0
        
        if args.save_status:
            launcher.save_status(args.save_status)
            return 0
        
        if args.load_status:
            loaded_status = launcher.load_status(args.load_status)
            print(json.dumps(loaded_status, indent=2, default=str))
            return 0
        
        # Determine launch configuration
        configs = []
        
        if args.preset:
            preset_configs = create_preset_configs()
            if args.preset in preset_configs:
                configs = preset_configs[args.preset]
            else:
                logger.error(f"Unknown preset: {args.preset}")
                return 1
        else:
            # Build config from arguments
            if not args.component:
                logger.error("Component or preset required")
                return 1
            
            mode = args.mode or 'basic'
            config = LaunchConfig(
                component=args.component,
                mode=mode,
                demo=args.demo,
                monitor=args.monitor
            )
            configs = [config]
        
        # Launch system
        if launcher.launch_system(configs):
            print("System launched successfully")
            
            # Keep running if monitoring
            if any(config.monitor for config in configs):
                try:
                    while True:
                        time.sleep(1)
                        status = launcher.get_system_status()
                        
                        # Check if all processes are still running
                        running_processes = sum(1 for p in status['processes'].values() if p['running'])
                        if running_processes == 0:
                            logger.info("All processes stopped")
                            break
                        
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
            
            return 0
        else:
            logger.error("Failed to launch system")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        launcher.stop_system()
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        launcher.stop_system()
        return 1
    
    finally:
        launcher.stop_system()


if __name__ == '__main__':
    sys.exit(main())
