#!/usr/bin/env python3
"""
System Requirements Check for LLM Fine-Tuning
From: Fine-Tuning Small LLMs with Docker Desktop - Part 1
"""

import sys
import subprocess
import shutil
import platform
import psutil
import os
from typing import Dict, List, Tuple
import json

class SystemChecker:
    """Comprehensive system requirements checker"""
    
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
        self.system_info = {}
        
    def check_python_version(self) -> bool:
        """Check Python version requirements"""
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        if version.major >= required_major and version.minor >= required_minor:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
            self.system_info['python_version'] = f"{version.major}.{version.minor}.{version.micro}"
            return True
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Required: {required_major}.{required_minor}+)")
            self.errors.append(f"Python version {required_major}.{required_minor}+ required")
            return False
    
    def check_docker(self) -> bool:
        """Check Docker installation and status"""
        print("🐳 Checking Docker...")
        
        # Check if Docker is installed
        if not shutil.which('docker'):
            print("❌ Docker not found")
            self.errors.append("Docker not installed")
            return False
        
        try:
            # Check Docker version
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            print(f"✅ {version}")
            
            # Check if Docker daemon is running
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, check=True)
            print("✅ Docker daemon is running")
            
            # Check Docker Compose
            if shutil.which('docker-compose'):
                result = subprocess.run(['docker-compose', '--version'], 
                                      capture_output=True, text=True, check=True)
                compose_version = result.stdout.strip()
                print(f"✅ {compose_version}")
                self.system_info['docker_compose'] = True
            else:
                print("⚠️  docker-compose not found, using 'docker compose' instead")
                self.warnings.append("docker-compose not installed")
                self.system_info['docker_compose'] = False
            
            self.system_info['docker_installed'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            if 'info' in e.cmd:
                print("❌ Docker daemon not running")
                self.errors.append("Docker daemon not running")
            else:
                print(f"❌ Docker error: {e}")
                self.errors.append(f"Docker error: {e}")
            return False
    
    def check_gpu_support(self) -> bool:
        """Check GPU and CUDA support"""
        print("🖥️  Checking GPU support...")
        
        # Check NVIDIA drivers
        nvidia_smi = shutil.which('nvidia-smi')
        if not nvidia_smi:
            print("⚠️  nvidia-smi not found (GPU training will use CPU)")
            self.warnings.append("NVIDIA drivers not detected")
            self.system_info['gpu_available'] = False
            return False
        
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, check=True)
            print("✅ NVIDIA drivers detected")
            
            # Check CUDA version
            try:
                result = subprocess.run(['nvcc', '--version'], 
                                      capture_output=True, text=True, check=True)
                cuda_version = result.stdout
                print("✅ CUDA toolkit detected")
                self.system_info['cuda_available'] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️  CUDA toolkit not found")
                self.warnings.append("CUDA toolkit not installed")
                self.system_info['cuda_available'] = False
            
            # Check Docker GPU support
            try:
                result = subprocess.run([
                    'docker', 'run', '--rm', '--gpus', 'all', 
                    'nvidia/cuda:11.8-base-ubuntu20.04', 'nvidia-smi'
                ], capture_output=True, text=True, check=True, timeout=30)
                print("✅ Docker GPU support working")
                self.system_info['docker_gpu'] = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print("❌ Docker GPU support not working")
                self.errors.append("Docker GPU support not configured")
                self.system_info['docker_gpu'] = False
            
            self.system_info['gpu_available'] = True
            return True
            
        except subprocess.CalledProcessError:
            print("❌ NVIDIA drivers not working")
            self.errors.append("NVIDIA drivers not working")
            self.system_info['gpu_available'] = False
            return False
    
    def check_memory_requirements(self) -> bool:
        """Check system memory requirements"""
        print("💾 Checking memory requirements...")
        
        # Get total system memory
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"📊 Total RAM: {total_gb:.1f} GB")
        print(f"📊 Available RAM: {available_gb:.1f} GB")
        
        # Check requirements
        min_required = 8  # GB
        recommended = 16  # GB
        
        if total_gb >= recommended:
            print(f"✅ Memory: {total_gb:.1f} GB (Excellent)")
        elif total_gb >= min_required:
            print(f"⚠️  Memory: {total_gb:.1f} GB (Minimum met, more recommended)")
            self.warnings.append(f"Only {total_gb:.1f} GB RAM (16+ GB recommended)")
        else:
            print(f"❌ Memory: {total_gb:.1f} GB (Insufficient, minimum 8 GB required)")
            self.errors.append(f"Insufficient RAM: {total_gb:.1f} GB (minimum 8 GB)")
            return False
        
        self.system_info['total_memory_gb'] = total_gb
        self.system_info['available_memory_gb'] = available_gb
        return True
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        print("💽 Checking disk space...")
        
        # Check current directory disk space
        disk = psutil.disk_usage('.')
        total_gb = disk.total / (1024**3)
        free_gb = disk.free / (1024**3)
        
        print(f"📊 Total disk: {total_gb:.1f} GB")
        print(f"📊 Free disk: {free_gb:.1f} GB")
        
        # Check requirements
        min_required = 20  # GB
        recommended = 50   # GB
        
        if free_gb >= recommended:
            print(f"✅ Disk space: {free_gb:.1f} GB (Excellent)")
        elif free_gb >= min_required:
            print(f"⚠️  Disk space: {free_gb:.1f} GB (Minimum met)")
            self.warnings.append(f"Only {free_gb:.1f} GB free (50+ GB recommended)")
        else:
            print(f"❌ Disk space: {free_gb:.1f} GB (Insufficient)")
            self.errors.append(f"Insufficient disk space: {free_gb:.1f} GB (minimum 20 GB)")
            return False
        
        self.system_info['total_disk_gb'] = total_gb
        self.system_info['free_disk_gb'] = free_gb
        return True
    
    def check_python_packages(self) -> bool:
        """Check required Python packages"""
        print("📦 Checking Python packages...")
        
        required_packages = [
            'torch', 'transformers', 'datasets', 'accelerate',
            'fastapi', 'uvicorn', 'streamlit', 'pandas', 'numpy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (not installed)")
                missing_packages.append(package)
        
        if missing_packages:
            self.warnings.append(f"Missing packages: {', '.join(missing_packages)}")
            print(f"⚠️  Install missing packages: pip install {' '.join(missing_packages)}")
        
        self.system_info['missing_packages'] = missing_packages
        return len(missing_packages) == 0
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity for downloads"""
        print("🌐 Checking network connectivity...")
        
        test_urls = [
            'https://huggingface.co',
            'https://github.com',
            'https://pypi.org'
        ]
        
        import urllib.request
        
        failed_connections = []
        for url in test_urls:
            try:
                urllib.request.urlopen(url, timeout=5)
                print(f"✅ {url}")
            except:
                print(f"❌ {url}")
                failed_connections.append(url)
        
        if failed_connections:
            self.warnings.append(f"Network issues with: {', '.join(failed_connections)}")
        
        return len(failed_connections) == 0
    
    def get_system_info(self) -> Dict:
        """Collect comprehensive system information"""
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'cpu_count': psutil.cpu_count(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
        }
        
        # Add collected information
        info.update(self.system_info)
        
        return info
    
    def run_all_checks(self) -> bool:
        """Run all system checks"""
        
        print("🔍 LLM Fine-Tuning System Requirements Check")
        print("=" * 50)
        
        checks = [
            self.check_python_version(),
            self.check_docker(),
            self.check_memory_requirements(),
            self.check_disk_space(),
            self.check_gpu_support(),  # Non-critical
            self.check_python_packages(),  # Non-critical
            self.check_network_connectivity()  # Non-critical
        ]
        
        # Summary
        print("\n📋 Summary")
        print("=" * 20)
        
        if self.errors:
            print("❌ Critical Issues:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print("⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        critical_checks_passed = len(self.errors) == 0
        
        if critical_checks_passed:
            print("✅ System ready for LLM fine-tuning!")
        else:
            print("❌ Please resolve critical issues before proceeding")
        
        # Save system info
        system_info = self.get_system_info()
        with open('system_check_results.json', 'w') as f:
            json.dump({
                'system_info': system_info,
                'errors': self.errors,
                'warnings': self.warnings,
                'ready': critical_checks_passed
            }, f, indent=2)
        
        print(f"📄 Detailed results saved to: system_check_results.json")
        
        return critical_checks_passed

def main():
    """Main function"""
    
    checker = SystemChecker()
    success = checker.run_all_checks()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
