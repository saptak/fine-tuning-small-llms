#!/usr/bin/env python3
"""
System Requirements Checker for LLM Fine-Tuning Environment
From: Fine-Tuning Small LLMs with Docker Desktop - Part 1
"""

import os
import sys
import platform
import subprocess
import psutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SystemChecker:
    def __init__(self):
        self.requirements = {
            "python_version": (3, 10),
            "min_ram_gb": 16,
            "min_disk_gb": 100,
            "min_cpu_cores": 4
        }
        self.results = {}
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Check Python version"""
        current_version = sys.version_info[:2]
        required_version = self.requirements["python_version"]
        
        is_valid = current_version >= required_version
        message = f"Python {current_version[0]}.{current_version[1]} (required: {required_version[0]}.{required_version[1]}+)"
        
        return is_valid, message
    
    def check_system_resources(self) -> Dict[str, Tuple[bool, str]]:
        """Check system resources"""
        results = {}
        
        # Check RAM
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        ram_ok = total_ram_gb >= self.requirements["min_ram_gb"]
        results["ram"] = (ram_ok, f"{total_ram_gb:.1f}GB (required: {self.requirements['min_ram_gb']}GB+)")
        
        # Check CPU cores
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_ok = cpu_cores >= self.requirements["min_cpu_cores"]
        results["cpu"] = (cpu_ok, f"{cpu_cores} cores (required: {self.requirements['min_cpu_cores']}+)")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        free_space_gb = disk_usage.free / (1024**3)
        disk_ok = free_space_gb >= self.requirements["min_disk_gb"]
        results["disk"] = (disk_ok, f"{free_space_gb:.1f}GB free (required: {self.requirements['min_disk_gb']}GB+)")
        
        return results
    
    def check_docker(self) -> Tuple[bool, str]:
        """Check Docker installation"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            return True, version
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "Docker not installed or not accessible"
    
    def check_docker_compose(self) -> Tuple[bool, str]:
        """Check Docker Compose"""
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            return True, version
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "Docker Compose not installed"
    
    def check_nvidia_gpu(self) -> Tuple[bool, str]:
        """Check NVIDIA GPU availability"""
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, check=True)
            # Parse GPU info
            lines = result.stdout.split('\n')
            gpu_info = []
            for line in lines:
                if 'MiB' in line and 'GeForce' in line or 'Tesla' in line or 'RTX' in line:
                    gpu_info.append(line.strip())
            
            if gpu_info:
                return True, f"NVIDIA GPU detected: {len(gpu_info)} GPU(s)"
            else:
                return True, "NVIDIA drivers installed (GPU details not parsed)"
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "NVIDIA GPU not available or nvidia-smi not found"
    
    def check_cuda(self) -> Tuple[bool, str]:
        """Check CUDA installation"""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, check=True)
            version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
            if version_line:
                return True, f"CUDA installed: {version_line[0].strip()}"
            else:
                return True, "CUDA installed (version not parsed)"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "CUDA not installed or nvcc not found"
    
    def check_git(self) -> Tuple[bool, str]:
        """Check Git installation"""
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, check=True)
            return True, result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "Git not installed"
    
    def run_all_checks(self) -> Dict[str, Dict]:
        """Run all system checks"""
        
        print("🔍 Running System Requirements Check")
        print("=" * 50)
        
        # Core requirements
        python_ok, python_msg = self.check_python_version()
        self.results["python"] = {"status": python_ok, "message": python_msg, "required": True}
        
        resource_checks = self.check_system_resources()
        for resource, (status, message) in resource_checks.items():
            self.results[resource] = {"status": status, "message": message, "required": True}
        
        # Docker requirements
        docker_ok, docker_msg = self.check_docker()
        self.results["docker"] = {"status": docker_ok, "message": docker_msg, "required": True}
        
        compose_ok, compose_msg = self.check_docker_compose()
        self.results["docker_compose"] = {"status": compose_ok, "message": compose_msg, "required": True}
        
        # Optional but recommended
        gpu_ok, gpu_msg = self.check_nvidia_gpu()
        self.results["nvidia_gpu"] = {"status": gpu_ok, "message": gpu_msg, "required": False}
        
        cuda_ok, cuda_msg = self.check_cuda()
        self.results["cuda"] = {"status": cuda_ok, "message": cuda_msg, "required": False}
        
        git_ok, git_msg = self.check_git()
        self.results["git"] = {"status": git_ok, "message": git_msg, "required": True}
        
        return self.results
    
    def print_results(self):
        """Print formatted results"""
        
        print("\n📊 System Check Results")
        print("=" * 50)
        
        required_passed = 0
        required_total = 0
        optional_passed = 0
        optional_total = 0
        
        # Required components
        print("\n🔴 Required Components:")
        for component, details in self.results.items():
            if details["required"]:
                required_total += 1
                status_icon = "✅" if details["status"] else "❌"
                if details["status"]:
                    required_passed += 1
                
                print(f"  {status_icon} {component.upper()}: {details['message']}")
        
        # Optional components
        print("\n🟡 Optional (Recommended) Components:")
        for component, details in self.results.items():
            if not details["required"]:
                optional_total += 1
                status_icon = "✅" if details["status"] else "⚠️"
                if details["status"]:
                    optional_passed += 1
                
                print(f"  {status_icon} {component.upper()}: {details['message']}")
        
        # Summary
        print(f"\n📋 Summary")
        print("=" * 20)
        print(f"Required: {required_passed}/{required_total} passed")
        print(f"Optional: {optional_passed}/{optional_total} passed")
        
        if required_passed == required_total:
            print("🎉 All required components are available!")
            if optional_passed < optional_total:
                print("💡 Consider installing optional components for better performance")
        else:
            print("⚠️  Some required components are missing. Please install them before proceeding.")
            
        return required_passed == required_total
    
    def generate_recommendations(self) -> List[str]:
        """Generate installation recommendations"""
        
        recommendations = []
        
        for component, details in self.results.items():
            if not details["status"]:
                if component == "docker":
                    recommendations.append("Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
                elif component == "docker_compose":
                    recommendations.append("Install Docker Compose: https://docs.docker.com/compose/install/")
                elif component == "python":
                    recommendations.append("Install Python 3.10+: https://www.python.org/downloads/")
                elif component == "git":
                    recommendations.append("Install Git: https://git-scm.com/downloads")
                elif component == "nvidia_gpu":
                    recommendations.append("Install NVIDIA drivers: https://www.nvidia.com/drivers/")
                elif component == "cuda":
                    recommendations.append("Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
                elif component == "ram":
                    recommendations.append("Consider upgrading RAM to 16GB+ for better performance")
                elif component == "disk":
                    recommendations.append("Free up disk space or add more storage")
        
        return recommendations

def main():
    """Main function"""
    
    checker = SystemChecker()
    results = checker.run_all_checks()
    
    all_passed = checker.print_results()
    
    # Show recommendations if needed
    recommendations = checker.generate_recommendations()
    if recommendations:
        print(f"\n🔧 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Save results to file
    output_file = "system_check_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": str(psutil.boot_time()),
            "platform": platform.platform(),
            "results": results,
            "recommendations": recommendations,
            "all_required_passed": all_passed
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
