#!/usr/bin/env python3
"""
Cost Optimization and Resource Management
From: Fine-Tuning Small LLMs with Docker Desktop - Part 6
"""

import argparse
import json
import time
import docker
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

class CostOptimizer:
    """Cost optimization and resource management for LLM infrastructure"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.cost_data = {
            'compute_hours': 0,
            'storage_gb': 0,
            'network_gb': 0,
            'gpu_hours': 0,
            'total_cost': 0.0
        }
        self.recommendations = []
        
        # Cost rates (example rates - adjust for your provider)
        self.rates = {
            'cpu_hour': 0.048,  # $/hour per vCPU
            'memory_gb_hour': 0.0067,  # $/hour per GB
            'storage_gb_month': 0.10,  # $/month per GB
            'gpu_hour': 1.52,  # $/hour for GPU instance
            'network_gb': 0.09  # $/GB for data transfer
        }
    
    def analyze_container_costs(self) -> Dict[str, Any]:
        """Analyze costs for running containers"""
        
        container_costs = {}
        total_cost = 0.0
        
        containers = self.docker_client.containers.list()
        
        for container in containers:
            try:
                # Get container stats
                stats = container.stats(stream=False)
                
                # Calculate resource usage
                cpu_usage = self._calculate_cpu_usage(stats)
                memory_usage_gb = stats['memory_stats']['usage'] / (1024**3)
                
                # Estimate hourly cost
                hourly_cost = (
                    (cpu_usage / 100) * self.rates['cpu_hour'] +
                    memory_usage_gb * self.rates['memory_gb_hour']
                )
                
                # Check if GPU is being used
                if self._container_uses_gpu(container):
                    hourly_cost += self.rates['gpu_hour']
                
                container_costs[container.name] = {
                    'cpu_usage_percent': cpu_usage,
                    'memory_usage_gb': memory_usage_gb,
                    'hourly_cost': hourly_cost,
                    'monthly_cost': hourly_cost * 24 * 30,
                    'uses_gpu': self._container_uses_gpu(container)
                }
                
                total_cost += hourly_cost
                
            except Exception as e:
                print(f"Error analyzing container {container.name}: {e}")
        
        return {
            'containers': container_costs,
            'total_hourly_cost': total_cost,
            'total_monthly_cost': total_cost * 24 * 30
        }
    
    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats"""
        
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
                return min(cpu_percent, 100.0)
        except (KeyError, ZeroDivisionError):
            pass
        
        return 0.0
    
    def _container_uses_gpu(self, container) -> bool:
        """Check if container is configured to use GPU"""
        
        try:
            inspect_data = container.attrs
            runtime = inspect_data.get('HostConfig', {}).get('Runtime')
            device_requests = inspect_data.get('HostConfig', {}).get('DeviceRequests', [])
            
            return runtime == 'nvidia' or len(device_requests) > 0
        except:
            return False
    
    def analyze_storage_costs(self) -> Dict[str, Any]:
        """Analyze storage costs"""
        
        storage_analysis = {
            'volumes': {},
            'total_storage_gb': 0,
            'monthly_storage_cost': 0
        }
        
        # Analyze Docker volumes
        volumes = self.docker_client.volumes.list()
        
        for volume in volumes:
            try:
                # Get volume size (rough estimate)
                volume_path = f"/var/lib/docker/volumes/{volume.name}/_data"
                size_gb = self._get_directory_size(volume_path) / (1024**3)
                
                monthly_cost = size_gb * self.rates['storage_gb_month']
                
                storage_analysis['volumes'][volume.name] = {
                    'size_gb': size_gb,
                    'monthly_cost': monthly_cost
                }
                
                storage_analysis['total_storage_gb'] += size_gb
                storage_analysis['monthly_storage_cost'] += monthly_cost
                
            except Exception as e:
                print(f"Error analyzing volume {volume.name}: {e}")
        
        return storage_analysis
    
    def _get_directory_size(self, path: str) -> int:
        """Get directory size in bytes"""
        
        try:
            import os
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
            return total_size
        except:
            return 0
    
    def generate_optimization_recommendations(self, container_costs: Dict, storage_costs: Dict) -> List[str]:
        """Generate cost optimization recommendations"""
        
        recommendations = []
        
        # Analyze container efficiency
        for container_name, stats in container_costs['containers'].items():
            cpu_usage = stats['cpu_usage_percent']
            memory_usage = stats['memory_usage_gb']
            monthly_cost = stats['monthly_cost']
            
            # Low CPU utilization
            if cpu_usage < 10 and monthly_cost > 50:
                recommendations.append(
                    f"🔧 Container '{container_name}' has low CPU usage ({cpu_usage:.1f}%). "
                    f"Consider downsizing or consolidating (saves ~${monthly_cost * 0.3:.2f}/month)"
                )
            
            # High cost without GPU usage
            if monthly_cost > 100 and not stats['uses_gpu']:
                recommendations.append(
                    f"💰 Container '{container_name}' costs ${monthly_cost:.2f}/month. "
                    f"Consider optimizing resource allocation or using spot instances"
                )
            
            # GPU usage optimization
            if stats['uses_gpu'] and cpu_usage < 20:
                recommendations.append(
                    f"🖥️ Container '{container_name}' uses GPU but low CPU ({cpu_usage:.1f}%). "
                    f"Consider optimizing workload distribution"
                )
        
        # Storage optimization
        large_volumes = [
            (name, data) for name, data in storage_costs['volumes'].items() 
            if data['size_gb'] > 10
        ]
        
        for volume_name, volume_data in large_volumes:
            recommendations.append(
                f"💾 Volume '{volume_name}' is {volume_data['size_gb']:.1f} GB "
                f"(${volume_data['monthly_cost']:.2f}/month). Consider cleanup or archival"
            )
        
        # Overall cost recommendations
        total_monthly = container_costs['total_monthly_cost'] + storage_costs['monthly_storage_cost']
        
        if total_monthly > 500:
            recommendations.append(
                f"⚠️ Total monthly cost is ${total_monthly:.2f}. "
                f"Consider implementing auto-scaling and scheduled shutdowns"
            )
        
        if len(container_costs['containers']) > 5:
            recommendations.append(
                f"🔄 Running {len(container_costs['containers'])} containers. "
                f"Consider container orchestration for better resource utilization"
            )
        
        return recommendations
    
    def generate_cost_report(self) -> Dict[str, Any]:
        """Generate comprehensive cost analysis report"""
        
        print("💰 Analyzing infrastructure costs...")
        
        # Analyze different cost components
        container_costs = self.analyze_container_costs()
        storage_costs = self.analyze_storage_costs()
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(container_costs, storage_costs)
        
        # Calculate totals
        total_hourly = container_costs['total_hourly_cost']
        total_monthly = container_costs['total_monthly_cost'] + storage_costs['monthly_storage_cost']
        
        # Estimate annual cost
        annual_cost = total_monthly * 12
        
        # System resource analysis
        system_stats = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_usage_percent': psutil.cpu_percent(interval=1),
            'memory_usage_percent': psutil.virtual_memory().percent
        }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'cost_summary': {
                'hourly_cost': total_hourly,
                'monthly_cost': total_monthly,
                'annual_cost': annual_cost
            },
            'container_analysis': container_costs,
            'storage_analysis': storage_costs,
            'system_stats': system_stats,
            'recommendations': recommendations,
            'savings_potential': self._calculate_savings_potential(recommendations)
        }
        
        return report
    
    def _calculate_savings_potential(self, recommendations: List[str]) -> Dict[str, float]:
        """Estimate potential savings from recommendations"""
        
        savings = {
            'monthly_savings': 0.0,
            'annual_savings': 0.0
        }
        
        # Extract savings estimates from recommendations
        for rec in recommendations:
            if 'saves ~$' in rec:
                try:
                    # Extract dollar amount
                    import re
                    matches = re.findall(r'saves ~\$(\d+\.?\d*)', rec)
                    if matches:
                        monthly_saving = float(matches[0])
                        savings['monthly_savings'] += monthly_saving
                except:
                    pass
        
        savings['annual_savings'] = savings['monthly_savings'] * 12
        return savings
    
    def run_optimization_analysis(self, save_report: bool = True) -> Dict[str, Any]:
        """Run complete cost optimization analysis"""
        
        print("🔍 Starting cost optimization analysis...")
        
        # Generate comprehensive report
        report = self.generate_cost_report()
        
        # Display summary
        print("\n💰 Cost Analysis Summary")
        print("=" * 40)
        print(f"Hourly cost: ${report['cost_summary']['hourly_cost']:.2f}")
        print(f"Monthly cost: ${report['cost_summary']['monthly_cost']:.2f}")
        print(f"Annual cost: ${report['cost_summary']['annual_cost']:.2f}")
        
        if report['savings_potential']['monthly_savings'] > 0:
            print(f"\n💡 Potential monthly savings: ${report['savings_potential']['monthly_savings']:.2f}")
            print(f"💡 Potential annual savings: ${report['savings_potential']['annual_savings']:.2f}")
        
        print(f"\n🔧 Optimization Recommendations ({len(report['recommendations'])}):")
        for i, rec in enumerate(report['recommendations'][:5], 1):  # Show top 5
            print(f"{i}. {rec}")
        
        if len(report['recommendations']) > 5:
            print(f"... and {len(report['recommendations']) - 5} more recommendations")
        
        # Save report if requested
        if save_report:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"cost_optimization_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report

def main():
    """Main cost optimization function"""
    
    parser = argparse.ArgumentParser(description="LLM Infrastructure Cost Optimization")
    parser.add_argument("--analyze", action="store_true", help="Run cost analysis")
    parser.add_argument("--recommendations", action="store_true", help="Show optimization recommendations")
    parser.add_argument("--save-report", action="store_true", help="Save detailed report to file")
    parser.add_argument("--rates", help="JSON file with custom cost rates")
    
    args = parser.parse_args()
    
    print("💰 LLM Infrastructure Cost Optimizer")
    print("=" * 50)
    
    optimizer = CostOptimizer()
    
    # Load custom rates if provided
    if args.rates:
        try:
            with open(args.rates, 'r') as f:
                custom_rates = json.load(f)
                optimizer.rates.update(custom_rates)
            print(f"✅ Loaded custom rates from {args.rates}")
        except Exception as e:
            print(f"⚠️ Failed to load custom rates: {e}")
    
    if args.analyze or not any([args.analyze, args.recommendations]):
        # Run full analysis
        report = optimizer.run_optimization_analysis(args.save_report)
        return 0
    
    elif args.recommendations:
        # Show recommendations only
        container_costs = optimizer.analyze_container_costs()
        storage_costs = optimizer.analyze_storage_costs()
        recommendations = optimizer.generate_optimization_recommendations(container_costs, storage_costs)
        
        print(f"🔧 Optimization Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    return 0

if __name__ == "__main__":
    exit(main())
