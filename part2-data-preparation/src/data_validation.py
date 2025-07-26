#!/usr/bin/env python3
"""
Data Validation and Quality Assurance
From: Fine-Tuning Small LLMs with Docker Desktop - Part 2
"""

import json
import pandas as pd
import argparse
import re
import sqlparse
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for training datasets"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.data = None
        self.validation_results = {
            'total_examples': 0,
            'valid_examples': 0,
            'issues': [],
            'statistics': {},
            'quality_score': 0.0
        }
        
    def load_data(self) -> bool:
        """Load dataset from file"""
        
        try:
            if self.dataset_path.suffix == '.json':
                with open(self.dataset_path, 'r') as f:
                    self.data = json.load(f)
            elif self.dataset_path.suffix == '.jsonl':
                self.data = []
                with open(self.dataset_path, 'r') as f:
                    for line in f:
                        self.data.append(json.loads(line.strip()))
            else:
                raise ValueError(f"Unsupported file format: {self.dataset_path.suffix}")
            
            self.validation_results['total_examples'] = len(self.data)
            logger.info(f"✅ Loaded {len(self.data)} examples from {self.dataset_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validate dataset structure"""
        
        logger.info("🔍 Validating dataset structure...")
        
        required_fields = ['instruction', 'input', 'output']
        issues = []
        
        for i, example in enumerate(self.data):
            if not isinstance(example, dict):
                issues.append(f"Example {i}: Not a dictionary")
                continue
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in example]
            if missing_fields:
                issues.append(f"Example {i}: Missing fields: {missing_fields}")
            
            # Check field types
            for field in required_fields:
                if field in example and not isinstance(example[field], str):
                    issues.append(f"Example {i}: Field '{field}' should be string, got {type(example[field])}")
        
        if issues:
            self.validation_results['issues'].extend(issues[:10])  # Limit to first 10 issues
            logger.warning(f"⚠️  Structure issues found: {len(issues)}")
        else:
            logger.info("✅ Structure validation passed")
        
        return len(issues) == 0
    
    def validate_content_quality(self) -> Dict[str, Any]:
        """Validate content quality"""
        
        logger.info("📊 Analyzing content quality...")
        
        stats = {
            'instruction_lengths': [],
            'input_lengths': [],
            'output_lengths': [],
            'empty_inputs': 0,
            'empty_outputs': 0,
            'very_short_instructions': 0,
            'very_long_examples': 0,
            'duplicate_instructions': 0
        }
        
        seen_instructions = set()
        
        for i, example in enumerate(self.data):
            if not isinstance(example, dict):
                continue
            
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output_text = example.get('output', '')
            
            # Length statistics
            stats['instruction_lengths'].append(len(instruction))
            stats['input_lengths'].append(len(input_text))
            stats['output_lengths'].append(len(output_text))
            
            # Quality checks
            if len(instruction.strip()) < 10:
                stats['very_short_instructions'] += 1
            
            if not input_text.strip():
                stats['empty_inputs'] += 1
            
            if not output_text.strip():
                stats['empty_outputs'] += 1
                self.validation_results['issues'].append(f"Example {i}: Empty output")
            
            if len(instruction) + len(input_text) + len(output_text) > 4000:
                stats['very_long_examples'] += 1
            
            # Duplicate detection
            if instruction in seen_instructions:
                stats['duplicate_instructions'] += 1
            else:
                seen_instructions.add(instruction)
        
        # Calculate averages
        if stats['instruction_lengths']:
            stats['avg_instruction_length'] = sum(stats['instruction_lengths']) / len(stats['instruction_lengths'])
            stats['avg_input_length'] = sum(stats['input_lengths']) / len(stats['input_lengths'])
            stats['avg_output_length'] = sum(stats['output_lengths']) / len(stats['output_lengths'])
        
        self.validation_results['statistics'] = stats
        
        # Report findings
        logger.info(f"📏 Average instruction length: {stats.get('avg_instruction_length', 0):.1f} chars")
        logger.info(f"📏 Average input length: {stats.get('avg_input_length', 0):.1f} chars")
        logger.info(f"📏 Average output length: {stats.get('avg_output_length', 0):.1f} chars")
        
        if stats['empty_outputs'] > 0:
            logger.warning(f"⚠️  {stats['empty_outputs']} examples with empty outputs")
        
        if stats['duplicate_instructions'] > 0:
            logger.warning(f"⚠️  {stats['duplicate_instructions']} duplicate instructions")
        
        return stats
    
    def validate_sql_syntax(self) -> Dict[str, Any]:
        """Validate SQL syntax in outputs (if dataset contains SQL)"""
        
        logger.info("🔍 Validating SQL syntax...")
        
        sql_stats = {
            'total_sql_examples': 0,
            'valid_sql': 0,
            'invalid_sql': 0,
            'sql_keywords': set(),
            'syntax_errors': []
        }
        
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']
        
        for i, example in enumerate(self.data):
            if not isinstance(example, dict):
                continue
            
            output = example.get('output', '').upper()
            
            # Check if this looks like SQL
            if any(keyword in output for keyword in sql_keywords):
                sql_stats['total_sql_examples'] += 1
                
                # Extract SQL keywords
                for keyword in sql_keywords:
                    if keyword in output:
                        sql_stats['sql_keywords'].add(keyword)
                
                # Validate SQL syntax
                try:
                    parsed = sqlparse.parse(example.get('output', ''))
                    if parsed and len(parsed) > 0:
                        sql_stats['valid_sql'] += 1
                    else:
                        sql_stats['invalid_sql'] += 1
                        sql_stats['syntax_errors'].append(f"Example {i}: Empty or invalid SQL")
                except Exception as e:
                    sql_stats['invalid_sql'] += 1
                    sql_stats['syntax_errors'].append(f"Example {i}: {str(e)}")
        
        if sql_stats['total_sql_examples'] > 0:
            logger.info(f"📊 SQL examples found: {sql_stats['total_sql_examples']}")
            logger.info(f"✅ Valid SQL: {sql_stats['valid_sql']}")
            if sql_stats['invalid_sql'] > 0:
                logger.warning(f"❌ Invalid SQL: {sql_stats['invalid_sql']}")
                # Log first few errors
                for error in sql_stats['syntax_errors'][:3]:
                    logger.warning(f"   {error}")
        
        return sql_stats
    
    def check_data_diversity(self) -> Dict[str, Any]:
        """Check dataset diversity"""
        
        logger.info("🎨 Analyzing dataset diversity...")
        
        diversity_stats = {
            'unique_instructions': 0,
            'instruction_patterns': {},
            'complexity_distribution': {'easy': 0, 'medium': 0, 'hard': 0},
            'domain_coverage': set()
        }
        
        # Track instruction patterns
        instruction_starts = {}
        
        for example in self.data:
            if not isinstance(example, dict):
                continue
            
            instruction = example.get('instruction', '').lower()
            
            # Track instruction starting patterns
            words = instruction.split()
            if words:
                start_pattern = ' '.join(words[:3])  # First 3 words
                instruction_starts[start_pattern] = instruction_starts.get(start_pattern, 0) + 1
            
            # Estimate complexity based on instruction keywords
            complexity_keywords = {
                'easy': ['select', 'show', 'list', 'find', 'get'],
                'medium': ['count', 'group', 'order', 'join', 'where'],
                'hard': ['subquery', 'having', 'union', 'window', 'recursive']
            }
            
            complexity = 'easy'  # default
            for level, keywords in complexity_keywords.items():
                if any(keyword in instruction for keyword in keywords):
                    complexity = level
            
            diversity_stats['complexity_distribution'][complexity] += 1
            
            # Track domains (basic heuristic)
            domain_keywords = {
                'users': ['user', 'customer', 'person'],
                'orders': ['order', 'purchase', 'transaction'],
                'products': ['product', 'item', 'inventory'],
                'finance': ['payment', 'invoice', 'revenue', 'cost']
            }
            
            for domain, keywords in domain_keywords.items():
                if any(keyword in instruction for keyword in keywords):
                    diversity_stats['domain_coverage'].add(domain)
        
        diversity_stats['unique_instructions'] = len(set(ex.get('instruction', '') for ex in self.data))
        diversity_stats['instruction_patterns'] = dict(sorted(instruction_starts.items(), key=lambda x: x[1], reverse=True)[:10])
        diversity_stats['domain_coverage'] = list(diversity_stats['domain_coverage'])
        
        logger.info(f"🎯 Unique instructions: {diversity_stats['unique_instructions']}")
        logger.info(f"🏗️  Domains covered: {', '.join(diversity_stats['domain_coverage'])}")
        logger.info(f"⚖️  Complexity: Easy={diversity_stats['complexity_distribution']['easy']}, Medium={diversity_stats['complexity_distribution']['medium']}, Hard={diversity_stats['complexity_distribution']['hard']}")
        
        return diversity_stats
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score"""
        
        score = 100.0  # Start with perfect score
        
        # Penalize structural issues
        if self.validation_results['issues']:
            score -= len(self.validation_results['issues']) * 2
        
        stats = self.validation_results.get('statistics', {})
        
        # Penalize empty outputs
        if stats.get('empty_outputs', 0) > 0:
            score -= (stats['empty_outputs'] / self.validation_results['total_examples']) * 30
        
        # Penalize very short instructions
        if stats.get('very_short_instructions', 0) > 0:
            score -= (stats['very_short_instructions'] / self.validation_results['total_examples']) * 10
        
        # Penalize duplicates
        if stats.get('duplicate_instructions', 0) > 0:
            score -= (stats['duplicate_instructions'] / self.validation_results['total_examples']) * 15
        
        # Bonus for good diversity
        diversity = self.validation_results.get('diversity', {})
        if diversity:
            unique_ratio = diversity.get('unique_instructions', 0) / self.validation_results['total_examples']
            if unique_ratio > 0.9:
                score += 5  # Bonus for high uniqueness
        
        return max(0.0, min(100.0, score))
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation pipeline"""
        
        logger.info("🔍 Starting comprehensive data validation...")
        
        if not self.load_data():
            return self.validation_results
        
        # Run all validations
        self.validate_structure()
        content_stats = self.validate_content_quality()
        sql_stats = self.validate_sql_syntax()
        diversity_stats = self.check_data_diversity()
        
        # Store all results
        self.validation_results.update({
            'content_statistics': content_stats,
            'sql_validation': sql_stats,
            'diversity': diversity_stats
        })
        
        # Calculate quality score
        self.validation_results['quality_score'] = self.calculate_quality_score()
        
        # Count valid examples
        self.validation_results['valid_examples'] = (
            self.validation_results['total_examples'] - 
            len([issue for issue in self.validation_results['issues'] if 'Empty output' in issue])
        )
        
        # Summary
        logger.info("\n📋 Validation Summary")
        logger.info("=" * 30)
        logger.info(f"Total examples: {self.validation_results['total_examples']}")
        logger.info(f"Valid examples: {self.validation_results['valid_examples']}")
        logger.info(f"Quality score: {self.validation_results['quality_score']:.1f}/100")
        
        if self.validation_results['quality_score'] >= 90:
            logger.info("✅ Excellent dataset quality!")
        elif self.validation_results['quality_score'] >= 70:
            logger.info("⚠️  Good dataset quality with room for improvement")
        else:
            logger.warning("❌ Dataset quality needs significant improvement")
        
        return self.validation_results
    
    def save_report(self, output_path: str = None) -> str:
        """Save validation report to file"""
        
        if output_path is None:
            output_path = f"{self.dataset_path.stem}_validation_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"📄 Validation report saved to: {output_path}")
        return output_path

def main():
    """Main validation function"""
    
    parser = argparse.ArgumentParser(description="Validate training dataset quality")
    parser.add_argument("--dataset", required=True, help="Path to dataset file (JSON/JSONL)")
    parser.add_argument("--output", help="Output path for validation report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("📊 Dataset Validation and Quality Assessment")
    print("=" * 50)
    
    validator = DataValidator(args.dataset)
    results = validator.run_full_validation()
    report_path = validator.save_report(args.output)
    
    # Return exit code based on quality
    if results['quality_score'] >= 70:
        return 0  # Success
    else:
        return 1  # Quality too low

if __name__ == "__main__":
    exit(main())
