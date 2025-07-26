#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Framework
From: Fine-Tuning Small LLMs with Docker Desktop - Part 4
"""

import json
import time
import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from evaluate import load
from tqdm import tqdm
import logging

@dataclass
class EvaluationResult:
    """Results from model evaluation"""
    model_name: str
    dataset_name: str
    timestamp: str
    metrics: Dict[str, float]
    examples: List[Dict[str, Any]]
    performance_stats: Dict[str, float]

class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.metrics = {}
        
        # Initialize metrics
        self._load_metrics()
        
    def _get_device(self, device: str) -> str:
        """Determine the best device for evaluation"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_metrics(self):
        """Load evaluation metrics"""
        try:
            self.metrics = {
                'bleu': load('bleu'),
                'rouge': load('rouge'),
                'meteor': load('meteor'),
                'bertscore': load('bertscore')
            }
            print("✅ Evaluation metrics loaded")
        except Exception as e:
            print(f"⚠️  Warning: Could not load some metrics: {e}")
    
    def load_model(self):
        """Load model and tokenizer"""
        print(f"📂 Loading model from: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 256, 
                         temperature: float = 0.7) -> Tuple[str, Dict[str, float]]:
        """Generate response from model with performance tracking"""
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Calculate performance metrics
        end_time = time.time()
        response_time = end_time - start_time
        tokens_generated = len(outputs[0]) - inputs['input_ids'].shape[1]
        tokens_per_second = tokens_generated / response_time if response_time > 0 else 0
        
        performance_stats = {
            "response_time_ms": response_time * 1000,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second
        }
        
        return response, performance_stats
    
    def evaluate_dataset(self, test_data: List[Dict[str, str]], 
                        max_examples: Optional[int] = None) -> EvaluationResult:
        """Evaluate model on a test dataset"""
        
        if max_examples:
            test_data = test_data[:max_examples]
        
        print(f"📊 Evaluating on {len(test_data)} examples")
        
        predictions = []
        references = []
        examples = []
        performance_stats = []
        
        # Generate predictions
        for i, example in enumerate(tqdm(test_data, desc="Generating responses")):
            try:
                # Create prompt
                if 'input' in example and example['input'].strip():
                    prompt = f"Instruction: {example['instruction']}\n\nInput: {example['input']}\n\nResponse:"
                else:
                    prompt = f"Instruction: {example['instruction']}\n\nResponse:"
                
                # Generate response
                prediction, perf_stats = self.generate_response(prompt)
                
                # Store results
                predictions.append(prediction)
                references.append(example['output'])
                performance_stats.append(perf_stats)
                
                # Store example for analysis
                examples.append({
                    "instruction": example['instruction'],
                    "input": example.get('input', ''),
                    "expected_output": example['output'],
                    "predicted_output": prediction,
                    "performance": perf_stats
                })
                
            except Exception as e:
                logging.error(f"Error processing example {i}: {e}")
                predictions.append("")
                references.append(example['output'])
                performance_stats.append({})
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, references)
        
        # Aggregate performance stats
        avg_performance = self._aggregate_performance_stats(performance_stats)
        
        # Create evaluation result
        result = EvaluationResult(
            model_name=Path(self.model_path).name,
            dataset_name="test_dataset",
            timestamp=pd.Timestamp.now().isoformat(),
            metrics=metrics,
            examples=examples,
            performance_stats=avg_performance
        )
        
        return result
    
    def calculate_metrics(self, predictions: List[str], 
                         references: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        metrics_results = {}
        
        # Exact match accuracy
        exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        metrics_results['exact_match'] = exact_matches / len(predictions)
        
        # BLEU score
        if 'bleu' in self.metrics:
            try:
                bleu_result = self.metrics['bleu'].compute(
                    predictions=predictions, 
                    references=[[r] for r in references]
                )
                metrics_results['bleu'] = bleu_result['bleu']
            except Exception as e:
                logging.warning(f"BLEU calculation failed: {e}")
        
        # ROUGE scores
        if 'rouge' in self.metrics:
            try:
                rouge_result = self.metrics['rouge'].compute(
                    predictions=predictions, 
                    references=references
                )
                metrics_results.update({
                    'rouge1': rouge_result['rouge1'],
                    'rouge2': rouge_result['rouge2'],
                    'rougeL': rouge_result['rougeL']
                })
            except Exception as e:
                logging.warning(f"ROUGE calculation failed: {e}")
        
        # METEOR score
        if 'meteor' in self.metrics:
            try:
                meteor_result = self.metrics['meteor'].compute(
                    predictions=predictions, 
                    references=references
                )
                metrics_results['meteor'] = meteor_result['meteor']
            except Exception as e:
                logging.warning(f"METEOR calculation failed: {e}")
        
        # BERTScore
        if 'bertscore' in self.metrics and len(predictions) <= 100:  # Limit for performance
            try:
                bertscore_result = self.metrics['bertscore'].compute(
                    predictions=predictions, 
                    references=references, 
                    lang="en"
                )
                metrics_results['bertscore_f1'] = np.mean(bertscore_result['f1'])
            except Exception as e:
                logging.warning(f"BERTScore calculation failed: {e}")
        
        # SQL-specific metrics (if applicable)
        if self._is_sql_dataset(predictions, references):
            sql_metrics = self._calculate_sql_metrics(predictions, references)
            metrics_results.update(sql_metrics)
        
        return metrics_results
    
    def _is_sql_dataset(self, predictions: List[str], references: List[str]) -> bool:
        """Check if dataset contains SQL queries"""
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INSERT', 'UPDATE', 'DELETE']
        
        sample_size = min(10, len(references))
        sql_count = 0
        
        for ref in references[:sample_size]:
            if any(keyword in ref.upper() for keyword in sql_keywords):
                sql_count += 1
        
        return sql_count / sample_size > 0.5
    
    def _calculate_sql_metrics(self, predictions: List[str], 
                              references: List[str]) -> Dict[str, float]:
        """Calculate SQL-specific metrics"""
        
        metrics = {}
        
        # SQL syntax validity (basic check)
        valid_sql_count = 0
        for pred in predictions:
            if self._is_valid_sql_syntax(pred):
                valid_sql_count += 1
        
        metrics['sql_syntax_validity'] = valid_sql_count / len(predictions)
        
        # Keyword accuracy
        keyword_matches = 0
        total_keywords = 0
        
        for pred, ref in zip(predictions, references):
            pred_keywords = self._extract_sql_keywords(pred)
            ref_keywords = self._extract_sql_keywords(ref)
            
            total_keywords += len(ref_keywords)
            keyword_matches += len(set(pred_keywords) & set(ref_keywords))
        
        metrics['sql_keyword_accuracy'] = keyword_matches / max(total_keywords, 1)
        
        return metrics
    
    def _is_valid_sql_syntax(self, sql: str) -> bool:
        """Basic SQL syntax validation"""
        try:
            import sqlparse
            parsed = sqlparse.parse(sql)
            return len(parsed) > 0 and parsed[0].tokens
        except:
            # Fallback to basic keyword check
            sql_keywords = ['SELECT', 'FROM', 'INSERT', 'UPDATE', 'DELETE']
            return any(keyword in sql.upper() for keyword in sql_keywords)
    
    def _extract_sql_keywords(self, sql: str) -> List[str]:
        """Extract SQL keywords from query"""
        keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 
                   'ORDER', 'GROUP', 'HAVING', 'INSERT', 'UPDATE', 'DELETE']
        
        found_keywords = []
        sql_upper = sql.upper()
        
        for keyword in keywords:
            if keyword in sql_upper:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _aggregate_performance_stats(self, stats_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate performance statistics"""
        
        if not stats_list or not stats_list[0]:
            return {}
        
        # Collect all valid stats
        valid_stats = [stats for stats in stats_list if stats]
        
        if not valid_stats:
            return {}
        
        # Calculate averages
        aggregated = {}
        for key in valid_stats[0].keys():
            values = [stats[key] for stats in valid_stats if key in stats]
            if values:
                aggregated[f"avg_{key}"] = np.mean(values)
                aggregated[f"std_{key}"] = np.std(values)
                aggregated[f"min_{key}"] = np.min(values)
                aggregated[f"max_{key}"] = np.max(values)
        
        return aggregated
    
    def save_results(self, result: EvaluationResult, output_dir: str):
        """Save evaluation results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_file = output_path / f"evaluation_results_{result.timestamp.replace(':', '-')}.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save CSV summary
        summary_data = {
            'model_name': [result.model_name],
            'timestamp': [result.timestamp],
            **{f"metric_{k}": [v] for k, v in result.metrics.items()},
            **{f"perf_{k}": [v] for k, v in result.performance_stats.items()}
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / f"evaluation_summary_{result.timestamp.replace(':', '-')}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"💾 Results saved to: {results_file}")
        print(f"📊 Summary saved to: {summary_file}")

def load_test_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """Load test dataset from JSON file"""
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Ensure consistent format
    formatted_data = []
    for item in data:
        formatted_item = {
            'instruction': item.get('instruction', ''),
            'input': item.get('input', ''),
            'output': item.get('output', '')
        }
        formatted_data.append(formatted_item)
    
    return formatted_data

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model-path", required=True, help="Path to the model directory")
    parser.add_argument("--test-data", default="../../data/datasets/sql_dataset_alpaca.json", 
                       help="Path to test dataset")
    parser.add_argument("--output-dir", default="./results", help="Output directory for results")
    parser.add_argument("--max-examples", type=int, help="Maximum number of examples to evaluate")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    print("📊 Model Evaluation Framework")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.device)
    evaluator.load_model()
    
    # Load test dataset
    print(f"📚 Loading test dataset: {args.test_data}")
    test_data = load_test_dataset(args.test_data)
    
    # Run evaluation
    result = evaluator.evaluate_dataset(test_data, args.max_examples)
    
    # Display results
    print("\n📊 Evaluation Results")
    print("=" * 30)
    for metric, value in result.metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n⚡ Performance Stats")
    print("=" * 30)
    for stat, value in result.performance_stats.items():
        print(f"{stat}: {value:.2f}")
    
    # Save results
    evaluator.save_results(result, args.output_dir)
    
    print("\n✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()
