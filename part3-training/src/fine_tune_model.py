#!/usr/bin/env python3
"""
Fine-Tuning with Unsloth - Training Script
From: Fine-Tuning Small LLMs with Docker Desktop - Part 3
"""

import torch
import wandb
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import os

class UnslothTrainer:
    """Unsloth-based training pipeline for LLM fine-tuning"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model(self):
        """Initialize model and tokenizer with Unsloth optimizations"""
        
        model_config = self.config['model']
        
        # Load model with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['name'],
            max_seq_length=model_config['max_seq_length'],
            dtype=None,  # Auto detection
            load_in_4bit=model_config.get('load_in_4bit', True),
        )
        
        # Configure LoRA adapters
        lora_config = self.config['lora']
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config['r'],
            target_modules=lora_config['target_modules'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        print(f"✅ Model loaded: {model_config['name']}")
        print(f"📊 Model parameters: {self.model.print_trainable_parameters()}")
    
    def load_dataset(self) -> Dataset:
        """Load and prepare training dataset"""
        
        dataset_config = self.config['dataset']
        
        if dataset_config['type'] == 'json':
            # Load from JSON file
            dataset = load_dataset('json', data_files=dataset_config['path'])['train']
        elif dataset_config['type'] == 'huggingface':
            # Load from Hugging Face Hub
            dataset = load_dataset(dataset_config['name'], split='train')
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")
        
        # Format dataset for training
        def format_example(example):
            """Format example for instruction fine-tuning"""
            
            model_name = model_config['name'].lower()
            
            if dataset_config['format'] == 'alpaca':
                if 'llama-3' in model_name:
                    # Llama 3.1 format
                    if example.get('input', '').strip():
                        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{example['instruction']}\n\n{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{example['output']}<|eot_id|><|end_of_text|>"
                    else:
                        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{example['output']}<|eot_id|><|end_of_text|>"
                elif 'mistral' in model_name or 'mixtral' in model_name:
                    # Mistral format
                    if example.get('input', '').strip():
                        text = f"<s>[INST] {example['instruction']}\n\n{example['input']} [/INST] {example['output']}</s>"
                    else:
                        text = f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
                elif 'phi-3' in model_name:
                    # Phi-3 format
                    if example.get('input', '').strip():
                        text = f"<|system|>You are an expert assistant.<|end|><|user|>{example['instruction']}\n\n{example['input']}<|end|><|assistant|>{example['output']}<|end|>"
                    else:
                        text = f"<|system|>You are an expert assistant.<|end|><|user|>{example['instruction']}<|end|><|assistant|>{example['output']}<|end|>"
                else:
                    # Generic Alpaca format (works with most models)
                    if example.get('input', '').strip():
                        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
                    else:
                        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            
            return {"text": text}
        
        # Apply formatting
        dataset = dataset.map(format_example)
        
        # Split into train/validation if needed
        if dataset_config.get('validation_split', 0) > 0:
            split_dataset = dataset.train_test_split(
                test_size=dataset_config['validation_split'],
                seed=42
            )
            train_dataset = split_dataset['train']
            val_dataset = split_dataset['test']
        else:
            train_dataset = dataset
            val_dataset = None
        
        print(f"📚 Training examples: {len(train_dataset)}")
        if val_dataset:
            print(f"📝 Validation examples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def setup_training_args(self) -> TrainingArguments:
        """Configure training arguments"""
        
        training_config = self.config['training']
        
        args = TrainingArguments(
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config.get('eval_batch_size', training_config['batch_size']),
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            warmup_steps=training_config['warmup_steps'],
            num_train_epochs=training_config['epochs'],
            learning_rate=training_config['learning_rate'],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=training_config.get('logging_steps', 50),
            optim="adamw_8bit",
            weight_decay=training_config.get('weight_decay', 0.01),
            lr_scheduler_type=training_config.get('lr_scheduler', "cosine"),
            seed=42,
            output_dir=training_config['output_dir'],
            save_strategy="steps",
            save_steps=training_config.get('save_steps', 500),
            eval_strategy="steps" if training_config.get('eval_steps') else "no",
            eval_steps=training_config.get('eval_steps'),
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', "eval_loss"),
            report_to="wandb" if training_config.get('use_wandb', False) else None,
            run_name=training_config.get('run_name', "unsloth-fine-tuning"),
        )
        
        return args
    
    def setup_trainer(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Initialize the SFT trainer"""
        
        training_args = self.setup_training_args()
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=self.config['model']['max_seq_length'],
            dataset_num_proc=4,
            args=training_args,
        )
        
        print("✅ Trainer initialized")
    
    def train(self):
        """Run the training process"""
        
        # Initialize Wandb if configured
        if self.config['training'].get('use_wandb', False):
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb'].get('entity'),
                name=self.config['training'].get('run_name', "unsloth-fine-tuning"),
                config=self.config
            )
        
        print("🚀 Starting training...")
        
        # Train the model
        trainer_stats = self.trainer.train()
        
        print("✅ Training completed!")
        print(f"📊 Training stats: {trainer_stats}")
        
        # Save the final model
        self.save_model()
        
        # Finish Wandb run
        if self.config['training'].get('use_wandb', False):
            wandb.finish()
    
    def save_model(self):
        """Save the trained model and adapters"""
        
        output_dir = Path(self.config['training']['output_dir'])
        
        # Save LoRA adapters
        lora_dir = output_dir / "lora_adapters"
        self.model.save_pretrained(str(lora_dir))
        self.tokenizer.save_pretrained(str(lora_dir))
        
        # Save merged model (optional)
        if self.config['training'].get('save_merged_model', True):
            merged_dir = output_dir / "merged_model"
            
            # Merge and save
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(str(merged_dir))
            self.tokenizer.save_pretrained(str(merged_dir))
            
            print(f"💾 Merged model saved to: {merged_dir}")
        
        print(f"💾 LoRA adapters saved to: {lora_dir}")
    
    def evaluate(self, test_dataset: Optional[Dataset] = None):
        """Evaluate the trained model"""
        
        if test_dataset is None and self.trainer.eval_dataset is None:
            print("⚠️  No evaluation dataset available")
            return
        
        print("📊 Running evaluation...")
        
        eval_results = self.trainer.evaluate(eval_dataset=test_dataset)
        
        print("✅ Evaluation completed!")
        print(f"📊 Evaluation results: {eval_results}")
        
        return eval_results

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Fine-tune LLM with Unsloth")
    parser.add_argument("--config", required=True, help="Path to training configuration file")
    parser.add_argument("--resume", help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Set environment variables for optimization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("🚀 Unsloth Fine-Tuning Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = UnslothTrainer(args.config)
    
    # Setup model
    trainer.setup_model()
    
    # Load dataset
    train_dataset, val_dataset = trainer.load_dataset()
    
    # Setup trainer
    trainer.setup_trainer(train_dataset, val_dataset)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        trainer.trainer.train(resume_from_checkpoint=args.resume)
    else:
        # Start training
        trainer.train()
    
    # Run evaluation
    trainer.evaluate()
    
    print("🎉 Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
