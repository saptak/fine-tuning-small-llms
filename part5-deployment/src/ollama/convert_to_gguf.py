#!/usr/bin/env python3
"""
Model Conversion to GGUF Format for Ollama
From: Fine-Tuning Small LLMs with Docker Desktop - Part 5
"""

import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConverter:
    """Convert fine-tuned models to GGUF format for Ollama"""
    
    def __init__(self, model_path: str, output_dir: str = "./models/gguf"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def merge_lora_adapters(self, save_path: Optional[str] = None) -> str:
        """Merge LoRA adapters with base model"""
        
        if save_path is None:
            save_path = self.output_dir / "merged_model"
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            # Load model with Unsloth (if it's a LoRA model)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,  # Don't quantize for merging
            )
            
            # Merge LoRA adapters
            logger.info("Merging LoRA adapters...")
            merged_model = model.merge_and_unload()
            
            # Save merged model
            logger.info(f"Saving merged model to: {save_path}")
            merged_model.save_pretrained(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            
            logger.info("✅ LoRA adapters merged successfully")
            return str(save_path)
            
        except Exception as e:
            # Fallback: try loading as regular model (already merged)
            logger.warning(f"Failed to load as LoRA model: {e}")
            logger.info("Attempting to load as regular model...")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                model.save_pretrained(str(save_path))
                tokenizer.save_pretrained(str(save_path))
                
                logger.info("✅ Model copied successfully")
                return str(save_path)
                
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise
    
    def convert_to_gguf(self, merged_model_path: str, quantization: str = "q4_0") -> str:
        """Convert merged model to GGUF format"""
        
        # Check if llama.cpp is available
        llama_cpp_path = shutil.which("llama-convert-hf-to-gguf.py")
        if not llama_cpp_path:
            # Try alternative paths
            possible_paths = [
                "/opt/llama.cpp/convert-hf-to-gguf.py",
                "./llama.cpp/convert-hf-to-gguf.py",
                "convert-hf-to-gguf.py"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    llama_cpp_path = path
                    break
            
            if not llama_cpp_path:
                logger.error("llama.cpp conversion script not found!")
                logger.info("Please install llama.cpp:")
                logger.info("git clone https://github.com/ggerganov/llama.cpp.git")
                logger.info("cd llama.cpp && make")
                raise FileNotFoundError("llama.cpp conversion script not found")
        
        # Convert to GGUF
        model_name = Path(merged_model_path).name
        gguf_path = self.output_dir / f"{model_name}.gguf"
        
        logger.info(f"Converting to GGUF format: {gguf_path}")
        
        try:
            # Convert HF model to GGUF
            cmd = [
                "python", llama_cpp_path,
                "--outfile", str(gguf_path),
                "--outtype", "f16",  # Use f16 for better quality
                str(merged_model_path)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Quantize if requested
            if quantization != "f16":
                quantized_path = self.output_dir / f"{model_name}-{quantization}.gguf"
                
                quantize_cmd = [
                    "./llama.cpp/quantize",
                    str(gguf_path),
                    str(quantized_path),
                    quantization
                ]
                
                logger.info(f"Quantizing to {quantization}: {quantized_path}")
                subprocess.run(quantize_cmd, check=True)
                
                # Remove unquantized version
                gguf_path.unlink()
                gguf_path = quantized_path
            
            logger.info(f"✅ GGUF conversion completed: {gguf_path}")
            return str(gguf_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"Required tool not found: {e}")
            raise
    
    def create_ollama_modelfile(self, gguf_path: str, model_name: str = "sql-expert") -> str:
        """Create Ollama Modelfile for the converted model"""
        
        modelfile_path = self.output_dir / "Modelfile"
        
        # Get model info for template
        base_model = "llama"  # Default, could be detected from model config
        
        modelfile_content = f"""FROM {gguf_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER num_predict 256

# System message
SYSTEM \"\"\"You are an expert SQL developer who generates accurate and efficient SQL queries. 
Always provide syntactically correct SQL that follows best practices.\"\"\"

# Template for instruction following
TEMPLATE \"\"\"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"

# Stop sequences
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
"""

        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        logger.info(f"✅ Modelfile created: {modelfile_path}")
        return str(modelfile_path)
    
    def install_in_ollama(self, gguf_path: str, model_name: str = "sql-expert") -> bool:
        """Install the converted model in Ollama"""
        
        try:
            # Create Modelfile
            modelfile_path = self.create_ollama_modelfile(gguf_path, model_name)
            
            # Create model in Ollama
            cmd = ["ollama", "create", model_name, "-f", modelfile_path]
            
            logger.info(f"Installing model in Ollama: {model_name}")
            subprocess.run(cmd, check=True)
            
            # Test the model
            test_cmd = ["ollama", "run", model_name, "SELECT * FROM users;"]
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Model {model_name} installed and tested successfully")
                return True
            else:
                logger.warning(f"Model installed but test failed: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install in Ollama: {e}")
            return False
        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama first:")
            logger.info("https://ollama.ai/download")
            return False
    
    def convert_full_pipeline(self, model_name: str = "sql-expert", 
                            quantization: str = "q4_0") -> str:
        """Run the complete conversion pipeline"""
        
        logger.info("🚀 Starting model conversion pipeline...")
        
        # Step 1: Merge LoRA adapters
        merged_path = self.merge_lora_adapters()
        
        # Step 2: Convert to GGUF
        gguf_path = self.convert_to_gguf(merged_path, quantization)
        
        # Step 3: Install in Ollama
        success = self.install_in_ollama(gguf_path, model_name)
        
        if success:
            logger.info(f"🎉 Model conversion completed successfully!")
            logger.info(f"Model available as: {model_name}")
            logger.info(f"Test with: ollama run {model_name}")
        else:
            logger.warning("Model converted but Ollama installation failed")
        
        return gguf_path

def main():
    """Main conversion function"""
    
    parser = argparse.ArgumentParser(description="Convert fine-tuned model to GGUF for Ollama")
    parser.add_argument("--model-path", required=True, help="Path to the fine-tuned model")
    parser.add_argument("--output-dir", default="./models/gguf", help="Output directory for GGUF files")
    parser.add_argument("--model-name", default="sql-expert", help="Name for the Ollama model")
    parser.add_argument("--quantization", default="q4_0", 
                       choices=["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"],
                       help="Quantization format")
    parser.add_argument("--merge-only", action="store_true", help="Only merge LoRA adapters, don't convert to GGUF")
    parser.add_argument("--skip-ollama", action="store_true", help="Don't install in Ollama")
    
    args = parser.parse_args()
    
    print("🔄 Model Conversion to GGUF for Ollama")
    print("=" * 50)
    
    converter = ModelConverter(args.model_path, args.output_dir)
    
    try:
        if args.merge_only:
            # Only merge LoRA adapters
            merged_path = converter.merge_lora_adapters()
            print(f"✅ LoRA adapters merged: {merged_path}")
        else:
            # Run full pipeline
            gguf_path = converter.convert_full_pipeline(args.model_name, args.quantization)
            print(f"✅ Conversion completed: {gguf_path}")
            
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
