# Part 2: Data Preparation and Model Selection

This directory contains comprehensive tools for creating, validating, and optimizing training datasets for LLM fine-tuning.

## 🎯 Features

- **Dataset Creation**: Automated generation of high-quality training examples
- **Data Validation**: Comprehensive quality checks and syntax validation
- **Format Conversion**: Convert between Alpaca, Chat, and Completion formats
- **Data Augmentation**: Intelligent dataset expansion techniques
- **Model Selection**: Smart model recommendation based on use case and resources

## 🚀 Quick Start

### Option A: Use Public Dataset (Recommended for Desktop)
```bash
# Load 1K examples for fast training (30min-1hr)
python src/dataset_creation.py --source huggingface --num-examples 1000 --format alpaca

# Load 5K examples for comprehensive training (1-2hr)  
python src/dataset_creation.py --source huggingface --num-examples 5000 --format alpaca

# Use different HuggingFace dataset
python src/dataset_creation.py --source huggingface --hf-dataset "spider" --num-examples 500
```

### Option B: Create Manual Dataset (Original Tutorial)
```bash
# Create a SQL generation dataset manually
python src/dataset_creation.py --source manual --format alpaca
```

### Validation and Conversion
```bash
# Validate dataset quality
python src/data_validation.py --dataset ../data/datasets/sql_dataset_hf_alpaca.json

# Convert formats
python src/format_converter.py --input sql_dataset_hf_alpaca.json --output sql_dataset_chat.json --from alpaca --to chat

# Get model recommendations
python src/model_selection.py --use-case coding --memory-gb 16
```

## 📁 Directory Contents

- `src/` - Core data preparation utilities
- `examples/` - Example datasets and templates
- `scripts/` - Automated data preparation workflows
- `docs/` - Data preparation guides and best practices

## 🔧 Key Components

### Dataset Creation (`src/dataset_creation.py`)
- SQLDatasetCreator class for SQL query generation
- Automated example generation for different difficulty levels
- Support for multiple output formats

### Data Validation (`src/data_validation.py`)
- Syntax validation for SQL queries
- Quality metrics and statistics
- Comprehensive error reporting

### Format Conversion (`src/format_converter.py`)
- Convert between training formats
- Preserve data integrity during conversion
- Batch processing support

### Model Selection (`src/model_selection.py`)
- Resource-aware model recommendations
- Performance vs. memory trade-offs
- Use case specific suggestions

## 📊 Dataset Options and Recommendations

### Desktop Training Recommendations:
- **1K examples**: ~30min training, 2-4GB memory, ideal for testing
- **5K examples**: ~1-2hr training, 4-6GB memory, good balance
- **10K+ examples**: 3hr+ training, 8GB+ memory, comprehensive but slow

### Available Public Datasets:
- **b-mc2/sql-create-context**: 78K examples, professionally curated
- **spider**: 10K examples, complex cross-domain queries  
- **wikisql**: 80K examples, simpler single-table queries

### Dataset Quality Metrics

The validation framework checks:

- **Syntax Correctness**: SQL query validation
- **Content Quality**: Instruction clarity and completeness
- **Format Consistency**: Proper field structure
- **Diversity**: Coverage of different query types
- **Balance**: Distribution across difficulty levels

## 🎨 Data Augmentation

Techniques included:

- **Table Name Variations**: Synonym replacement
- **Condition Modifications**: Different time periods and filters
- **Query Complexity**: Graduated difficulty levels
- **Schema Variations**: Different table structures

## 📖 Related Blog Post

[Part 2: Data Preparation and Model Selection](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part2-data-preparation/)

## 🔗 Usage Examples

See the `examples/` directory for:
- Sample dataset configurations
- Quality validation reports
- Format conversion examples
- Model selection scenarios
