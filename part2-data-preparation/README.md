# Part 2: Data Preparation and Model Selection

This directory contains comprehensive tools for creating, validating, and optimizing training datasets for LLM fine-tuning.

## 🎯 Features

- **Dataset Creation**: Automated generation of high-quality training examples
- **Data Validation**: Comprehensive quality checks and syntax validation
- **Format Conversion**: Convert between Alpaca, Chat, and Completion formats
- **Data Augmentation**: Intelligent dataset expansion techniques
- **Model Selection**: Smart model recommendation based on use case and resources

## 🚀 Quick Start

```bash
# Create a SQL generation dataset
python src/dataset_creation.py --output-dir ../data/datasets --format alpaca

# Validate dataset quality
python src/data_validation.py --dataset ../data/datasets/sql_dataset_alpaca.json

# Convert formats
python src/format_converter.py --input sql_dataset_alpaca.json --output sql_dataset_chat.json --from alpaca --to chat

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

## 📊 Dataset Quality Metrics

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
