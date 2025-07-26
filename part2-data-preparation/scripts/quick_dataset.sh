#!/bin/bash
# Quick Dataset Generation Script
# From: Fine-Tuning Small LLMs with Docker Desktop - Part 2

set -e

echo "🚀 Quick Dataset Generation for Desktop Fine-Tuning"
echo ""

# Default values
SOURCE="huggingface"
NUM_EXAMPLES=1000
FORMAT="alpaca"
DATASET_NAME="b-mc2/sql-create-context"
OUTPUT_DIR="../data/datasets"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --size)
            case $2 in
                small)
                    NUM_EXAMPLES=500
                    echo "📊 Selected: Small dataset (500 examples, ~15min training)"
                    ;;
                medium)
                    NUM_EXAMPLES=1000
                    echo "📊 Selected: Medium dataset (1K examples, ~30min training)"
                    ;;
                large)
                    NUM_EXAMPLES=5000
                    echo "📊 Selected: Large dataset (5K examples, ~2hr training)"
                    ;;
                *)
                    echo "❌ Invalid size. Use: small, medium, or large"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --manual)
            SOURCE="manual"
            echo "🔧 Using manual dataset creation"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --size small|medium|large    Dataset size (default: medium)"
            echo "  --manual                     Use manual dataset creation"
            echo "  --help                       Show this help"
            echo ""
            echo "Dataset sizes:"
            echo "  small:  500 examples  (~15min training, 2GB memory)"
            echo "  medium: 1K examples   (~30min training, 4GB memory)"  
            echo "  large:  5K examples   (~2hr training, 6GB memory)"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "⚙️  Configuration:"
echo "  Source: $SOURCE"
echo "  Examples: $NUM_EXAMPLES"
echo "  Format: $FORMAT"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create the dataset
if [ "$SOURCE" = "huggingface" ]; then
    echo "🔄 Creating dataset from HuggingFace..."
    python src/dataset_creation.py \
        --source huggingface \
        --hf-dataset "$DATASET_NAME" \
        --num-examples $NUM_EXAMPLES \
        --format $FORMAT \
        --output-dir "$OUTPUT_DIR"
else
    echo "🔧 Creating manual dataset..."
    python src/dataset_creation.py \
        --source manual \
        --format $FORMAT \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "✅ Dataset creation completed!"
echo ""
echo "Next steps:"
echo "1. Validate dataset: python src/data_validation.py --dataset $OUTPUT_DIR/sql_dataset_*_alpaca.json"
echo "2. Start training: cd ../part3-training && python src/fine_tune_model.py"
echo ""
echo "💡 Tip: For faster training on desktop, start with --size small"