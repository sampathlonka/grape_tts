#!/bin/bash

# SupertonicTTS Quick Start Script
# This script runs all 3 training stages on DGX Spark GB10

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-.}/outputs"
STAGE1_CONFIG="${STAGE1_CONFIG:-.}/config_autoencoder.yaml"
STAGE2_CONFIG="${STAGE2_CONFIG:-.}/config_text_to_latent.yaml"
STAGE3_CONFIG="${STAGE3_CONFIG:-.}/config_duration.yaml"
RESUME_STAGE="${RESUME_STAGE:-}"

echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}SupertonicTTS Training Pipeline${NC}"
echo -e "${GREEN}==============================================================${NC}"
echo ""
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Stage 1 config: $STAGE1_CONFIG"
echo "  Stage 2 config: $STAGE2_CONFIG"
echo "  Stage 3 config: $STAGE3_CONFIG"
if [ ! -z "$RESUME_STAGE" ]; then
    echo "  Resume from stage: $RESUME_STAGE"
fi
echo ""

# Check if config files exist
if [ ! -f "$STAGE1_CONFIG" ]; then
    echo -e "${RED}Error: Stage 1 config not found: $STAGE1_CONFIG${NC}"
    exit 1
fi

if [ ! -f "$STAGE2_CONFIG" ]; then
    echo -e "${RED}Error: Stage 2 config not found: $STAGE2_CONFIG${NC}"
    exit 1
fi

if [ ! -f "$STAGE3_CONFIG" ]; then
    echo -e "${RED}Error: Stage 3 config not found: $STAGE3_CONFIG${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check CUDA availability
echo -e "${YELLOW}Checking CUDA availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# Build command
CMD="python run_all_stages.py \
    --output_dir $OUTPUT_DIR \
    --stage1_config $STAGE1_CONFIG \
    --stage2_config $STAGE2_CONFIG \
    --stage3_config $STAGE3_CONFIG"

if [ ! -z "$RESUME_STAGE" ]; then
    CMD="$CMD --resume_stage $RESUME_STAGE"
fi

echo -e "${GREEN}Starting training pipeline...${NC}"
echo ""

# Run training
$CMD

echo ""
echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}Training pipeline completed!${NC}"
echo -e "${GREEN}==============================================================${NC}"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. View tensorboard logs:"
echo "     tensorboard --logdir $OUTPUT_DIR"
echo "  2. Check training logs:"
echo "     tail -f $OUTPUT_DIR/orchestration_logs/training_orchestration.log"
echo "  3. Inspect checkpoints:"
echo "     ls -lh $OUTPUT_DIR/stage*/checkpoints/"
echo ""
