#!/bin/bash
# Convenience script to run batch processing with common options

# Default values
DATASETS_ROOT="datasets"
OUTPUT_DIR="/opt/dlami/nvme/datasets/processed_droid"
LOG_FILE="batch_processing_$(date +%Y%m%d_%H%M%S).log"

# Detect number of CPU cores
NUM_CPUS=$(nproc)
# Use 75% of available CPUs (leave some for system)
NUM_WORKERS=$((NUM_CPUS * 1 / 4))
# Ensure at least 1 worker
if [ "$NUM_WORKERS" -lt 1 ]; then
    NUM_WORKERS=1
fi

echo "Detected $NUM_CPUS CPU cores, using $NUM_WORKERS parallel workers"

# Run the batch processing script
python batch_process_scenes.py \
    --datasets_root "$DATASETS_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --log_file "$LOG_FILE" \
    --num_workers "$NUM_WORKERS" \
    --validate_scenes \
    --delete_after_success

# Note: Add --dry_run to test without actually processing
# Note: Add --start_from N to resume from scene N
# Note: Add --limit N to process only N scenes
# Note: Adjust --num_workers N to control parallelism (default: auto-detect)
# Note: Remove --delete_after_success to keep original directories after successful processing
# Note: Remove --validate_scenes to skip scene validation and keep incomplete scenes
