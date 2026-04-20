#!/bin/bash
# 
# Sequential LLaMA Training Script with GPU Memory Management
#
# This script runs Greedy LR followed by Cosine LR training with robust
# GPU memory management between phases, ensuring clean transitions and
# preventing resource conflicts or memory leaks.
#
# Author: HuggingFace Greedy-LR Team
# Last updated: June 2025

set -e  # Exit on any error

# Configuration
SCRIPT="pre-train-llama3.2-1b.py"
NUM_GPUS=4
CLEANUP_WAIT=30
MEMORY_THRESHOLD_GB=2.0  # GPU memory should be below this before starting next phase

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions with consistent timestamp format
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to check if conda environment is activated
check_environment() {
    
    # Verify that the correct conda environment is activated.
    # Exits with error code 1 if environment is not activated.
    
    log_info "Checking conda environment..."
    if [[ "$CONDA_DEFAULT_ENV" != "pytorch_p310_greedy_v2" ]]; then
        log_error "Please activate the required conda environment first:"
        log_error "conda activate pytorch_p310_greedy_v2"
        exit 1
    fi
    log_success "Conda environment verified: $CONDA_DEFAULT_ENV"
}

# Function to get GPU memory usage in GB
get_gpu_memory_usage() {
   
    # Calculate total GPU memory usage across all GPUs.
    
    # Returns:
    #     Total GPU memory usage in GB.
    
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/1024}'
}

# Function to display current GPU status
show_gpu_status() {
    
    # Display detailed GPU status information including memory usage and utilization.
    # Shows information for each GPU and total memory consumption.
    
    log_info "Current GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    while IFS=, read -r gpu_id name mem_used mem_total util; do
        echo "  GPU $gpu_id ($name): ${mem_used}MB/${mem_total}MB (${util}% util)"
    done
    
    local total_memory=$(get_gpu_memory_usage)
    echo "  Total GPU Memory Used: ${total_memory}GB"
}

# Function to kill training processes
kill_training_processes() {
    
    # Terminate all running training processes, including torchrun, python and DeepSpeed.
    # Uses a two-stage termination approach (graceful, then forced) to ensure cleanup.
    
    log_info "Terminating any existing training processes..."
    
    # First attempt: Graceful termination
    pkill -f "torchrun.*$SCRIPT" 2>/dev/null || true
    pkill -f "python.*$SCRIPT" 2>/dev/null || true
    pkill -f "deepspeed" 2>/dev/null || true
    
    # Wait for processes to terminate
    sleep 5
    
    # Second attempt: Force kill if still running
    pkill -9 -f "torchrun.*$SCRIPT" 2>/dev/null || true
    pkill -9 -f "python.*$SCRIPT" 2>/dev/null || true
    pkill -9 -f "deepspeed" 2>/dev/null || true
    
    log_success "Training processes terminated"
}

# Function to clear GPU memory
clear_gpu_memory() {
    
    # Clear GPU memory caches on all devices using PyTorch.
    # Also attempts a GPU reset if permissions allow.
    
    log_info "Clearing GPU memory caches..."
    
    # Clear CUDA cache using Python
    python3 -c "
import torch
import gc
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    print('CUDA cache cleared on all devices')
else:
    print('CUDA not available')
" 2>/dev/null || log_warning "Failed to clear CUDA cache via Python"
    
    # Reset GPU state (requires elevated permissions, might fail)
    nvidia-smi --gpu-reset 2>/dev/null || log_warning "GPU reset not available (requires root)"
    
    log_success "GPU memory clearing completed"
}

# Function to verify memory is cleared
verify_memory_cleared() {
    log_info "Verifying GPU memory is cleared..."
    
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        local current_memory=$(get_gpu_memory_usage)
        log_info "Attempt $attempt/$max_attempts: GPU memory usage = ${current_memory}GB"
        
        # Convert to integer comparison (multiply by 1000 to handle decimals)
        local current_mb=$(echo "$current_memory * 1000" | awk '{print int($1)}')
        local threshold_mb=$(echo "$MEMORY_THRESHOLD_GB * 1000" | awk '{print int($1)}')
        
        if [ $current_mb -lt $threshold_mb ]; then
            log_success "GPU memory sufficiently cleared (${current_memory}GB < ${MEMORY_THRESHOLD_GB}GB threshold)"
            return 0
        fi
        
        log_warning "GPU memory still high (${current_memory}GB). Waiting..."
        sleep 3
        ((attempt++))
    done
    
    log_error "Failed to clear GPU memory below threshold after $max_attempts attempts"
    log_error "Current: ${current_memory}GB, Threshold: ${MEMORY_THRESHOLD_GB}GB"
    return 1
}

# Function to perform comprehensive cleanup
comprehensive_cleanup() {
    log_info "Performing comprehensive cleanup..."
    
    kill_training_processes
    sleep 2
    clear_gpu_memory
    sleep 2
    
    if ! verify_memory_cleared; then
        log_warning "Memory verification failed, but continuing..."
    fi
    
    log_info "Waiting ${CLEANUP_WAIT} seconds for system stabilization..."
    sleep $CLEANUP_WAIT
    
    show_gpu_status
    log_success "Comprehensive cleanup completed"
}

# Function to find latest checkpoint for a given scheduler
find_latest_checkpoint() {
    local scheduler=$1
    local custom_dir=$2
    local output_dir
    
    if [[ -n "$custom_dir" ]]; then
        # Use custom directory provided as argument
        if [[ "$custom_dir" == *"/output" ]]; then
            # If the path already ends with /output, use it directly
            output_dir="$custom_dir"
        else
            # Look for /output inside the provided path
            if [[ -d "$custom_dir/output" ]]; then
                output_dir="$custom_dir/output"
            else
                # Otherwise, use the directory as is (might already contain checkpoints)
                output_dir="$custom_dir"
            fi
        fi
    else
        # Use default directory structure with current date
        output_dir="./logs/redpajama/meta-llama/Llama-3.2-1B/run_3/$scheduler/$(date +%Y-%m-%d)/output"
    fi
    
    log_info "Searching for checkpoints in: $output_dir"
    
    if [[ -d "$output_dir" ]]; then
        # Find highest numbered checkpoint
        local latest=$(ls -1d "$output_dir"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [[ -n "$latest" ]]; then
            local step_num=$(basename "$latest" | cut -d'-' -f2)
            log_info "Found checkpoint at step $step_num"
            echo "$latest"
        else
            log_info "No checkpoint directories found in $output_dir"
        fi
    else
        log_info "Directory not found: $output_dir"
    fi
}

# Function to run a training phase with checkpoint resume support
run_training_phase() {
    local scheduler=$1
    local phase_name=$2
    local custom_dir=$3
    
    log_info "Starting $phase_name (Scheduler: $scheduler)"
    
    # Check for existing checkpoint
    local checkpoint=$(find_latest_checkpoint "$scheduler" "$custom_dir")
    local resume_info=""
    
    if [[ -n "$checkpoint" ]]; then
        local step_num=$(basename "$checkpoint" | cut -d'-' -f2)
        log_info "🔄 Found existing checkpoint: $checkpoint"
        log_info "🔄 Will resume $phase_name from step $step_num"
        resume_info=" (resuming from step $step_num)"
    else
        if [[ -n "$custom_dir" ]]; then
            log_info "🆕 No existing checkpoint found for $scheduler in custom directory: $custom_dir"
        else
            log_info "🆕 No existing checkpoint found for $scheduler"
        fi
        log_info "🆕 Starting $phase_name from beginning"
        resume_info=" (fresh start)"
    fi
    
    # Build command with checkpoint directory if specified
    local cmd="torchrun --nproc_per_node=$NUM_GPUS $SCRIPT --mode multi --lr_scheduler $scheduler"
    if [[ -n "$custom_dir" ]]; then
        cmd="$cmd --checkpoint_dir $custom_dir"
        log_info "Command: $cmd (with custom checkpoint directory)"
    else
        log_info "Command: $cmd (using default directory structure)"
    fi
    log_info "Training will auto-detect and resume from checkpoint if available"
    
    # Show initial GPU status
    show_gpu_status
    
    # Run training
    if eval "$cmd"; then
        log_success "$phase_name completed successfully$resume_info"
        return 0
    else
        local exit_code=$?
        log_error "$phase_name failed with exit code $exit_code$resume_info"
        return $exit_code
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -d, --directory <path>    Directory path to look for existing checkpoints"
    echo "                           (e.g., './logs/redpajama/meta-llama/Llama-3.2-1B/run_3/greedy/2025-06-20')"
    echo "  -h, --help               Show this help message"
    echo
    echo "Examples:"
    echo "  $0                                                    # Use default directory structure"
    echo "  $0 -d ./logs/redpajama/meta-llama/Llama-3.2-1B/run_3/greedy/2025-06-20"
    echo "  $0 --directory ./logs/redpajama/meta-llama/Llama-3.2-1B/run_3/greedy/2025-06-20"
}

# Parse command line arguments
parse_arguments() {
    CUSTOM_DIR=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--directory)
                CUSTOM_DIR="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Main execution function
main() {
    
    # Main execution function for sequential training.
    
    # Workflow:
    # 1. Parse command line arguments
    # 2. Perform pre-flight environment checks
    # 3. Clean up any existing processes
    # 4. Run Phase 1: Greedy LR Training
    # 5. Perform inter-phase cleanup
    # 6. Run Phase 2: Cosine LR Training
    # 7. Perform final cleanup
    
    # Parse command line arguments
    parse_arguments "$@"
    
    log_info "=== Sequential LLaMA Training Started ==="
    log_info "Training phases: Greedy LR → Cosine LR"
    log_info "Number of GPUs: $NUM_GPUS"
    
    if [[ -n "$CUSTOM_DIR" ]]; then
        log_info "Using custom directory for checkpoint detection: $CUSTOM_DIR"
    else
        log_info "Using default directory structure with current date"
    fi
    
    # Pre-flight checks
    check_environment
    
    # Initial cleanup to ensure clean start
    log_info "Performing initial cleanup..."
    comprehensive_cleanup
    
    # Phase 1: Greedy LR Training
    log_info "=== PHASE 1: GREEDY LR TRAINING ==="
    if ! run_training_phase "greedy" "Greedy LR Training" "$CUSTOM_DIR"; then
        log_error "Phase 1 failed. Stopping execution."
        exit 1
    fi
    
    # Cleanup between phases
    log_info "=== INTER-PHASE CLEANUP ==="
    comprehensive_cleanup
    
    # Phase 2: Cosine LR Training
    log_info "=== PHASE 2: COSINE LR TRAINING ==="
    if ! run_training_phase "cosine" "Cosine LR Training" ""; then
        log_error "Phase 2 failed."
        exit 1
    fi
    
    # Final cleanup
    log_info "=== FINAL CLEANUP ==="
    comprehensive_cleanup
    
    log_success "=== ALL TRAINING PHASES COMPLETED SUCCESSFULLY ==="
    log_info "Check the following directories for results:"
    log_info "  - Greedy LR: ./logs/redpajama/meta-llama/Llama-3.2-1B/run_3/greedy/"
    log_info "  - Cosine LR: ./logs/redpajama/meta-llama/Llama-3.2-1B/run_3/cosine/"
}

# Error handling
trap 'log_error "Script interrupted or failed. Performing cleanup..."; comprehensive_cleanup; exit 1' ERR INT TERM

# Check if script exists
if [[ ! -f "$SCRIPT" ]]; then
    log_error "Training script not found: $SCRIPT"
    log_error "Please run this script from the examples/greedy-lr directory"
    exit 1
fi

# Note: Using awk for floating point calculations (no bc dependency needed)

# Run main function
main "$@"
