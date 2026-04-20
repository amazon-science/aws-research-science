#!/usr/bin/env python3
"""
Training Stability Test Framework

This script validates training components incrementally to prevent failures during long runs.
It performs a series of progressive tests with different configurations to ensure
all features work properly before starting a full training run.

Key features:
- System resource checking
- Process cleanup and GPU memory clearing
- Progressive testing of training features
- Detailed reporting and error analysis
"""

import os
import sys
import time
import subprocess
import json
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
import torch
import psutil
import GPUtil
import shutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level for more verbose logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_stability_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure PyTorch distributed debug settings
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # Enable detailed distributed logging
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"    # Show C++ stack traces

class TrainingStabilityTester:
    """
    Test framework to validate LLaMA 3.2 1B training stability through
    a series of incremental tests with increasing complexity.
    """
    
    def __init__(self, base_dir="./test_runs"):
        """
        Initialize the stability tester with test configurations.
        
        Args:
            base_dir (str): Base directory for test outputs and logs
        """
        self.base_dir = base_dir
        self.test_results = {}
        self.start_time = time.time()
        
        # Create test directories
        os.makedirs(base_dir, exist_ok=True)
        
        # Test configurations - progressive complexity
        self.test_configs = {
            "basic": {
                "max_steps": 10,
                "eval_steps": None,  # No evaluation
                "save_steps": None,  # No checkpointing
                "description": "Basic training loop validation"
            },
            "with_eval": {
                "max_steps": 50,
                "eval_steps": 25,
                "save_steps": None,
                "description": "Training with evaluation steps"
            },
            "with_checkpoints": {
                "max_steps": 10,
                "eval_steps": 4,
                "save_steps": 5,
                "description": "Training with checkpointing"
            },
            "full_features": {
                "max_steps": 100,
                "eval_steps": 50,
                "save_steps": 50,
                "description": "Full training with all features"
            },
            "extended": {
                "max_steps": 500,
                "eval_steps": 100,
                "save_steps": 100,
                "description": "Extended training test"
            }
        }
    
    def check_system_resources(self):
        """
        Check and log system resources before testing.
        
        This function checks:
        - CPU usage and memory
        - GPU memory, utilization, and temperature
        - CUDA availability and device properties
        
        Returns:
            dict: Dictionary containing system resource information
        """
        logger.info("=== System Resource Check ===")
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        logger.info(f"CPU Usage: {cpu_percent}%")
        logger.info(f"Memory: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent}%)")
        
        # GPU info
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                logger.info(f"GPU {i}: {gpu.name}")
                logger.info(f"  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
                logger.info(f"  Utilization: {gpu.load*100:.1f}%")
                logger.info(f"  Temperature: {gpu.temperature}°C")
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
        
        # Check CUDA availability
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Device {i}: {props.name}")
                logger.info(f"    Total memory: {props.total_memory/1024**3:.1f}GB")
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def cleanup_processes(self):
        """
        Kill any existing training processes and clear GPU memory.
        
        This ensures a clean environment before starting tests,
        preventing interference from previous runs.
        """
        logger.info("Cleaning up existing processes...")
        
        processes_to_kill = [
            "torchrun",
            "python.*pre-train-llama3.2-1b.py"
        ]
        
        for proc_pattern in processes_to_kill:
            try:
                subprocess.run(
                    f"pkill -f '{proc_pattern}'", 
                    shell=True, 
                    check=False,
                    capture_output=True
                )
            except Exception as e:
                logger.warning(f"Error killing processes: {e}")
        
        # Wait for processes to terminate
        time.sleep(3)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
            except Exception as e:
                logger.warning(f"Error clearing GPU memory: {e}")
    
    def log_gpu_memory_stats(self):
        """
        Capture and log detailed GPU memory information.
        
        Returns:
            dict: Current GPU memory usage statistics
        """
        memory_stats = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # Get PyTorch stats
                    allocated = torch.cuda.memory_allocated(i) / (1024**2)
                    reserved = torch.cuda.memory_reserved(i) / (1024**2)
                    max_allocated = torch.cuda.max_memory_allocated(i) / (1024**2)
                    
                    memory_stats[f"gpu_{i}"] = {
                        "allocated_mb": allocated,
                        "reserved_mb": reserved,
                        "max_allocated_mb": max_allocated
                    }
                    
                    logger.debug(f"GPU {i} Memory: "
                                f"Allocated: {allocated:.1f}MB, "
                                f"Reserved: {reserved:.1f}MB, "
                                f"Peak: {max_allocated:.1f}MB")
                except Exception as e:
                    logger.warning(f"Error getting memory stats for GPU {i}: {e}")
        
        # Also get nvidia-smi stats
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                if f"gpu_{i}" not in memory_stats:
                    memory_stats[f"gpu_{i}"] = {}
                    
                memory_stats[f"gpu_{i}"].update({
                    "used_mb": gpu.memoryUsed,
                    "total_mb": gpu.memoryTotal,
                    "utilization_pct": gpu.load * 100
                })
                    
                logger.debug(f"GPU {i} (nvidia-smi): "
                            f"Used: {gpu.memoryUsed}MB, "
                            f"Total: {gpu.memoryTotal}MB, "
                            f"Util: {gpu.load*100:.1f}%")
        except Exception as e:
            logger.warning(f"Error getting nvidia-smi GPU stats: {e}")
            
        return memory_stats
    
    def monitor_directory_size(self, dir_path):
        """
        Get the size of a directory and its contents.
        
        Args:
            dir_path: Path to the directory to check
        
        Returns:
            tuple: (size_bytes, file_count)
        """
        total_size = 0
        file_count = 0
        
        try:
            for path in Path(dir_path).rglob("*"):
                if path.is_file():
                    file_count += 1
                    total_size += path.stat().st_size
                    
            logger.debug(f"Directory {dir_path}: {total_size/(1024**2):.1f}MB, {file_count} files")
            return (total_size, file_count)
        except Exception as e:
            logger.warning(f"Error monitoring directory {dir_path}: {e}")
            return (0, 0)

    def run_training_test(self, test_name, config, scheduler="cosine"):
        """
        Run a single training test with given configuration.
        
        Args:
            test_name: Name of the test (e.g., "basic", "with_eval")
            config: Dictionary containing test parameters
            scheduler: Learning rate scheduler to use ("cosine" or "greedy")
            
        Returns:
            dict: Test result data including success status and metrics
        """
        logger.info(f"=== Running Test: {test_name} ===")
        logger.info(f"Description: {config['description']}")
        logger.info(f"Configuration: {config}")
        
        test_start_time = time.time()
        
        # Create test-specific directory
        test_dir = os.path.join(self.base_dir, f"{test_name}_{scheduler}")
        if os.path.exists(test_dir):
            logger.info(f"Clearing previous test directory: {test_dir}")
            try:
                # Remove output directory to ensure clean test
                shutil.rmtree(os.path.join(test_dir, "output"), ignore_errors=True)
                shutil.rmtree(os.path.join(test_dir, "tensorboard"), ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error clearing previous test outputs: {e}")
                
        os.makedirs(test_dir, exist_ok=True)
        
        # Create process-specific log directories
        logs_dir = os.path.join(test_dir, "process_logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create error output directory for distributed logs
        error_log_dir = os.path.join(test_dir, "error_logs")
        os.makedirs(error_log_dir, exist_ok=True)
        
        # Log initial system state
        logger.info("=== Initial System State ===")
        self.log_gpu_memory_stats()
        
        # Prepare command - use the test version of the script
        cmd = [
            "torchrun",
            "--nproc_per_node=4",
            "--log_dir", logs_dir,
            "pre-train-llama3.2-1b-test.py",
            "--mode", "multi",
            "--lr_scheduler", scheduler
        ]
        
        # Set environment variables for this test
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": os.getcwd(),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            # Override some training parameters via environment
            "TEST_MAX_STEPS": str(config["max_steps"]),
            "TEST_EVAL_STEPS": str(config["eval_steps"]) if config["eval_steps"] else "None",
            "TEST_SAVE_STEPS": str(config["save_steps"]) if config["save_steps"] else "None",
            "TEST_OUTPUT_DIR": test_dir,
            
            # Enhanced debugging and tracing
            "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
            "TORCH_SHOW_CPP_STACKTRACES": "1",
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "ALL",
            "TORCHELASTIC_ERROR_FILE": os.path.join(error_log_dir, "distributed_error_%r.log"),
            
            # Additional debugging flags
            "TORCH_CPP_LOG_LEVEL": "INFO",
            "TORCH_DISTRIBUTED_DETAIL": "DETAIL",
            "TEST_ENABLE_MEMORY_TRACKING": "1",
            "TEST_CHECKPOINT_DEBUG": "1"
        })
        
        # Log checkpoint directories before running
        output_dir = os.path.join(test_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        self.monitor_directory_size(output_dir)
        
        try:
            # Run the training command
            logger.info(f"Executing: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=os.getcwd()
            )
            
            # Monitor the process
            output_lines = []
            memory_tracking = []
            checkpoint_steps = set()  # Track steps where checkpoints happen
            evaluation_steps = set()  # Track steps where evaluations happen
            
            # Key events to watch for
            checkpoint_patterns = [
                "Saving model checkpoint",
                "Model checkpoint saved",
                "Error saving checkpoint",
                "Saving optimizer",
                "Checkpoint complete"
            ]
            
            eval_patterns = [
                "Running evaluation",
                "Evaluation complete",
                "Computing metrics",
                "Error during evaluation"
            ]
            
            current_step = 0
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    line = output.strip()
                    output_lines.append(line)
                    logger.info(f"[{test_name}] {line}")
                    
                    # Extract current step if available
                    if "step " in line:
                        try:
                            step_info = line.split("step ")[1].split()[0]
                            if step_info.isdigit():
                                current_step = int(step_info)
                        except:
                            pass
                    
                    # Track checkpoint operations
                    for pattern in checkpoint_patterns:
                        if pattern in line:
                            checkpoint_steps.add(current_step)
                            # Log memory usage during checkpoint operations
                            memory_snapshot = self.log_gpu_memory_stats()
                            memory_tracking.append({
                                "step": current_step,
                                "event": "checkpoint",
                                "memory_stats": memory_snapshot,
                                "time": datetime.now().isoformat()
                            })
                            # Log checkpoint directory size
                            dir_size = self.monitor_directory_size(output_dir)
                            logger.info(f"[CHECKPOINT] Step {current_step}, "
                                        f"Output dir size: {dir_size[0]/(1024**2):.1f}MB, "
                                        f"{dir_size[1]} files")
                            break
                            
                    # Track evaluation operations
                    for pattern in eval_patterns:
                        if pattern in line:
                            evaluation_steps.add(current_step)
                            # Log memory usage during evaluation
                            memory_snapshot = self.log_gpu_memory_stats()
                            memory_tracking.append({
                                "step": current_step,
                                "event": "evaluation",
                                "memory_stats": memory_snapshot,
                                "time": datetime.now().isoformat()
                            })
                            break
                    
                    # Periodic memory tracking (every 10 lines)
                    if len(output_lines) % 10 == 0:
                        memory_snapshot = self.log_gpu_memory_stats()
                        memory_tracking.append({
                            "step": current_step,
                            "event": "periodic",
                            "memory_stats": memory_snapshot,
                            "time": datetime.now().isoformat()
                        })
            
            # Wait for completion
            return_code = process.wait()
            
            test_duration = time.time() - test_start_time
            
            # Check for process log files and distributed error logs
            process_logs_dir = os.path.join(test_dir, "process_logs")
            error_logs_dir = os.path.join(test_dir, "error_logs")
            
            process_log_files = []
            error_log_files = []
            
            if os.path.exists(process_logs_dir):
                process_log_files = os.listdir(process_logs_dir)
                logger.info(f"Process log files: {process_log_files}")
                
                # Capture contents of process logs
                for log_file in process_log_files:
                    try:
                        with open(os.path.join(process_logs_dir, log_file), 'r') as f:
                            logger.info(f"=== Contents of {log_file} ===")
                            logger.info(f.read())
                    except Exception as e:
                        logger.error(f"Error reading process log {log_file}: {e}")
            
            if os.path.exists(error_logs_dir):
                error_log_files = os.listdir(error_logs_dir)
                logger.info(f"Error log files: {error_log_files}")
                
                # Capture contents of error logs
                for log_file in error_log_files:
                    try:
                        with open(os.path.join(error_logs_dir, log_file), 'r') as f:
                            logger.info(f"=== Contents of {log_file} ===")
                            logger.info(f.read())
                    except Exception as e:
                        logger.error(f"Error reading error log {log_file}: {e}")
            
            # Final memory check
            final_memory = self.log_gpu_memory_stats()
            
            # Analyze results
            success = return_code == 0
            
            # Collect detailed information about any checkpoint-related files
            checkpoint_files = []
            if os.path.exists(output_dir):
                for item in os.listdir(output_dir):
                    if item.startswith('checkpoint-'):
                        checkpoint_path = os.path.join(output_dir, item)
                        try:
                            stats = os.stat(checkpoint_path)
                            checkpoint_files.append({
                                "name": item,
                                "size": stats.st_size,
                                "is_dir": os.path.isdir(checkpoint_path),
                                "contents": os.listdir(checkpoint_path) if os.path.isdir(checkpoint_path) else None
                            })
                        except Exception as e:
                            checkpoint_files.append({
                                "name": item,
                                "error": str(e)
                            })
            
            result = {
                "test_name": test_name,
                "scheduler": scheduler,
                "config": config,
                "success": success,
                "return_code": return_code,
                "duration": test_duration,
                "output_lines": output_lines[-50:],  # Keep last 50 lines
                "checkpoint_steps": list(checkpoint_steps),
                "evaluation_steps": list(evaluation_steps),
                "memory_tracking": memory_tracking,
                "final_memory": final_memory,
                "process_logs": process_log_files,
                "error_logs": error_log_files,
                "checkpoint_files": checkpoint_files,
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                logger.info(f"✅ Test {test_name} PASSED ({test_duration:.1f}s)")
            else:
                logger.error(f"❌ Test {test_name} FAILED ({test_duration:.1f}s)")
                logger.error(f"Return code: {return_code}")
                # Log error output
                for line in output_lines[-10:]:
                    logger.error(f"  {line}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Test {test_name} FAILED with exception: {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            
            # Capture memory state at failure
            failure_memory = self.log_gpu_memory_stats()
            
            return {
                "test_name": test_name,
                "scheduler": scheduler,
                "config": config,
                "success": False,
                "error": str(e),
                "error_traceback": traceback.format_exc(),
                "memory_state_at_failure": failure_memory,
                "duration": time.time() - test_start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def modify_training_script_for_testing(self):
        """
        Create a test version of the training script with configurable parameters.
        
        This function:
        1. Creates a modified copy of the training script
        2. Adds environment variable-based parameter overrides
        3. Adds dataset size limiting for testing
        4. Configures test-specific output directories
        5. Adds enhanced debugging and memory tracking
        """
        logger.info("Creating test version of training script...")
        
        # Read the original script
        with open("pre-train-llama3.2-1b.py", "r") as f:
            script_content = f.read()
        
        # Add import for memory tracking
        if "import torch.cuda" not in script_content:
            script_content = script_content.replace(
                "import torch", 
                "import torch\nimport torch.cuda\nimport gc\nimport psutil\nimport os.path"
            )
        
        # Add test parameter overrides - place right after training_args_dict definition
        test_overrides = '''
    
    # TEST PARAMETER OVERRIDES - Applied via environment variables
    
    # Memory tracking function
    def log_memory_usage(tag):
        """Log memory usage at specific points in the training process"""
        try:
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            gpu_info = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**2)
                    reserved = torch.cuda.memory_reserved(i) / (1024**2)
                    gpu_info.append(f"GPU {i}: {allocated:.1f}MB alloc, {reserved:.1f}MB reserv")
            
            logger.info(f"MEMORY_TRACKING [{tag}] - Process: {process_memory:.1f}MB, {', '.join(gpu_info)}")
            
            # Only force garbage collection during critical operations to avoid performance impact
            if tag in ["pre_checkpoint", "post_checkpoint", "pre_eval", "post_eval"]:
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"MEMORY_TRACKING [{tag}] - Garbage collection performed")
        except Exception as e:
            logger.warning(f"Memory tracking error [{tag}]: {e}")
    if "TEST_MAX_STEPS" in os.environ:
        training_args_dict["max_steps"] = int(os.environ["TEST_MAX_STEPS"])
        logger.info(f"🧪 TEST OVERRIDE: max_steps = {training_args_dict['max_steps']}")

    if "TEST_EVAL_STEPS" in os.environ and os.environ["TEST_EVAL_STEPS"] != "None":
        training_args_dict["eval_steps"] = int(os.environ["TEST_EVAL_STEPS"])
        training_args_dict["eval_strategy"] = "steps"
        logger.info(f"🧪 TEST OVERRIDE: eval_steps = {training_args_dict['eval_steps']}")
    elif "TEST_EVAL_STEPS" in os.environ and os.environ["TEST_EVAL_STEPS"] == "None":
        training_args_dict["eval_strategy"] = "no"
        logger.info("🧪 TEST OVERRIDE: evaluation disabled")

    if "TEST_SAVE_STEPS" in os.environ and os.environ["TEST_SAVE_STEPS"] != "None":
        training_args_dict["save_steps"] = int(os.environ["TEST_SAVE_STEPS"])
        training_args_dict["save_strategy"] = "steps"
        logger.info(f"🧪 TEST OVERRIDE: save_steps = {training_args_dict['save_steps']}")
    elif "TEST_SAVE_STEPS" in os.environ and os.environ["TEST_SAVE_STEPS"] == "None":
        training_args_dict["save_strategy"] = "no"
        logger.info("🧪 TEST OVERRIDE: evaluation disabled")

    # Always set save_total_limit to 1 for testing to ensure we only keep the latest checkpoint
    training_args_dict["save_total_limit"] = 1
    logger.info("🧪 TEST OVERRIDE: save_total_limit = 1")
    
    # Enable detailed error reporting for checkpoint saving
    training_args_dict["hub_always_push"] = False
    training_args_dict["save_safetensors"] = False  # Simpler saving format
    training_args_dict["report_to"] = []  # Disable extra reporters
    
    # Memory optimizations for testing
    training_args_dict["fp16"] = True  # Use mixed precision to save memory
    training_args_dict["gradient_checkpointing"] = True  # Enable gradient checkpointing
    training_args_dict["optim"] = "adamw_8bit"  # Use 8-bit optimizer
    
    if "TEST_CHECKPOINT_DEBUG" in os.environ:
        # Add extra memory debugging
        training_args_dict["debug"] = "underflow_overflow"
        training_args_dict["logging_strategy"] = "steps"
        training_args_dict["logging_steps"] = 1
        logger.info("🧪 TEST OVERRIDE: Debug mode enabled")

    if "TEST_OUTPUT_DIR" in os.environ:
        # Override the entire directory structure for testing
        test_base_dir = os.environ["TEST_OUTPUT_DIR"]
        output_dir = os.path.join(test_base_dir, "output")
        logging_dir = os.path.join(test_base_dir, "tensorboard")
        model_dir = os.path.join(test_base_dir, "model")
        config_dir = os.path.join(test_base_dir, "configs")
        
        # Update all directory paths
        training_args_dict["output_dir"] = output_dir
        training_args_dict["logging_dir"] = logging_dir
        
        # Create test directories
        for dir_path in [logging_dir, output_dir, model_dir, config_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        logger.info(f"🧪 TEST OVERRIDE: Using test directory structure at {test_base_dir}")
'''
        
        # Find the right insertion point after training_args_dict definition
        insert_point = '    # Add common parameters - disabled debug mode to save memory'
        if insert_point in script_content:
            script_content = script_content.replace(insert_point, test_overrides + "\n" + insert_point)
        else:
            # Fallback insertion point
            insert_point = "training_args_dict.update({"
            script_content = script_content.replace(insert_point, test_overrides + "\n    " + insert_point)
        
        # Add checkpoint debugging hooks to Trainer - to be added after Trainer import
        trainer_callback_code = '''
# Add the memory tracking callback
from transformers.trainer_callback import TrainerCallback

class MemoryTrackingCallback(TrainerCallback):
    """Custom callback for tracking memory usage during training."""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 2 == 0:  # Track every few steps
            log_memory_usage(f"step_{state.global_step}")
    
    def on_evaluate(self, args, state, control, **kwargs):
        log_memory_usage("pre_eval")
        
    def on_evaluate_end(self, args, state, control, **kwargs):
        log_memory_usage("post_eval")
        
    def on_save(self, args, state, control, **kwargs):
        log_memory_usage("pre_checkpoint")
        # Check if checkpoint path exists and log its contents
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        logger.info(f"Checkpoint about to be saved at: {checkpoint_dir}")
        if os.path.exists(args.output_dir):
            existing_files = os.listdir(args.output_dir)
            logger.info(f"Output directory contents: {existing_files}")
    
    def on_save_end(self, args, state, control, **kwargs):
        log_memory_usage("post_checkpoint")
'''

        # Add the callback registration code right after Trainer initialization
        trainer_init_code = '''
# Register the memory tracking callback
trainer.add_callback(MemoryTrackingCallback())
logger.info("Memory tracking callback registered")
'''

        # Find the Trainer import location to insert our callback class
        if "from transformers import Trainer" in script_content:
            script_content = script_content.replace(
                "from transformers import Trainer",
                "from transformers import Trainer\n" + trainer_callback_code
            )
        elif "from transformers import" in script_content:
            # Add to existing transformers import
            script_content = script_content.replace(
                "from transformers import",
                "from transformers import TrainerCallback, "
            )
            # Then add our callback class after imports
            import_end = script_content.find("\n\n", script_content.find("import"))
            if import_end > 0:
                script_content = script_content[:import_end] + "\n\n" + trainer_callback_code + script_content[import_end:]
        
        # Find Trainer initialization to add our callback registration
        if "trainer = Trainer(" in script_content:
            # Find the end of trainer initialization
            trainer_init_end = script_content.find("\n", script_content.find("trainer = Trainer("))
            if trainer_init_end > 0:
                script_content = script_content[:trainer_init_end] + "\n" + trainer_init_code + script_content[trainer_init_end:]
        
        # Write the test version
        with open("pre-train-llama3.2-1b-test.py", "w") as f:
            f.write(script_content)
        
        logger.info("Test script created: pre-train-llama3.2-1b-test.py")
    
    def run_all_tests(self, schedulers=["cosine", "greedy"], single_test=None):
        """
        Run stability tests for the specified schedulers.
        
        This function:
        1. Prepares the test environment
        2. Creates a testing version of the training script
        3. Runs tests for each scheduler in sequence
        4. Generates a comprehensive report
        
        Args:
            schedulers: List of LR schedulers to test (cosine, greedy)
            single_test: If specified, only run this specific test
            
        Returns:
            list: Results of all tests run
        """
        logger.info("🚀 Starting Training Stability Tests")
        logger.info(f"Testing schedulers: {schedulers}")
        
        if single_test:
            logger.info(f"Running only the '{single_test}' test")
        
        # System resource check
        system_info = self.check_system_resources()
        
        # Cleanup before starting
        self.cleanup_processes()
        
        # Modify training script for testing
        self.modify_training_script_for_testing()
        
        all_results = []
        
        for scheduler in schedulers:
            logger.info(f"\n=== Testing Scheduler: {scheduler.upper()} ===")
            
            if single_test:
                # Run only the specified test
                if single_test in self.test_configs:
                    self.cleanup_processes()
                    time.sleep(5)
                    
                    result = self.run_training_test(single_test, self.test_configs[single_test], scheduler)
                    all_results.append(result)
                else:
                    logger.error(f"Test '{single_test}' not found in test configurations")
                    continue
            else:
                # Original code to run all tests
                for test_name, config in self.test_configs.items():
                    # Cleanup between tests
                    self.cleanup_processes()
                    time.sleep(5)  # Wait for cleanup
                    
                    # Run the test
                    result = self.run_training_test(test_name, config, scheduler)
                    all_results.append(result)
                    
                    # Stop if a test fails (unless it's the basic test)
                    if not result["success"] and test_name != "basic":
                        logger.error(f"Test {test_name} failed, stopping further tests for {scheduler}")
                        break
                    
                    # Wait between tests
                    time.sleep(3)
        
        # Generate final report
        self.generate_report(all_results, system_info)
        
        return all_results
    
    def generate_report(self, results, system_info):
        """Generate a comprehensive test report."""
        logger.info("\n" + "="*60)
        logger.info("TRAINING STABILITY TEST REPORT")
        logger.info("="*60)
        
        total_duration = time.time() - self.start_time
        
        # Summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        logger.info(f"Total Duration: {total_duration:.1f}s")
        
        # Individual test results
        logger.info("\nDetailed Results:")
        for result in results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            logger.info(f"  {result['test_name']} ({result['scheduler']}): {status} ({result.get('duration', 0):.1f}s)")
            if not result["success"]:
                logger.info(f"    Error: {result.get('error', 'Unknown error')}")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests/total_tests*100,
                "total_duration": total_duration
            },
            "detailed_results": results
        }
        
        report_file = os.path.join(self.base_dir, "stability_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        # Recommendations
        logger.info("\nRecommendations:")
        if failed_tests == 0:
            logger.info("✅ All tests passed! Training should be stable for full runs.")
        else:
            logger.info("⚠️  Some tests failed. Check the following:")
            logger.info("   - GPU memory usage during evaluation")
            logger.info("   - NCCL communication stability")
            logger.info("   - Dataset loading performance")
            logger.info("   - DeepSpeed configuration")

def main():
    parser = argparse.ArgumentParser(description="Training Stability Test Framework")
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=["cosine", "greedy"],
        choices=["cosine", "greedy"],
        help="Schedulers to test"
    )
    parser.add_argument(
        "--test-dir",
        default="./test_runs",
        help="Directory for test outputs"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        choices=["basic", "with_eval", "with_checkpoints", "full_features", "extended"],
        help="Run only the specified test configuration (runs all by default)"
    )
    
    args = parser.parse_args()
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create and run tester
    tester = TrainingStabilityTester(args.test_dir)
    results = tester.run_all_tests(args.schedulers, single_test=args.test)
    
    # Exit with appropriate code
    success_count = sum(1 for r in results if r["success"])
    if success_count == len(results):
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error(f"💥 {len(results) - success_count} tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
