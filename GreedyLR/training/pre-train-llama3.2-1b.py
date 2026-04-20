import os
import torch
import torch.nn as nn
import argparse
import sys
import datetime
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    LlamaConfig,
    LlamaForCausalLM,
    TrainerCallback
)
from datasets import load_from_disk
import logging
from tqdm.auto import tqdm
import json

# Create a custom callback to fix the progress bar
class TqdmFixCallback(TrainerCallback):
    """Fix the tqdm progress bar to start from the correct step."""
    
    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step > 0:
            # Hack the tqdm progress bar to start from the correct position
            for metric in kwargs.get("metrics", {}).values():
                if isinstance(metric, tqdm):
                    # Update the progress bar to show correct step
                    metric.n = state.global_step
                    metric.last_print_n = state.global_step
                    metric.refresh()
                    logger.info(f"🔄 Progress bar updated to start at step {state.global_step}")

# For running on Sagemaker Studio VSCode Instances
os.environ["CUDA_HOME"] = "/opt/conda"

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA pre-training with flexible GPU options")
    parser.add_argument(
        "--mode", 
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Training mode: 'single' for single GPU, 'multi' for multi-GPU with DeepSpeed"
    )
    parser.add_argument(
        "--num_gpus", 
        type=int,
        default=2,
        help="Number of GPUs to use in multi-GPU mode (default: 2)"
    )
    parser.add_argument(
        "--lr_scheduler", 
        type=str,
        default="cosine",
        choices=["greedy", "cosine"],
        help="Learning rate scheduler: 'greedy' for Greedy LR or 'cosine' for Cosine LR"
    )
    parser.add_argument(
        "--local_rank", 
        type=int,
        default=-1,
        help="Local rank passed from distributed launcher"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str,
        default=None,
        help="Path to checkpoint to resume from. If not specified, will auto-detect latest checkpoint."
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str,
        default=None,
        help="Base directory to use for training outputs (overrides date-based directory creation)."
    )
    return parser.parse_args()

# Generate DeepSpeed config based on parameters
def get_deepspeed_config(num_gpus, batch_size, grad_accum_steps, checkpoint_path=None):
    """Generate DeepSpeed config dict based on training parameters."""
    config = {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,  # ZeRO-3 for better memory efficiency
            "offload_optimizer": {"device": "none"},  # Disable CPU offloading to avoid compilation issues
            "offload_param": {"device": "none"},      # Disable parameter offloading
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }
    
    # Add load path if checkpoint is provided
    if checkpoint_path:
        load_dir = os.path.abspath(checkpoint_path)
        if os.path.exists(load_dir):
            config["load_path"] = load_dir
            logger.info(f"🔄 DEEPSPEED: Will load checkpoint from {load_dir}")
            
            # Add the global_step directory to the load path if it exists
            latest_file = os.path.join(load_dir, "latest")
            if os.path.exists(latest_file):
                with open(latest_file, "r") as f:
                    global_step_dir = f.read().strip()
                    global_step_path = os.path.join(load_dir, global_step_dir)
                    if os.path.exists(global_step_path):
                        config["load_path"] = global_step_path
                        logger.info(f"🔄 DEEPSPEED: Using global_step directory: {global_step_path}")
    
    return config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    if not os.path.exists(output_dir):
        return None
    
    # Look for checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        return None
    
    # Sort by step number and return the latest one
    def get_step_number(checkpoint_name):
        try:
            return int(checkpoint_name.split('-')[1])
        except (IndexError, ValueError):
            return 0
    
    latest_checkpoint = max(checkpoint_dirs, key=get_step_number)
    checkpoint_path = os.path.join(output_dir, latest_checkpoint)
    
    # Verify it's a valid checkpoint directory
    if os.path.isdir(checkpoint_path):
        return checkpoint_path
    return None

def setup_training(model_name):
    # Set random seed
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)
    
    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load prepared dataset
    # dataset_name = 'tokensized_dataset_train_50K_seed_42'
    dataset_name = "redpajama"
    dataset_variant = "redpajama_50K_seed_42"
    load_dir = f'./datasets/{dataset_name}/{dataset_variant}/'
    
    # Check if dataset exists
    if not os.path.exists(load_dir):
        logger.error(f"Dataset not found at {load_dir}. Please ensure the dataset is prepared.")
        raise FileNotFoundError(f"Dataset directory {load_dir} does not exist")
        
    tokenized_datasets = load_from_disk(load_dir)
    
    # 🚨 CRITICAL FIX: Limit validation dataset to prevent NCCL timeouts during evaluation
    # The full validation dataset (~10,520 samples) causes 13+ hour hangs during evaluation
    MAX_EVAL_SAMPLES = 500  # Reasonable size for meaningful evaluation without timeouts
    
    if len(tokenized_datasets['valid']) > MAX_EVAL_SAMPLES:
        original_eval_size = len(tokenized_datasets['valid'])
        tokenized_datasets['valid'] = tokenized_datasets['valid'].select(range(MAX_EVAL_SAMPLES))
        logger.info(f"🎯 EVALUATION DATASET LIMITED: {original_eval_size} → {MAX_EVAL_SAMPLES} samples")
        logger.info(f"This prevents NCCL timeouts during evaluation steps!")
    
    logger.info(f"Dataset columns: {tokenized_datasets['train'].column_names}")
    logger.info(f"Final dataset sizes - Train: {len(tokenized_datasets['train'])}, Valid: {len(tokenized_datasets['valid'])}")
    
    # Model configuration
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=2048,        # Appropriate for 1B parameter model
        intermediate_size=5632,  # Standard for LLaMA 3.2 1B
        num_hidden_layers=22,    # Appropriate for 1B parameter model
        num_attention_heads=16,  # Standard for this size
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,       # LLaMA 3 specific 
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        initializer_range=0.02,  # Important for stability
        use_cache=False,
        tie_word_embeddings=False,
        torch_dtype=torch.float32,  # Start with fp32, convert later if needed
    )
    
    # Initialize model with config
    try:
        logger.info(f"Initializing LLaMA model with config: {config}")
        model = LlamaForCausalLM(config)

        for layer in model.model.layers:
            # Common practice for transformer architectures
            nn.init.normal_(layer.self_attn.q_proj.weight, mean=0.0, std=config.initializer_range)
            nn.init.normal_(layer.self_attn.k_proj.weight, mean=0.0, std=config.initializer_range)
            nn.init.normal_(layer.self_attn.v_proj.weight, mean=0.0, std=config.initializer_range)
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=config.initializer_range)

        # Log model size
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {param_count/1e6:.2f}M parameters")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise
    
    return model, tokenizer, tokenized_datasets

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set memory optimization and NCCL timeout environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_TIMEOUT"] = "28800"  # 8 hour timeout for NCCL operations (increased from 2 hours)
    os.environ["NCCL_DEBUG"] = "INFO"    # Enable NCCL debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Allow async CUDA operations
    
    # Additional NCCL optimizations for stability
    os.environ["NCCL_IB_DISABLE"] = "1"      # Disable InfiniBand (can cause issues in some setups)
    os.environ["NCCL_P2P_DISABLE"] = "1"     # Disable P2P for stability
    os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Force ring algorithm for stability
    
    # Force NCCL timeout at process level (sometimes env vars aren't picked up)
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    # Configure GPU usage based on mode
    if args.mode == "single":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        use_deepspeed = False
        logger.info(f"Running in single-GPU mode")
    else:
        # For multi-GPU, let torchrun/distributed launch handle GPU assignment
        use_deepspeed = True
        # Don't set CUDA_VISIBLE_DEVICES in multi-GPU mode, as it's managed by the launcher
        logger.info(f"Running in multi-GPU mode with local_rank: {args.local_rank}")
    # Print environment information safely
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    # In multi-GPU mode, each process should only access its assigned GPU
    if args.mode == "multi":
        # When running with torchrun, each process should only access its assigned device
        if torch.cuda.is_available() and args.local_rank >= 0:
            # Set device based on local_rank
            device = torch.device(f"cuda:{args.local_rank}")
            torch.cuda.set_device(args.local_rank)
            logger.info(f"Process {args.local_rank} using device: {device}")
            
            # Clear GPU cache for this device only
            torch.cuda.empty_cache()
            logger.info(f"GPU {args.local_rank} memory allocated: {torch.cuda.memory_allocated(args.local_rank) / 1024**3:.2f} GB")
    else:
        # Single GPU mode - safe to query all devices
        if torch.cuda.is_available():
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"GPU name: {torch.cuda.get_device_properties(0).name}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            logger.info(f"GPU 0 memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"GPU 0 memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    
    # Setup
    model, tokenizer, tokenized_datasets = setup_training(model_name=MODEL_NAME)
    
    # Create output directories
    exp_name = "redpajama"
    run_num = '3'
    run_name = args.lr_scheduler
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Use custom directory if provided, otherwise use date-based directory
    if args.checkpoint_dir:
        base_dir = args.checkpoint_dir
        logger.info(f"🔄 USING CUSTOM DIRECTORY: {base_dir}")
    else:
        base_dir = f"./logs/{exp_name}/{MODEL_NAME}/run_{run_num}/{run_name}/{date}"
        logger.info(f"🆕 USING DATE-BASED DIRECTORY: {base_dir}")
    
    logging_dir = f"{base_dir}/tensorboard"
    output_dir = f"{base_dir}/output"
    model_dir = f"{base_dir}/model"
    config_dir = f"{base_dir}/configs"  # Directory to store config files
    
    logger.info(f"logs directory: {logging_dir}")
    
    for dir_path in [logging_dir, output_dir, model_dir, config_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Training arguments with conditional DeepSpeed setup
    training_args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 2,  # Memory-optimized batch size
        "per_device_eval_batch_size": 1,   # Smaller eval batch to prevent OOM during evaluation
        "logging_dir": logging_dir,
        "logging_steps": 10,
        "num_train_epochs": 1,
        "max_steps": 2000,
        "learning_rate": 2e-4,
        "weight_decay": 0.1,
        "gradient_accumulation_steps": 32,  # Effective batch size = 2×4×32 = 256
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "eval_strategy": "steps",  # Fixed deprecated parameter name
        "eval_steps": 500,  # Less frequent evaluation to prevent NCCL timeouts
        "warmup_steps": 100,  # 5% warmup 
        "save_steps": 400,  # More frequent saves for better checkpoint granularity
        "save_total_limit": 1,  # Keep last 2 checkpoints
        "report_to": "tensorboard",
        "lr_scheduler_type": run_name,
        # Additional stability optimizations
        "dataloader_drop_last": True,        # Ensure consistent batch sizes
        "dataloader_num_workers": 0,         # Use main process for data loading (more stable)
        "prediction_loss_only": True,        # Speed up evaluation by only computing loss
        "eval_accumulation_steps": 1,        # Process eval batches one at a time
    }
    
    # Add common parameters - disabled debug mode to save memory
    training_args_dict.update({
        "remove_unused_columns": False,
        "logging_first_step": True
    })
    
    # Add greedy LR specific parameters only when using greedy scheduler
    if run_name == "greedy":
        training_args_dict.update({
            "min_lr": 1.85e-5,
            "smooth": True,
            "factor": 0.95
        })
    
    # Handle checkpoint resumption
    checkpoint_to_resume = None
    if args.resume_from_checkpoint:
        # User specified a checkpoint path
        checkpoint_to_resume = args.resume_from_checkpoint
        logger.info(f"🔄 EXPLICIT RESUME: Using checkpoint: {checkpoint_to_resume}")
    else:
        # Auto-detect latest checkpoint
        auto_checkpoint = find_latest_checkpoint(output_dir)
        if auto_checkpoint:
            checkpoint_to_resume = auto_checkpoint
            step_num = os.path.basename(auto_checkpoint).split('-')[1]
            logger.info(f"🔄 AUTO-RESUME: Found checkpoint at step {step_num}: {checkpoint_to_resume}")
        else:
            logger.info("🆕 FRESH START: No existing checkpoint found, starting from beginning")
    
    # Add checkpoint to training arguments if found
    if checkpoint_to_resume:
        # Convert to absolute path to ensure Trainer can find it
        abs_checkpoint_path = os.path.abspath(checkpoint_to_resume)
        training_args_dict["resume_from_checkpoint"] = abs_checkpoint_path
        logger.info(f"🔄 CHECKPOINT PATH: {abs_checkpoint_path}")
    
    if use_deepspeed:
        # For multi-GPU training, add DeepSpeed config and DistributedDataParallel settings
        # Use the num_gpus argument directly in multi-GPU mode
        num_gpus = args.num_gpus if args.mode == "multi" else 1
        ds_config = get_deepspeed_config(
            num_gpus=num_gpus,
            batch_size=training_args_dict["per_device_train_batch_size"],
            grad_accum_steps=training_args_dict["gradient_accumulation_steps"],
            checkpoint_path=checkpoint_to_resume if checkpoint_to_resume else None
        )
        
        # Save a reference copy of the DeepSpeed config for future reference
        import json
        reference_ds_config_path = os.path.join(config_dir, "deepspeed_config.json")
        with open(reference_ds_config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)
            logger.info(f"DeepSpeed config saved to {reference_ds_config_path}")
            
        # Create a temporary DeepSpeed config file with unique name per process
        ds_config_path = os.path.join(base_dir, f"ds_config_rank{args.local_rank}.json")
        with open(ds_config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)
            
        training_args_dict["deepspeed"] = ds_config_path
        # Use the parsed local_rank argument instead of environment variable
        training_args_dict["local_rank"] = args.local_rank
        
        # Add DDP parameters for multi-GPU training
        training_args_dict["ddp_find_unused_parameters"] = True
        
        logger.info(f"DeepSpeed config written to {ds_config_path}")
        
    # Save a copy of the training arguments for reference
    training_args_serializable = {}
    for k, v in training_args_dict.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            training_args_serializable[k] = v
        else:
            training_args_serializable[k] = str(v)
    
    training_args_path = os.path.join(config_dir, "training_args.json")
    with open(training_args_path, 'w') as f:
        json.dump(training_args_serializable, f, indent=2)
        logger.info(f"Training arguments saved to {training_args_path}")
    
    # Save model config for reference
    model_config_path = os.path.join(config_dir, "model_config.json")
    with open(model_config_path, 'w') as f:
        # Access configuration from the model object
        model_config = model.config.to_dict()
        json.dump(model_config, f, indent=2)
        logger.info(f"Model config saved to {model_config_path}")
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create custom callbacks
    callbacks = []
    
    # Add progress bar fix callback if resuming from checkpoint
    if checkpoint_to_resume:
        tqdm_fix_callback = TqdmFixCallback()
        callbacks.append(tqdm_fix_callback)
        logger.info("🔧 Added tqdm progress bar fix callback for checkpoint resumption")
    
    # Initialize trainer - simplified for both single and multi-GPU modes
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # If checkpoint found, pre-load the trainer state directly for DeepSpeed compatibility
    if checkpoint_to_resume:
        try:
            step_num = int(os.path.basename(checkpoint_to_resume).split('-')[1])
            # Create trainer state object
            from transformers.trainer_utils import TrainerState
            
            # Directly load the trainer_state.json file
            trainer_state_path = os.path.join(checkpoint_to_resume, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                logger.info(f"🔍 Loading trainer state from: {trainer_state_path}")
                import json
                with open(trainer_state_path, 'r') as f:
                    state_dict = json.load(f)
                trainer.state = TrainerState.from_dict(state_dict)
                
                # Log the loaded state
                logger.info(f"📊 LOADED TRAINER STATE - Step: {trainer.state.global_step}/{training_args.max_steps} (Progress: {trainer.state.global_step/training_args.max_steps:.1%}), Epoch: {trainer.state.epoch:.4f}")
            else:
                logger.warning(f"⚠️ No trainer_state.json found at: {trainer_state_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error loading trainer state: {e}")
    
    if use_deepspeed:
        logger.info("Using DeepSpeed for distributed training")
    else:
        # For single GPU, explicitly move model to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if hasattr(model, "to"):
            model.to(device)
        logger.info(f"Model moved to device: {device}")
    
    # Disable cache for training
    if isinstance(model, torch.nn.DataParallel):
        model.module.config.use_cache = False
    else:
        model.config.use_cache = False
    
    # Train
    logger.info(f"Starting training with {MODEL_NAME} using {run_name} LR Scheduler...")
    
    # Resume training from checkpoint if specified
    if checkpoint_to_resume:
        abs_checkpoint_path = os.path.abspath(checkpoint_to_resume)
        logger.info(f"🔄 RESUMING TRAINING FROM: {abs_checkpoint_path}")
        
        # Debug: Check what files exist in the checkpoint
        logger.info("📁 Checkpoint contents:")
        try:
            checkpoint_files = os.listdir(abs_checkpoint_path)
            for f in sorted(checkpoint_files):
                file_path = os.path.join(abs_checkpoint_path, f)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    logger.info(f"  - {f}: {size:.2f} MB")
                else:
                    logger.info(f"  - {f}/ (directory)")
        except Exception as e:
            logger.warning(f"Could not list checkpoint contents: {e}")
        
        # Verify trainer state before training
        logger.info(f"📊 TRAINER STATE BEFORE train():")
        logger.info(f"  - global_step: {trainer.state.global_step}")
        logger.info(f"  - max_steps: {trainer.state.max_steps}")
        logger.info(f"  - epoch: {trainer.state.epoch}")
        logger.info(f"  - best_metric: {trainer.state.best_metric}")
        logger.info(f"  - best_model_checkpoint: {trainer.state.best_model_checkpoint}")
        
        # Simply pass the checkpoint path to trainer.train()
        # This is the standard way to resume training
        trainer.train(resume_from_checkpoint=abs_checkpoint_path)
    else:
        logger.info("🆕 STARTING FRESH TRAINING")
        trainer.train()
    
    # Save model
    logger.info(f"Saving model to {model_dir}")
    if isinstance(model, torch.nn.DataParallel):
        trainer.model.module.save_pretrained(
            model_dir,
            safe_serialization=False
        )
    else:
        trainer.model.save_pretrained(
            model_dir,
            safe_serialization=False
        )

if __name__ == "__main__":
    """
    Run modes:
    
    1. Single GPU (default uses cosine scheduler):
       python3 pre-train-llama3.2-1b.py
       python3 pre-train-llama3.2-1b.py --mode single --lr_scheduler cosine
       
    2. Single GPU with greedy scheduler:
       python3 pre-train-llama3.2-1b.py --lr_scheduler greedy
       
    3. Multi-GPU with DeepSpeed (works with 2, 4, or 8 GPUs):
       # From within the examples/greedy-lr directory with 2 GPUs:
       torchrun --nproc_per_node=2 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
       
       # With 4 GPUs:
       torchrun --nproc_per_node=4 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
       
       # With 8 GPUs:
       torchrun --nproc_per_node=8 pre-train-llama3.2-1b.py --mode multi --lr_scheduler cosine
       
       # With greedy scheduler:
       torchrun --nproc_per_node=2 pre-train-llama3.2-1b.py --mode multi --lr_scheduler greedy
    """
    main()
