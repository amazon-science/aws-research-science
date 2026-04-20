import os
import torch
from accelerate import Accelerator
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    GPT2LMHeadModel
)
from datasets import load_from_disk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_training(model_name):
    # Set random seed
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load prepared dataset
    dataset_name = 'tokensized_dataset_train_50K_seed_42'
    load_dir = f'./logs/codeparrot-ds/{dataset_name}/'
    tokenized_datasets = load_from_disk(load_dir)
    
    logger.info(f"Dataset columns: {tokenized_datasets['train'].column_names}")
    
    # Model configuration
    config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=len(tokenizer),
        n_ctx=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for Llama
    )
    
    # Initialize model with config
    try:
        logger.info(f"Initializing model with config: {config}")
        model = AutoModelForCausalLM.from_config(config)

        # Log model size
        # param_count = sum(p.numel() for p in model.parameters())
        # logger.info(f"Model initialized with {param_count/1e6:.2f}M parameters")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise
    
    return model, tokenizer, tokenized_datasets

def main():
    # Print environment information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_properties(0).name}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        logger.info(f"GPU 0 memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"GPU 0 memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    
    # Setup
    model, tokenizer, tokenized_datasets = setup_training(model_name=MODEL_NAME)
    
    # Create output directories
    exp_name = "codeparrot-ds"
    run_num = 'test1'
    run_name = "cosine"
    date = "2025-04-20"
    
    base_dir = f"./logs/{exp_name}/{MODEL_NAME}/run{run_num}/{run_name}/{date}"
    logging_dir = f"{base_dir}/tensorboard"
    output_dir = f"{base_dir}/output"
    model_dir = f"{base_dir}/model"
    
    logger.info(f"logs directory: {logging_dir}")
    
    for dir_path in [logging_dir, output_dir, model_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir=logging_dir,
        logging_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        weight_decay=0.1,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=1_000,
        warmup_steps=1_000,
        save_steps=500,
        save_total_limit=3,
        report_to="tensorboard",
        lr_scheduler_type=run_name,
        min_lr=1.85e-5,
        smooth=True,
        factor=0.95,
        remove_unused_columns=False,
        # optim='adafactor'
)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,  
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator
    )
    
    # Disable cache for training
    if isinstance(model, torch.nn.DataParallel):
        model.module.config.use_cache = False
    else:
        model.config.use_cache = False
    
    # Train
    logger.info(f"Starting training with {MODEL_NAME} using {run_name} LR Scheduler...")
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
    main()
