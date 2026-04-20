"""
Data Preparation Script for LLaMA 3.2 1B Pre-training

This script tokenizes and prepares various HuggingFace datasets for LLaMA 3.2 1B pre-training,
with special handling for:
1. code-search-net-tokenizer
2. RedPajama-Data-1T

Key features:
- Tokenization with context length management
- Training/validation split creation
- Keywords filtering for dataset subset selection
- Error handling with HuggingFace authentication
- Custom output directory structure

Usage examples:
- For CodeSearchNet:
  python llama3.2-1b_pretraining_data_prep.py --dataset_id "huggingface-course/code-search-net-tokenizer" \\
    --tokenizer_id "huggingface-course/code-search-net-tokenizer" --train_samples 50000 \\
    --output_name "tokensized_dataset_train_50K_seed_42"

- For RedPajama:
  python llama3.2-1b_pretraining_data_prep.py --dataset_id "togethercomputer/RedPajama-Data-1T" \\
    --tokenizer_id "meta-llama/Llama-3.2-1B" --train_samples 50000 \\
    --text_column "text" --output_name "redpajama_50K_seed_42"
"""

import os
import argparse
import logging
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from transformers import AutoTokenizer
from collections import defaultdict
import torch
import random
import time
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hf_access(model_id: str) -> bool:
    """
    Test access to HuggingFace model or tokenizer before proceeding with data prep.
    
    This function validates:
    1. Authentication status with HuggingFace
    2. Access rights to the requested model/tokenizer
    3. Proper functioning of the tokenizer
    
    Args:
        model_id: The HuggingFace model/tokenizer ID to test
        
    Returns:
        True if access is granted and tokenizer works, False otherwise
    """
    logger.info(f"Testing access to HuggingFace resource: {model_id}")
    
    try:
        # Try to get model info - this will check authentication and access rights
        api = HfApi()
        api.model_info(model_id)
        
        # Additionally test tokenizer loading to ensure it works
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        
        # Test a simple tokenization
        test_text = "Testing tokenizer access."
        tokens = tokenizer(test_text)
        token_count = len(tokens["input_ids"])
        
        logger.info(f"✓ Successfully accessed {model_id}")
        logger.info(f"✓ Tokenizer test: '{test_text}' → {token_count} tokens")
        return True
        
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            logger.error(f"❌ Authentication error: You need to log in to access {model_id}")
            logger.error("   Run: huggingface-cli login")
            logger.error("   Or use: from huggingface_hub import login; login(token='YOUR_TOKEN')")
        elif e.response.status_code == 403:
            logger.error(f"❌ Access denied: You don't have permission to access {model_id}")
            logger.error("   Make sure you've accepted the model license on the HuggingFace website")
            logger.error(f"   Visit: https://huggingface.co/{model_id}")
        else:
            logger.error(f"❌ Error accessing {model_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error accessing {model_id}: {e}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation for LLaMA 3.2 pre-training")
    parser.add_argument(
        "--dataset_id", 
        type=str,
        required=True,
        help="HuggingFace dataset identifier (e.g., 'huggingface-course/code-search-net-tokenizer')"
    )
    parser.add_argument(
        "--tokenizer_id", 
        type=str,
        default=None,
        help="HuggingFace tokenizer identifier (default: auto-selected based on dataset)"
    )
    parser.add_argument(
        "--context_length", 
        type=int,
        default=2048,
        help="Context length for tokenization (default: 2048 to match LLaMA 3.2 position embeddings)"
    )
    parser.add_argument(
        "--train_samples", 
        type=int,
        default=None,
        help="Number of samples to use for training (default: all available)"
    )
    parser.add_argument(
        "--valid_samples", 
        type=int,
        default=None,
        help="Number of samples to use for validation (default: 10%% of train_samples)"
    )
    parser.add_argument(
        "--output_name", 
        type=str,
        default=None,
        help="Name for the output directory (default: tokenized_dataset_[train_samples]_seed_[random_seed])"
    )
    parser.add_argument(
        "--train_split", 
        type=str,
        default="train",
        help="Name of the training split in the dataset (default: 'train')"
    )
    parser.add_argument(
        "--valid_split", 
        type=str,
        default="validation",
        help="Name of the validation split in the dataset (default: 'validation')"
    )
    parser.add_argument(
        "--text_column", 
        type=str,
        default=None,
        help="Column name containing the text to tokenize (auto-detected if not specified)"
    )
    parser.add_argument(
        "--random_seed", 
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--filtering_keywords",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of keywords to filter the dataset (space-separated)"
    )

    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only test model/tokenizer access without preparing the dataset"
    )

    return parser.parse_args()

def any_keyword_in_string(string: str, keywords: List[str]) -> bool:
    """Check if any keyword is present in a string."""
    if keywords is None or not keywords:
        return True
        
    for keyword in keywords:
        if keyword in string:
            return True
    return False

def filter_streaming_dataset(dataset, text_column: str, keywords: List[str] = None):
    """Filter a streaming dataset based on keywords."""
    if keywords is None or not keywords:
        return dataset
        
    filtered_dict = defaultdict(list)
    total = 0
    
    for sample in tqdm(iter(dataset), desc="Filtering dataset"):
        total += 1
        if text_column in sample and any_keyword_in_string(sample[text_column], keywords):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    
    if total > 0:
        logger.info(f"{len(filtered_dict[text_column])/total:.2%} of data retained after filtering.")
    else:
        logger.warning("No data found to filter.")
        
    return Dataset.from_dict(filtered_dict) if filtered_dict else None

def detect_text_column(dataset):
    """Attempt to detect the column containing text data."""
    if not dataset:
        return None
        
    # Common text column names
    potential_columns = ['text', 'content', 'code', 'source', 'document', 'sentence', 'data']
    
    # Get a sample to examine column contents
    if hasattr(dataset, 'take') and callable(getattr(dataset, 'take')):
        try:
            sample = dataset.take(1)
            sample = list(sample)[0] if sample else None
        except Exception as e:
            logger.warning(f"Could not take sample from dataset: {e}")
            sample = None
    else:
        # For regular datasets
        try:
            sample = dataset[0] if len(dataset) > 0 else None
        except Exception as e:
            logger.warning(f"Could not access dataset sample: {e}")
            sample = None
    
    if not sample:
        return None
        
    # First check columns likely to contain text
    for col in potential_columns:
        if col in sample:
            return col
    
    # If no standard column found, check for string columns
    for col, value in sample.items():
        if isinstance(value, str) and len(value) > 50:  # Reasonably long string
            return col
            
    # If still not found, return the first column name
    return list(sample.keys())[0] if sample else None

def get_appropriate_tokenizer(dataset_id, text_column=None):
    """
    Determine the appropriate tokenizer based on the dataset.
    
    This function uses heuristics to select the most suitable tokenizer:
    - For RedPajama datasets: LLaMA 3.2 1B tokenizer
    - For code datasets: Code-specific tokenizers
    - For general text: LLaMA 3.2 1B tokenizer
    
    Args:
        dataset_id: HuggingFace dataset identifier
        text_column: Name of the column containing text data
        
    Returns:
        str: HuggingFace tokenizer ID appropriate for the dataset
    """
    if "RedPajama" in dataset_id:
        return "meta-llama/Llama-3.2-1B"
    elif "code-search-net" in dataset_id:
        return "huggingface-course/code-search-net-tokenizer"
    else:
        # For other datasets, try to make a reasonable guess
        if text_column == "code" or "code" in dataset_id.lower():
            return "huggingface-course/code-search-net-tokenizer"
        else:
            return "meta-llama/Llama-3.2-1B"  # Default to LLaMA tokenizer

def load_and_prepare_dataset(
    dataset_id: str, 
    train_split: str, 
    valid_split: str, 
    train_samples: Optional[int] = None,
    valid_samples: Optional[int] = None,
    text_column: Optional[str] = None,
    random_seed: int = 42,
    filtering_keywords: Optional[List[str]] = None
) -> tuple:
    """Load and prepare the dataset based on its identifier."""
    
    logger.info(f"Loading dataset: {dataset_id}")
    
    # Special handling for RedPajama dataset
    if "RedPajama" in dataset_id:
        datasets_dict, detected_text_column = prepare_redpajama_dataset(
            dataset_id, train_samples, valid_samples, 
            text_column, random_seed, filtering_keywords
        )
        return datasets_dict, detected_text_column
    
    # Special handling for code-search-net dataset
    elif "code-search-net" in dataset_id:
        datasets_dict, detected_text_column = prepare_codesearchnet_dataset(
            dataset_id, train_samples, valid_samples, 
            train_split, valid_split, text_column, random_seed, filtering_keywords
        )
        return datasets_dict, detected_text_column
    
    # General case handling
    else:
        return prepare_generic_dataset(
            dataset_id, train_samples, valid_samples,
            train_split, valid_split, text_column, random_seed, filtering_keywords
        )

def prepare_generic_dataset(
    dataset_id: str, 
    train_samples: Optional[int],
    valid_samples: Optional[int],
    train_split: str, 
    valid_split: str,
    text_column: Optional[str], 
    random_seed: int,
    filtering_keywords: Optional[List[str]]
) -> tuple:
    """Handle generic dataset loading with fallback mechanisms."""
    raw_datasets = DatasetDict()
    detected_text_column = text_column
    
    try:
        # First try loading both splits
        logger.info(f"Attempting to load {train_split} and {valid_split} splits...")
        
        # Load training split
        train_dataset = load_dataset(dataset_id, split=train_split, streaming=(train_samples is not None))
        
        # Detect text column if not provided
        if detected_text_column is None:
            detected_text_column = detect_text_column(train_dataset)
            if detected_text_column:
                logger.info(f"Detected text column: '{detected_text_column}'")
            else:
                logger.warning("Could not auto-detect text column. Please specify with --text_column")
                detected_text_column = "text"  # Default fallback
        
        if filtering_keywords:
            logger.info(f"Filtering training data with keywords: {filtering_keywords}")
            if hasattr(train_dataset, 'filter') and callable(getattr(train_dataset, 'filter')):
                # For regular datasets
                train_dataset = train_dataset.filter(
                    lambda x: detected_text_column in x and any_keyword_in_string(x[detected_text_column], filtering_keywords)
                )
            else:
                # For streaming datasets
                train_dataset = filter_streaming_dataset(train_dataset, detected_text_column, filtering_keywords)
                if not train_dataset or (hasattr(train_dataset, '__len__') and len(train_dataset) == 0):
                    raise ValueError("Filtering resulted in empty training dataset")
            
        # Sample if needed
        if train_samples is not None:
            logger.info(f"Sampling {train_samples} examples from training data")
            train_dataset = train_dataset.shuffle(seed=random_seed)
            if hasattr(train_dataset, 'select') and callable(getattr(train_dataset, 'select')):
                train_dataset = train_dataset.select(range(min(train_samples, len(train_dataset))))
            else:
                # For streaming datasets
                examples = []
                for i, example in enumerate(tqdm(train_dataset, desc="Sampling training data")):
                    if i >= train_samples:
                        break
                    examples.append(example)
                if not examples:
                    raise ValueError("No training examples found")
                train_dataset = Dataset.from_dict({k: [d[k] for d in examples] for k in examples[0].keys()})
                
        raw_datasets['train'] = train_dataset
        
        # Load validation split
        valid_dataset = load_dataset(dataset_id, split=valid_split, streaming=(valid_samples is not None))
        
        if filtering_keywords:
            logger.info(f"Filtering validation data with keywords: {filtering_keywords}")
            if hasattr(valid_dataset, 'filter') and callable(getattr(valid_dataset, 'filter')):
                valid_dataset = valid_dataset.filter(
                    lambda x: detected_text_column in x and any_keyword_in_string(x[detected_text_column], filtering_keywords)
                )
            else:
                valid_dataset = filter_streaming_dataset(valid_dataset, detected_text_column, filtering_keywords)
                if not valid_dataset or (hasattr(valid_dataset, '__len__') and len(valid_dataset) == 0):
                    raise ValueError("Filtering resulted in empty validation dataset")
            
        # Sample if needed
        if valid_samples is not None:
            logger.info(f"Sampling {valid_samples} examples from validation data")
            valid_dataset = valid_dataset.shuffle(seed=random_seed)
            if hasattr(valid_dataset, 'select') and callable(getattr(valid_dataset, 'select')):
                valid_dataset = valid_dataset.select(range(min(valid_samples, len(valid_dataset))))
            else:
                # For streaming datasets
                examples = []
                for i, example in enumerate(tqdm(valid_dataset, desc="Sampling validation data")):
                    if i >= valid_samples:
                        break
                    examples.append(example)
                if not examples:
                    raise ValueError("No validation examples found")
                valid_dataset = Dataset.from_dict({k: [d[k] for d in examples] for k in examples[0].keys()})
                
        raw_datasets['valid'] = valid_dataset
        
    except Exception as e:
        logger.warning(f"Error loading dataset splits separately: {e}")
        logger.info("Attempting to load full dataset and split manually...")
        
        try:
            # Fallback: load full dataset and split it manually
            full_dataset = load_dataset(dataset_id)
            
            # Detect text column if not provided
            if detected_text_column is None:
                for split_name in full_dataset:
                    detected_text_column = detect_text_column(full_dataset[split_name])
                    if detected_text_column:
                        logger.info(f"Detected text column: '{detected_text_column}'")
                        break
                
                if detected_text_column is None:
                    logger.warning("Could not auto-detect text column. Defaulting to 'text'")
                    detected_text_column = "text"  # Default fallback
            
            # Check available splits
            if 'train' in full_dataset:
                raw_datasets = full_dataset
                
                # If only train exists, create a validation split
                if 'valid' not in raw_datasets and 'validation' not in raw_datasets:
                    logger.info("Creating validation split from training data")
                    train_val = raw_datasets['train'].train_test_split(
                        test_size=0.1, seed=random_seed
                    )
                    raw_datasets['train'] = train_val['train']
                    raw_datasets['valid'] = train_val['test']
                
                # Standardize validation split name
                if 'validation' in raw_datasets and 'valid' not in raw_datasets:
                    raw_datasets['valid'] = raw_datasets['validation']
                    
            else:
                # If no 'train' split, use the first available split and create train/valid
                logger.info(f"No 'train' split found. Creating splits from '{list(full_dataset.keys())[0]}'")
                first_split = list(full_dataset.keys())[0]
                train_val = full_dataset[first_split].train_test_split(
                    test_size=0.1, seed=random_seed
                )
                raw_datasets = DatasetDict({
                    'train': train_val['train'],
                    'valid': train_val['test']
                })
                
            # Apply filtering if needed
            if filtering_keywords:
                logger.info(f"Filtering data with keywords: {filtering_keywords}")
                for split in raw_datasets:
                    raw_datasets[split] = raw_datasets[split].filter(
                        lambda x: detected_text_column in x and any_keyword_in_string(x[detected_text_column], filtering_keywords)
                    )
                
            # Apply sampling if needed
            if train_samples is not None:
                logger.info(f"Sampling {train_samples} examples from training data")
                raw_datasets['train'] = raw_datasets['train'].shuffle(seed=random_seed).select(
                    range(min(train_samples, len(raw_datasets['train'])))
                )
                
            if valid_samples is not None:
                logger.info(f"Sampling {valid_samples} examples from validation data")
                raw_datasets['valid'] = raw_datasets['valid'].shuffle(seed=random_seed).select(
                    range(min(valid_samples, len(raw_datasets['valid'])))
                )
                
        except Exception as nested_e:
            logger.error(f"Failed to load dataset: {nested_e}")
            raise RuntimeError(f"Could not load dataset '{dataset_id}': {str(nested_e)}")
    
    logger.info(f"Dataset loaded successfully with train:{len(raw_datasets['train'])} and valid:{len(raw_datasets['valid'])} samples")
    return raw_datasets, detected_text_column

def prepare_redpajama_dataset(
    dataset_id: str, 
    train_samples: Optional[int], 
    valid_samples: Optional[int],
    text_column: Optional[str],
    random_seed: int,
    filtering_keywords: Optional[List[str]]
) -> tuple:
    """Simplified preparation for RedPajama dataset focused on speed."""
    logger.info("Preparing RedPajama dataset with simplified approach")
    
    # Default to 'text' column for RedPajama if not specified
    detected_text_column = text_column or "text"
    
    # For RedPajama, we need a reasonable default sample size
    actual_train_samples = 50000 if train_samples is None else train_samples
    actual_valid_samples = 5000 if valid_samples is None else valid_samples
    
    logger.info(f"Loading {actual_train_samples} training samples and {actual_valid_samples} validation samples from RedPajama")
    
    train_examples = []
    valid_examples = []
    
    # Maximum retries for handling rate limiting
    max_retries = 5
    retry_delay = 60  # seconds
    
    # Direct approach - load with streaming and minimal processing
    try:
        logger.info("Loading RedPajama dataset directly")
        
        # Try different configurations to find one that works quickly
        configs_to_try = ["arxiv", "github", "stackexchange", "common_crawl", "default"]
        config = None
        raw_dataset = None

        for cfg in configs_to_try:
            logger.info(f"Trying config: {cfg}")
            try:
                # Use streaming to handle the large dataset size
                raw_dataset = load_dataset(dataset_id, cfg, streaming=True)
                
                if "train" not in raw_dataset:
                    raw_dataset = {"train": raw_dataset}
                
                train_iter = iter(raw_dataset["train"])
                # Test the iterator by getting one item
                next(train_iter)
                train_iter = iter(raw_dataset["train"])  # Reset iterator
                config = cfg
                logger.info(f"Successfully loaded dataset with config: {config}")
                break
            except Exception as e:
                logger.warning(f"Config '{cfg}' failed: {e}")
                continue
        
        if raw_dataset is None:
            raise RuntimeError("Failed to load any RedPajama configuration")
        
        # Sample training examples with simple approach
        logger.info(f"Collecting {actual_train_samples} training examples (simplified approach)...")
        
        retries = 0
        processed = 0
        
        while len(train_examples) < actual_train_samples and retries < max_retries:
            try:
                for example in tqdm(train_iter, desc="Collecting training samples"):
                    processed += 1
                    
                    # Simple validation checks
                    if detected_text_column not in example:
                        continue
                        
                    if filtering_keywords and not any_keyword_in_string(example[detected_text_column], filtering_keywords):
                        continue
                    
                    # Take the example without any subset balancing
                    train_examples.append(example)
                    
                    # Break once we have enough
                    if len(train_examples) >= actual_train_samples:
                        logger.info(f"Collected {len(train_examples)} training examples after processing {processed} items")
                        break
                        
                    # Occasional progress update
                    if len(train_examples) % 1000 == 0:
                        logger.info(f"Collected {len(train_examples)} training examples so far")
                
                # If we're here without breaks, we reached the end of the iterator
                if len(train_examples) < actual_train_samples:
                    logger.warning(f"Reached end of dataset with only {len(train_examples)} examples")
                    
                break  # We completed without exceptions, exit retry loop
                
            except Exception as e:
                retries += 1
                logger.warning(f"Error during collection (attempt {retries}/{max_retries}): {e}")
                
                if retries < max_retries:
                    logger.info(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    
                    # Recreate iterator
                    try:
                        train_iter = iter(raw_dataset["train"])
                    except Exception as e2:
                        logger.warning(f"Failed to recreate iterator: {e2}")
        
        # For validation, use a portion of the training set
        logger.info("Creating validation set from training examples to avoid additional API calls")
        # Split training examples into train and validation
        if len(train_examples) <= actual_train_samples:
            # We don't have enough samples, so use a fraction of what we have
            logger.warning(f"Only collected {len(train_examples)} examples total. Splitting for validation.")
            split_ratio = 0.9  # Use 90% for training, 10% for validation
            random.seed(random_seed)
            random.shuffle(train_examples)
            split_idx = int(len(train_examples) * split_ratio)
            valid_examples = train_examples[split_idx:]
            train_examples = train_examples[:split_idx]
        else:
            # We have more than needed, so use the extras for validation
            valid_examples = train_examples[actual_train_samples:actual_train_samples + actual_valid_samples]
            train_examples = train_examples[:actual_train_samples]
        
    except Exception as e:
        logger.error(f"Error loading RedPajama dataset: {e}")
        raise RuntimeError(f"Failed to load RedPajama dataset: {e}")
    
    # Check if we got enough examples
    if len(train_examples) == 0:
        raise ValueError("Could not collect any training examples from RedPajama dataset")
    
    if len(valid_examples) == 0:
        logger.warning("Could not collect validation examples. Creating validation set from training examples.")
        # Split train examples into train and validation
        random.seed(random_seed)
        random.shuffle(train_examples)
        split_idx = int(len(train_examples) * 0.9)
        valid_examples = train_examples[split_idx:]
        train_examples = train_examples[:split_idx]
    
    logger.info(f"Final dataset: {len(train_examples)} training and {len(valid_examples)} validation examples")
    
    # Convert lists of examples to Dataset objects
    try:
        train_dataset = Dataset.from_dict({k: [d[k] for d in train_examples] for k in train_examples[0].keys()})
        valid_dataset = Dataset.from_dict({k: [d[k] for d in valid_examples] for k in valid_examples[0].keys()})
    except Exception as e:
        logger.error(f"Error creating datasets from examples: {e}")
        # Try a more robust approach
        train_dict = defaultdict(list)
        for example in train_examples:
            for k, v in example.items():
                train_dict[k].append(v)
                
        valid_dict = defaultdict(list)
        for example in valid_examples:
            for k, v in example.items():
                valid_dict[k].append(v)
                
        train_dataset = Dataset.from_dict(train_dict)
        valid_dataset = Dataset.from_dict(valid_dict)
    
    return DatasetDict({"train": train_dataset, "valid": valid_dataset}), detected_text_column


def prepare_codesearchnet_dataset(
    dataset_id: str, 
    train_samples: Optional[int], 
    valid_samples: Optional[int],
    train_split: str,
    valid_split: str,
    text_column: Optional[str],
    random_seed: int,
    filtering_keywords: Optional[List[str]]
) -> tuple:
    """Specific preparation for CodeSearchNet dataset."""
    logger.info("Preparing CodeSearchNet dataset")
    
    # Default text column for CodeSearchNet if not specified
    detected_text_column = text_column or "content"
    
    # For code-search-net-tokenizer, try to load pre-prepared splits
    try:
        # First check if we can load the codeparrot-ds datasets directly
        ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
        ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
        
        raw_datasets = DatasetDict({
            "train": ds_train,
            "valid": ds_valid,
        })
        
        logger.info("Successfully loaded pre-prepared CodeSearchNet datasets")
    except Exception as e:
        logger.info(f"Could not load pre-prepared datasets: {e}")
        logger.info("Falling back to standard dataset loading")
        
        # Use the generic loading function
        raw_datasets, detected_text_column = prepare_generic_dataset(
            dataset_id, train_samples, valid_samples,
            train_split, valid_split, detected_text_column, 
            random_seed, filtering_keywords
        )
    
    # Apply sampling if needed and not already applied
    if train_samples is not None:
        logger.info(f"Sampling {train_samples} examples from training data")
        raw_datasets['train'] = raw_datasets['train'].shuffle(seed=random_seed).select(
            range(min(train_samples, len(raw_datasets['train'])))
        )
        
    if valid_samples is not None:
        logger.info(f"Sampling {valid_samples} examples from validation data")
        raw_datasets['valid'] = raw_datasets['valid'].shuffle(seed=random_seed).select(
            range(min(valid_samples, len(raw_datasets['valid'])))
        )
    
    return raw_datasets, detected_text_column

def tokenize_dataset(
    dataset: DatasetDict, 
    tokenizer_id: str, 
    context_length: int, 
    text_column: str
) -> DatasetDict:
    """
    Tokenize the dataset using the specified tokenizer.
    
    This function:
    1. Loads the appropriate tokenizer
    2. Processes each example in the dataset
    3. Filters to keep only sequences of exactly context_length
    
    Args:
        dataset: Dictionary containing train/validation splits
        tokenizer_id: HuggingFace tokenizer identifier
        context_length: Maximum sequence length for tokenization
        text_column: Column name containing the text to tokenize
        
    Returns:
        DatasetDict: Dictionary containing tokenized train/validation splits
    """
    logger.info(f"Loading tokenizer: {tokenizer_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{tokenizer_id}': {e}")
        raise
    
    logger.info(f"Tokenizing dataset with context length: {context_length}")
    
    def tokenize_function(examples):
        """Tokenize examples and filter to keep only full-length sequences."""
        outputs = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        
        return {"input_ids": input_batch}
    
    # Process each split
    tokenized_datasets = DatasetDict()
    for split, ds in dataset.items():
        logger.info(f"Tokenizing {split} split...")
        try:
            tokenized_split = ds.map(
                tokenize_function,
                batched=True,
                remove_columns=ds.column_names,
                desc=f"Tokenizing {split} split"
            )
            tokenized_datasets[split] = tokenized_split
        except Exception as e:
            logger.error(f"Error tokenizing {split} split: {e}")
            raise
    
    logger.info(f"Tokenization complete. Result: {tokenized_datasets}")
    return tokenized_datasets

def get_dataset_name(dataset_id: str) -> str:
    """Determine the dataset name from the dataset identifier."""
    if "RedPajama" in dataset_id:
        return "redpajama"
    elif "code-search-net" in dataset_id:
        return "codesearchnet"
    else:
        # For other datasets, use a simplified version of the dataset_id
        # Remove organization/user prefix and convert to lowercase
        if "/" in dataset_id:
            dataset_name = dataset_id.split("/")[-1]
        else:
            dataset_name = dataset_id
        return dataset_name.lower().replace("-", "_")

def save_tokenized_dataset(
    tokenized_datasets: DatasetDict, 
    output_name: Optional[str], 
    train_samples: Optional[int],
    random_seed: int,
    dataset_id: str
) -> str:
    """
    Save the tokenized dataset to disk using a consistent directory structure.
    
    Args:
        tokenized_datasets: The processed datasets to save
        output_name: Custom name for the dataset variant (or None for auto-generated)
        train_samples: Number of training samples used (for naming)
        random_seed: Random seed used for reproducibility
        dataset_id: Original HuggingFace dataset identifier
        
    Returns:
        str: The dataset variant name used
    
    Directory Structure:
        ./datasets/{dataset_name}/{dataset_variant}/
    """
    if output_name is None:
        # Generate default name
        sample_str = f"{train_samples}" if train_samples is not None else "full"
        output_name = f"tokenized_dataset_{sample_str}_seed_{random_seed}"
    
    # Determine dataset name from dataset_id
    dataset_name = get_dataset_name(dataset_id)
    dataset_variant = output_name
    
    # Use the new directory structure: ./datasets/{dataset_name}/{dataset_variant}/
    save_dir = f"./datasets/{dataset_name}/{dataset_variant}/"
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Saving tokenized dataset to {save_dir}")
    logger.info(f"Dataset name: {dataset_name}, Dataset variant: {dataset_variant}")
    tokenized_datasets.save_to_disk(save_dir)
    logger.info(f"Dataset saved successfully to {save_dir}")
    
    return output_name

def main():
    """
    Main function to process and prepare datasets for LLaMA 3.2 1B pre-training.
    
    Workflow:
    1. Parse command-line arguments
    2. Set random seed for reproducibility
    3. Auto-select appropriate tokenizer if not specified
    4. Test HuggingFace access permissions
    5. Load and prepare the dataset with train/validation splits
    6. Tokenize the dataset with appropriate context length
    7. Save the prepared dataset to disk with consistent naming
    """
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    # Determine appropriate tokenizer if not specified
    if args.tokenizer_id is None:
        args.tokenizer_id = get_appropriate_tokenizer(args.dataset_id, args.text_column)
        logger.info(f"Auto-selected tokenizer: {args.tokenizer_id}")

    # Test HuggingFace access before proceeding
    if not test_hf_access(args.tokenizer_id):
        logger.error("Aborting data preparation due to access issues with the tokenizer.")
        return

    # Skip dataset preparation if only testing tokenizer access
    if args.test_only: 
        logger.info("Test successful. Exiting without preparing dataset (--test_only was specified).")
        return
    
    # Load and prepare the dataset
    logger.info(f"Preparing dataset: {args.dataset_id}")
    raw_datasets, detected_text_column = load_and_prepare_dataset(
        args.dataset_id,
        args.train_split,
        args.valid_split,
        args.train_samples,
        args.valid_samples,
        args.text_column,
        args.random_seed,
        args.filtering_keywords
    )
    
    # Update text column if auto-detected
    if args.text_column is None:
        args.text_column = detected_text_column
        logger.info(f"Using detected text column: {args.text_column}")
    
    # Tokenize the dataset
    tokenized_datasets = tokenize_dataset(
        raw_datasets,
        args.tokenizer_id,
        args.context_length,
        args.text_column
    )
    
    # Save the tokenized dataset
    dataset_name = save_tokenized_dataset(
        tokenized_datasets,
        args.output_name,
        args.train_samples,
        args.random_seed,
        args.dataset_id
    )
    
    logger.info(f"Dataset preparation complete. Dataset saved as '{dataset_name}'")
    logger.info(f"You can use this dataset with pre-train-llama3.2-1b.py by setting dataset_name to '{dataset_name}'")

if __name__ == "__main__":
    main()
