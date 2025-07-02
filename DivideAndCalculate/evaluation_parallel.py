import json
import re
import numpy as np
import time
from datasets import load_dataset
import matplotlib.pyplot as plt
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm

def load_gsm8k_data(split="test", limit=0):
    """Load GSM8K dataset from HuggingFace"""
    dataset = load_dataset("gsm8k", "main", split=split)
    data = []
    
    for i, item in enumerate(dataset):
        answer_match = re.search(r'####\s*(-?\d+\.?\d*)', item['answer'])
        final_answer = float(answer_match.group(1)) if answer_match else item['answer']
        
        data.append({
            "id": f"gsm8k_{i}",
            "question": item['question'],
            "answer": final_answer,
            "full_answer": item['answer']
        })
    
    return data[:limit] if limit > 0 else data

def normalize_answer(answer):
    """Extract numerical answers from text"""
    if isinstance(answer, (int, float)):
        return float(answer)
        
    number_match = re.search(r'-?\d+\.?\d*', answer)
    if number_match:
        return float(number_match.group(0))
    return answer.lower().strip()

def is_correct(predicted, expected, tolerance=1e-6):
    """Check if predicted answer matches expected"""
    if isinstance(predicted, float) and isinstance(expected, float):
        return abs(predicted - expected) <= tolerance
    return predicted == expected

def process_problem(args):
    """Process a single problem - designed for parallel execution"""
    i, problem, model_name, tokenizer, model, device_id = args
    
    # Set device for this worker
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    # Format prompt
    prompt = f"Solve this math problem step by step:\n\n{problem['question']}\n\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_tokens = len(inputs.input_ids[0])
    
    # Time the generation
    q_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=False
        )
    q_time = time.time() - q_start
    
    # Process response
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = output_text[len(prompt):]
    output_tokens = len(outputs[0]) - input_tokens
    
    # Check correctness
    pred_answer = normalize_answer(response)
    expected_answer = normalize_answer(problem['answer'])
    correct = is_correct(pred_answer, expected_answer)
    
    # Return results
    print( {
        "id": problem["id"],
        "response": response,
        "predicted": pred_answer,
        "correct": correct,
        "tokens": input_tokens + output_tokens,
        "time": q_time
    }
    )
    return {
        "id": problem["id"],
        "question": problem["question"],
        "expected": problem["answer"],
        "response": response,
        "predicted": pred_answer,
        "correct": correct,
        "tokens": input_tokens + output_tokens,
        "time": q_time
    }

def evaluate_model_parallel(model_name, test_data, result_path, vis_path=None, num_workers=None):
    """Evaluate a model on test data using parallel processing"""
    print(f"Loading model: {model_name}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), torch.cuda.device_count() if torch.cuda.is_available() else 1)
    
    num_workers = min(num_workers, len(test_data))
    
    print(f"Using {num_workers} workers for parallel evaluation")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # For parallel processing, we'll need to load the model onto each worker's device
    # This is a simple way to distribute across GPUs if available
    gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
    main_device = f"cuda:{gpu_ids[0]}" if gpu_ids[0] is not None else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=main_device
    )
    
    results = {
        "model_name": model_name,
        "accuracy": 0.0,
        "total_tokens": 0,
        "per_question": []
    }
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for concurrent processing
    # This works well for language model inference which is mostly IO-bound
    with tqdm(total=len(test_data), desc="Evaluating problems") as progress_bar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Prepare the problems for parallel processing
            problems = []
            for i, problem in enumerate(test_data):
                device_id = i % len(gpu_ids) if gpu_ids[0] is not None else None
                problems.append((i, problem, model_name, tokenizer, model, device_id))
            
            # Submit all problems to the executor
            futures = {executor.submit(process_problem, args): args for args in problems}
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results["per_question"].append(result)
                    
                    # Update progress
                    progress_bar.update(1)
                    
                    # Update running stats
                    results["total_tokens"] += result["tokens"]
                    
                except Exception as e:
                    print(f"Error processing problem: {e}")
    
    # Calculate metrics
    correct_count = sum(1 for r in results["per_question"] if r["correct"])
    results["accuracy"] = correct_count / len(test_data)
    results["avg_tokens"] = results["total_tokens"] / len(test_data)
    results["total_time"] = time.time() - start_time
    results["avg_time"] = results["total_time"] / len(test_data)
    
    # Sort results by problem ID
    results["per_question"].sort(key=lambda x: x["id"])
    
    # Save results
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualization if requested
    if vis_path:
        visualize_results(results, vis_path)
    
    return results

def visualize_results(results, output_path):
    """Create simple visualization of results"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy
    ax[0].bar(["Accuracy"], [results["accuracy"]], color="blue")
    ax[0].set_ylim(0, 1)
    ax[0].set_title(f"Accuracy: {results['accuracy']:.2%}")
    
    # Tokens
    ax[1].bar(["Avg Tokens"], [results["avg_tokens"]], color="green")
    ax[1].set_title(f"Avg Tokens: {results['avg_tokens']:.1f}")
    
    plt.suptitle(f"Model: {results['model_name']}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compare_models(result_files, output_path):
    """Compare results from multiple models"""
    results = []
    for file in result_files:
        with open(file, 'r') as f:
            results.append(json.load(f))
    
    model_names = [r["model_name"].split("/")[-1] for r in results]
    accuracies = [r["accuracy"] for r in results]
    avg_tokens = [r["avg_tokens"] for r in results]
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    ax[0].bar(model_names, accuracies, color="blue")
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Accuracy Comparison")
    
    # Token usage comparison
    ax[1].bar(model_names, avg_tokens, color="green")
    ax[1].set_title("Average Tokens Comparison")
    
    plt.suptitle("Model Comparison")
    plt.tight_layout()
    plt.savefig(output_path)

# New function to run tests with system prompt
def evaluate_with_system_prompt(model_name, test_data, system_prompt, result_path, vis_path=None, num_workers=None):
    """Evaluate a model with a system prompt"""
    # We'll modify the process_problem function to include the system prompt
    def process_problem_with_system(args):
        i, problem, model_name, tokenizer, model, device_id, system_prompt = args
        
        # Set device for this worker
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        # Format prompt with system instruction
        if system_prompt:
            prompt = problem['question'] + system_prompt
        else:
            prompt = problem['question']
        print(prompt)
        # Continue with normal processing
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = len(inputs.input_ids[0])
        
        q_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                do_sample=False
            )
        q_time = time.time() - q_start
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_text[len(prompt):]
        output_tokens = len(outputs[0]) - input_tokens
        
        pred_answer = normalize_answer(response)
        expected_answer = normalize_answer(problem['answer'])
        correct = is_correct(pred_answer, expected_answer)
        
        return {
            "id": problem["id"],
            "question": problem["question"],
            "expected": problem["answer"],
            "response": response,
            "predicted": pred_answer,
            "correct": correct,
            "tokens": input_tokens + output_tokens,
            "time": q_time
        }
    
    print(f"Loading model: {model_name}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), torch.cuda.device_count() if torch.cuda.is_available() else 1)
    
    num_workers = min(num_workers, len(test_data))
    
    print(f"Using {num_workers} workers for parallel evaluation with system prompt")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
    main_device = f"cuda:{gpu_ids[0]}" if gpu_ids[0] is not None else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=main_device
    )
    
    results = {
        "model_name": model_name,
        "system_prompt": system_prompt is not None,
        "accuracy": 0.0,
        "total_tokens": 0,
        "per_question": []
    }
    
    start_time = time.time()
    
    with tqdm(total=len(test_data), desc="Evaluating problems") as progress_bar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            problems = []
            for i, problem in enumerate(test_data):
                device_id = i % len(gpu_ids) if gpu_ids[0] is not None else None
                problems.append((i, problem, model_name, tokenizer, model, device_id, system_prompt))
            
            futures = {executor.submit(process_problem_with_system, args): args for args in problems}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results["per_question"].append(result)
                    progress_bar.update(1)
                    results["total_tokens"] += result["tokens"]
                except Exception as e:
                    print(f"Error processing problem: {e}")
    
    # Calculate metrics
    correct_count = sum(1 for r in results["per_question"] if r["correct"])
    results["accuracy"] = correct_count / len(test_data)
    results["avg_tokens"] = results["total_tokens"] / len(test_data)
    results["total_time"] = time.time() - start_time
    results["avg_time"] = results["total_time"] / len(test_data)
    
    # Sort results by problem ID
    results["per_question"].sort(key=lambda x: x["id"])
    
    # Save results
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualization if requested
    if vis_path:
        visualize_results(results, vis_path)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language models on math problems using parallel processing")
    parser.add_argument("--models", nargs="+", required=True, help="Model IDs to evaluate")
    parser.add_argument("--dataset", default="gsm8k", help="Dataset to use")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of problems (0 for all)")
    parser.add_argument("--compare", action="store_true", help="Generate comparison chart")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--system_prompt", action="store_true", help="Use system prompt for calculator")
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "gsm8k":
        test_data = load_gsm8k_data(limit=args.limit)
    else:
        with open(args.dataset, 'r') as f:
            test_data = json.load(f)
            if args.limit > 0:
                test_data = test_data[:args.limit]
    
    print(f"Loaded {len(test_data)} problems")
    
    # Define system prompt for calculator
    calculator_prompt = """
    Respond in the following format:
    <think>
    ...
    <api>[Calculator(...)] -> response </api>
    ...
    </think>
    <answer>
    ...
    </answer>
    
    Here is an example for reference
    
    Question: solve 18 + 12 x 3
    <think>
    I have to solve this expression "18 + 12 x 3". let me use the calculator to solve this expression
    
    <api> [Calculator(18 + 12 * 3)] </api> -> 54
    
    I got 54 as an answer from the Calculator. So the answer is 54
    </think>
    <answer>
    54
    <answer>


    """
    
    # Evaluate each model
    result_files = []
    for model_name in args.models:
        model_short_name = model_name.split('/')[-1]
        
        # Without system prompt
        normal_result_file = f"{model_short_name}_results.json"
        normal_vis_file = f"{model_short_name}_vis.pdf"
        
        results = evaluate_model_parallel(
            model_name, 
            test_data, 
            normal_result_file,
            normal_vis_file,
            num_workers=args.workers
        )
        
        result_files.append(normal_result_file)
        
        print(f"\nResults for {model_name} (without system prompt):")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Average tokens: {results['avg_tokens']:.1f}")
        print(f"Results saved to {normal_result_file}")
        
        # With system prompt if requested
        if args.system_prompt:
            system_result_file = f"{model_short_name}_with_system_results.json"
            system_vis_file = f"{model_short_name}_with_system_vis.pdf"
            
            system_results = evaluate_with_system_prompt(
                model_name, 
                test_data, 
                calculator_prompt,
                system_result_file,
                system_vis_file,
                num_workers=args.workers
            )
            
            result_files.append(system_result_file)
            
            print(f"\nResults for {model_name} (with system prompt):")
            print(f"Accuracy: {system_results['accuracy']:.2%}")
            print(f"Average tokens: {system_results['avg_tokens']:.1f}")
            print(f"Results saved to {system_result_file}")
    
    # Generate comparison if requested
    if args.compare and len(result_files) > 1:
        compare_file = "model_comparison.pdf"
        compare_models(result_files, compare_file)
        print(f"\nComparison chart saved to {compare_file}")