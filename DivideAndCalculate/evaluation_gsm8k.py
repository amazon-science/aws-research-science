from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import torch
from utils import generate_with_api
import re
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

def load_gsm8k_test():
    """Load the GSM8K test dataset from HuggingFace."""
    from datasets import load_dataset
    dataset = load_dataset("gsm8k", "main", split="test")
    return [{"question": item["question"], "answer": item["answer"]} for item in dataset]

def extract_answer(output_text):
    """Extract the numeric answer from the model output."""
    # Look for an answer tag if it exists
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    if answer_match:
        # Extract numeric value from the answer
        answer_text = answer_match.group(1).strip()
        numeric_match = re.search(r'\d+\.?\d*', answer_text)
        if numeric_match:
            return numeric_match.group(0)
    
    # Fallback: look for the last number in the text
    numbers = re.findall(r'\d+\.?\d*', output_text)
    if numbers:
        return numbers[-1]
    
    return None

def evaluate_model(model, tokenizer, test_data, use_system_prompt=False, num_samples=None):
    """
    Evaluate model on GSM8K test set.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        test_data: GSM8K test data
        use_system_prompt: Whether to use the system prompt
        num_samples: Number of samples to evaluate (None for all)
    
    Returns:
        DataFrame with results
    """
    system_prompt = """
    use a calculator by following this format

    <api> [Calculator( The expression to be calculated )] </api>

    Here is an example for reference

    solve 18 + 12 x 3

    <think>
    okay, I have to solve this expression "18 + 12 x 3". let me use the calculator to solve this expression

    <api> [Calculator((18+12)*3)] </api> -> 54

    I got 54 as an answer from the Calculator. So the answer is 54

    </think>
    <answer>
    54
    </answer>
    """
    
    results = []
    
    # Limit the number of samples if specified
    if num_samples is not None:
        test_data = test_data[:num_samples]
    
    for item in tqdm(test_data):
        question = item["question"]
        correct_answer = item["answer"].split("####")[1].strip()
        correct_answer = re.search(r'\d+\.?\d*', correct_answer).group(0)
        
        # Prepare prompt based on flag
        final_prompt = question
        if use_system_prompt:
            final_prompt += system_prompt
        
        # Generate text
        inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
        
        # Count input tokens
        input_tokens = len(inputs.input_ids[0])
        
        # Generate response
        output = generate_with_api(inputs.input_ids, model=model, tokenizer=tokenizer, max_new_tokens=1000)
        
        # Count output tokens
        output_tokens = len(tokenizer.encode(output)) - input_tokens
        
        # Extract answer
        predicted_answer = extract_answer(output)
        
        # Check if the answer is correct
        is_correct = False
        if predicted_answer is not None:
            is_correct = predicted_answer == correct_answer
        
        print({
            "question_id": item.get("id", len(results)),
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        })

        results.append({
            "question_id": item.get("id", len(results)),
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        })
    
    return pd.DataFrame(results)

def plot_results(results_with_prompt, results_without_prompt):
    """
    Create plots for accuracy and token usage.
    
    Args:
        results_with_prompt: DataFrame with results using system prompt
        results_without_prompt: DataFrame with results without system prompt
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy comparison
    accuracy_with = results_with_prompt["is_correct"].mean() * 100
    accuracy_without = results_without_prompt["is_correct"].mean() * 100
    
    ax1.bar(["Without System Prompt", "With System Prompt"], 
            [accuracy_without, accuracy_with],
            color=["lightblue", "lightgreen"])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("GSM8K Accuracy Comparison")
    
    # Add accuracy values on top of bars
    for i, acc in enumerate([accuracy_without, accuracy_with]):
        ax1.text(i, acc + 1, f"{acc:.2f}%", ha='center')
    
    # Plot 2: Token usage
    avg_tokens_with = results_with_prompt[["input_tokens", "output_tokens"]].mean()
    avg_tokens_without = results_without_prompt[["input_tokens", "output_tokens"]].mean()
    
    x = [0, 1]
    width = 0.35
    
    ax2.bar([p - width/2 for p in x], 
            [avg_tokens_without["input_tokens"], avg_tokens_with["input_tokens"]], 
            width, label='Input Tokens', color='lightblue')
    ax2.bar([p + width/2 for p in x], 
            [avg_tokens_without["output_tokens"], avg_tokens_with["output_tokens"]], 
            width, label='Output Tokens', color='lightcoral')
    
    ax2.set_ylabel("Average Token Count")
    ax2.set_title("Token Usage Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Without System Prompt", "With System Prompt"])
    ax2.legend()
    
    # Add token counts on top of bars
    for i, (input_tokens, output_tokens) in enumerate([
        (avg_tokens_without["input_tokens"], avg_tokens_without["output_tokens"]),
        (avg_tokens_with["input_tokens"], avg_tokens_with["output_tokens"])
    ]):
        ax2.text(i - width/2, input_tokens + 5, f"{input_tokens:.0f}", ha='center')
        ax2.text(i + width/2, output_tokens + 5, f"{output_tokens:.0f}", ha='center')
    
    plt.tight_layout()
    plt.savefig("gsm8k_evaluation_results.png")
    plt.show()

def main():
    # Model configuration
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # GitHub/HuggingFace token for authentication
    auth_token = " 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"  # Replace with your actual token
    
    # Removed invalid BitsAndBytesConfig parameters
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load model and tokenizer with authentication
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=auth_token,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        load_in_8bit=True,
        token=auth_token
    )
    
    model = accelerator.prepare(model)
    
    # Load GSM8K test data
    test_data = load_gsm8k_test()
    
    # Set to a smaller number for testing, or None for full dataset
    num_samples = 100  # Adjust as needed
    
    # Run evaluation without system prompt
    print("Evaluating without system prompt...")
    results_without_prompt = evaluate_model(model, tokenizer, test_data, 
                                           use_system_prompt=False, 
                                           num_samples=num_samples)
    
    # Run evaluation with system prompt
    print("Evaluating with system prompt...")
    results_with_prompt = evaluate_model(model, tokenizer, test_data, 
                                        use_system_prompt=True, 
                                        num_samples=num_samples)
    
    # Save results to CSV
    results_without_prompt.to_csv("gsm8k_results_without_prompt.csv", index=False)
    results_with_prompt.to_csv("gsm8k_results_with_prompt.csv", index=False)
    
    # Plot and save results
    plot_results(results_with_prompt, results_without_prompt)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Accuracy without system prompt: {results_without_prompt['is_correct'].mean() * 100:.2f}%")
    print(f"Accuracy with system prompt: {results_with_prompt['is_correct'].mean() * 100:.2f}%")
    print(f"Avg tokens without system prompt: {results_without_prompt['total_tokens'].mean():.0f}")
    print(f"Avg tokens with system prompt: {results_with_prompt['total_tokens'].mean():.0f}")

if __name__ == "__main__":
    main()