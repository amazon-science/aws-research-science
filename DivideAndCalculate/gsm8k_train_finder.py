import boto3
import pandas as pd
import concurrent.futures
import re
import time
import json
from datasets import load_dataset
from tqdm import tqdm

# Load the GSM8K dataset from Huggingface
def load_gsm8k_dataset():
    dataset = load_dataset("gsm8k", "main", split="train")
    #load only top 10
    dataset = dataset.select(range(100))
    return dataset

# Function to extract numerical answer from the response
def extract_answer(text):
    # Method 1: Look for answer after '####' marker
    if '####' in text:
        parts = text.split('####')
        if len(parts) > 1:
            return float(parts[1].strip())
    
    # Method 2: Look for boxed answer format
    if r'\boxed{' in text:
        import re
        match = re.search(r'\\boxed\{(.*?)\}', text)
        if match:
            return float(match.group(1).strip())
    
    # Method 3: Look for "final answer is:" pattern
    if "final answer is:" in text.lower():
        import re
        match = re.search(r'final answer is:?\s*\\$?\\?boxed\{?(.*?)\}?\\$?', text.lower())
        if match:
            return float(match.group(1).strip())
    
    return None


# Function to call Bedrock API for a single problem
def process_problem(problem):
    question = problem["question"]
    reference_answer = problem["answer"]
    
    # Extract the ground truth numerical answer
    ground_truth = extract_answer(reference_answer)
    
    # Construct prompt for the model
    prompt = f"""Question: {question}
Please solve this step-by-step and provide the final numerical answer.
"""

    try:
        # Initialize Bedrock client
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-west-2"  # Update with your desired region
        )
        
        # Set model parameters for Meta model (adjust as needed)
        model_id = "us.meta.llama3-3-70b-instruct-v1:0"  # Update with correct model ID
        
        # Prepare request body
        request_body = {
            "prompt": prompt,
            "max_gen_len": 2048,
            "temperature": 0.2,
            "top_p": 0.9
        }
        
        # Call Bedrock API
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # Parse the response
        response_body = json.loads(response.get("body").read())
        model_response = response_body.get("generation", "")
        
        # Extract the numerical answer from the model's response
        model_answer = extract_answer(model_response)
        
        # Check if the answer is correct
        is_correct = False
        if model_answer is not None and ground_truth is not None:
            is_correct = abs(model_answer - ground_truth) < 1e-6
        
        return {
            "question": question,
            "reference_answer": reference_answer,
            "model_response": model_response,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "is_correct": is_correct
        }
        
    except Exception as e:
        return {
            "question": question,
            "reference_answer": reference_answer,
            "model_response": str(e),
            "ground_truth": ground_truth,
            "model_answer": None,
            "is_correct": False,
            "error": str(e)
        }

# Main function to process the dataset
def main():
    print("Loading GSM8K dataset...")
    dataset = load_gsm8k_dataset()
    
    # You can limit the number of examples to process for testing
    # dataset = dataset.select(range(100))  # Uncomment to process only first 100 examples
    
    results = []
    
    print(f"Processing {len(dataset)} problems in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all problems to the executor
        future_to_problem = {executor.submit(process_problem, problem): i for i, problem in enumerate(dataset)}
        
        # Process the results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_problem), total=len(future_to_problem)):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Problem generated an exception: {exc}")
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Calculate accuracy
    accuracy = df["is_correct"].mean()
    print(f"Overall accuracy: {accuracy:.2%}")
    
    # Save results to CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"debug-stage/gsm8k_bedrock_meta_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()