import re
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
emodel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")


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

# Load and prep dataset
SYSTEM_PROMPT = """# Reasoning Format Instructions
When answering questions, please structure your response in the following format:

<think>
[Show your step-by-step reasoning process here. This is where you work through the problem]
[If calculations are needed, use the calculator API like this:]
<api>[Calculator(expression)]</api> -> result
[Continue your reasoning until you reach the final answer]
</think>

<answer>
[Provide your final, concise answer here without repeating all the reasoning]
</answer>

Example:
Question: If a store offers a 15% discount on a $80 item and then charges 8% sales tax, what is the final price?

<think> First, I need to calculate the price after the 15% discount. Discount amount = \$80 × 0.15 <api>[Calculator(80 * 0.15)]</api> -> 12 Discounted price = \$80 - \$12 <api>[Calculator(80 - 12)]</api> -> 68
Now I need to add the 8% sales tax to this discounted price. Tax amount = $68 × 0.08 <api>[Calculator(68 * 0.08)]</api> -> 5.44 Final price = $68 + $5.44 <api>[Calculator(68 + 5.44)]</api> -> 73.44

So the final price is $73.44 </think>

<answer>73.44</answer>
Please follow this format for your response. The <think> section should contain your complete reasoning process and any calculations, while the <answer> section should provide only the final answer in a clear and concise manner. </answer>
"""

XML_COT_FORMAT = """\
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""

# # Function to extract numerical answer from the response
# def extract_answer(text):
#     # Make sure text is a string
#     if not isinstance(text, str):
#         if isinstance(text, dict) and 'generated_text' in text:
#             # Extract the text from the response dictionary
#             text = text['generated_text']
#         else:
#             # Try to convert to string or return None if not possible
#             try:
#                 text = str(text)
#             except:
#                 return None
    
#     # Method 1: Look for answer after '####' marker
#     if '####' in text:
#         parts = text.split('####')
#         if len(parts) > 1:
#             # Extract number from the part after ####
#             answer_text = parts[1].strip()
#             numbers = re.findall(r'-?\d+\.?\d*', answer_text)
#             if numbers:
#                 return float(numbers[0])
    
#     # Method 2: Look for boxed answer format
#     if r'\boxed{' in text:
#         match = re.search(r'\\boxed\{(.*?)\}', text)
#         if match:
#             answer_text = match.group(1).strip()
#             numbers = re.findall(r'-?\d+\.?\d*', answer_text)
#             if numbers:
#                 return float(numbers[0])
    
#     # Method 3: Look for "final answer is:" pattern
#     if "final answer is:" in text.lower():
#         match = re.search(r'final answer is:?\s*\\$?\\?boxed\{?(.*?)\}?\\$?', text.lower())
#         if match:
#             answer_text = match.group(1).strip()
#             numbers = re.findall(r'-?\d+\.?\d*', answer_text)
#             if numbers:
#                 return float(numbers[0])
    
#     # Method 4: Just try to find the last number in the text
#     numbers = re.findall(r'-?\d+\.?\d*', text)
#     if numbers:
#         return float(numbers[-1])
    
#     return None

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
# def get_gsm8k_questions(split = "train") -> Dataset:
#     data_question = load_dataset('simplescaling/s1K')[split] # type: ignore
#     data_answer = load_dataset('Satyach/grpo')[split]
#     data = data.map(lambda x: { # type: ignore
#         'prompt': [
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': x['question']}
#         ],
#         'answer': extract_hash_answer(x['answer'])
#     }) # type: ignore

#     return data # type: ignore

# from datasets import load_dataset, Dataset

# def load_and_combine_datasets(split="train") -> Dataset:
#     """Load and combine the question and answer datasets."""
#     question_dataset = load_dataset('simplescaling/s1K-claude-3-7-sonnet_tokenized')[split]
#     answer_dataset = load_dataset('Satyach/grpo')[split]
    
#     # Ensure datasets have the same length
#     assert len(question_dataset) == len(answer_dataset), "Datasets must have the same length"
    
#     # Create dictionary of lists using list comprehensions
#     combined_dict = {
#         'prompt': [
#             [
#                 {'role': 'system', 'content': SYSTEM_PROMPT},
#                 {'role': 'user', 'content': question}
#             ] 
#             for question in question_dataset['question'][:1000]
#         ],
#         'answer': [
#             completion
#             for completion in answer_dataset['completion'][:1000]
#         ]
#     }
    
#     # Convert the dictionary of lists to a Dataset object
#     return Dataset.from_dict(combined_dict)


# dataset = load_and_combine_datasets()

# def answer_similarity(prompts, completions, answer, **kwargs) -> list[float]:
#     # Extract ground truth completions
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")

#     # Calculate embeddings and similarities
#     completion_embeddings = emodel.encode(responses[0])
#     ground_truth_embeddings = emodel.encode(answer[0])
#     similarities = emodel.similarity(completion_embeddings, ground_truth_embeddings).diagonal()
#     #print(similarities)
    
#     return similarities






# Reward functions
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# def strict_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r'<think>[\s\S]*?</think>\s*<answer>\d+</answer>'
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [2.5 if match else 0.0 for match in matches]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    #pattern = r'<think>.*?</think>\s*<answer>\d+(\.\d+)?</answer>'
    pattern= r'<think>[\s\S]*?</think>\s*<answer>[0-9.]+</answer>'
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [2.5 if match else 0.0 for match in matches]


def calculate_reward_strict(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the Calculator API format."""
    # Pattern to match <api> [Calculator(...)] </api>
    api_pattern = r'<api>\s*\[Calculator\((.*?)\)\]\s*</api>'
    
    # Extract content from completions
    responses = [completion[0]["content"] for completion in completions]
    
    # Check pattern matches for each response
    matches = [re.search(api_pattern, r) is not None for r in responses]
    
    # Return 1.0 for matches, 0.0 for non-matches
    return [1.5 if match else 0.0 for match in matches]




def tool_call_format_reward(completions, **kwargs):
    """Reward function that checks if tool calls have the correct format.
    
    The format should match: [Calculator(expr)] followed by result
    """
    pattern = r'\[Calculator\(.*?\)\]'
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.finditer(pattern, r) for r in responses]
    #matches = [re.finditer(pattern, content) for content in completions]
    #matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.25
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.25
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]



# reward function to check the count of number of api calls. if the api calls are equal to the number of api calls in the answer, then give good reward. if it is lower or higher give it lower reward
def tool_usage_reward(completions, prompts, answer, **kwargs):
        """Reward function that evaluates tool usage patterns.
        
        Rewards proper tool usage and penalizes excessive tool calls compared to golden response.
        
        Args:
            completions: List of model completions
            solution: List of ground truth solutions containing tool usage
        
        Returns:
            List of rewards, one per completion
        """
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        return [2.0 if r.count("[Calculator(") == a.count("[Calculator(") else 1.0 if (r.count("[Calculator(") > a.count("[Calculator(") or r.count("[Calculator(") < a.count("[Calculator(")) and r.count("[Calculator(") != 0 else 0.0 for r, a in zip(extracted_responses, answer)]



# a reward function the verifies if the math expression in api tags is valid or not (The validation is if it is accepted by the numexpr, then it is valid else it is not)
def validate_math_expression_reward(completions, **kwargs) -> list[float]:
    """Reward function that validates if the math expression in API tags is accepted by numexpr."""
    from utils import extract_last_calculator_expression, calculate_safe
    
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        # Extract the expression from Calculator API tags
        expression = extract_last_calculator_expression(response)
        print(expression)
        
        if expression is None:
            rewards.append(0.0)
            continue
            
        # Check if the expression is valid using numexpr
        try:
            calculate_safe(expression)
            rewards.append(1.0)  # Expression is valid
        except Exception as e:
            rewards.append(0.0)  # Expression is invalid
            
    return rewards



# def numerical_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     """Reward function that compares numerical answers, allowing for different formats.
    
#     Args:
#         prompts: List of conversation histories
#         completions: List of model completions
#         answer: List of correct answers
#         **kwargs: Additional arguments
        
#     Returns:
#         List of rewards (1.0 for correct numerical answer, 0.0 otherwise)
#     """
#     #from evaluate_gsm8k import extract_answer
    
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_answer(r) for r in responses]
#     correct_answers = [extract_answer(str(a)) for a in answer]
    
#     # Compare numerical values with conversion to float for proper comparison
#     rewards = []
#     for resp, ans in zip(extracted_responses, correct_answers):
#         if resp is not None and ans is not None:
#             try:
#                 resp_num = float(resp)
#                 ans_num = float(ans)
#                 rewards.append(1.0 if resp_num == ans_num else 0.0)
#             except ValueError:
#                 rewards.append(0.0)
#         else:
#             rewards.append(0.0)
    
#     return rewards
def numerical_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # Check if the extracted response is a numerical value (integer or float)
    def is_numerical(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return [1.0 if is_numerical(r) else 0.0 for r in extracted_responses]
