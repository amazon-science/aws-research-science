from datasets import load_dataset, Dataset


SYSTEM_PROMPT = """

<think>
...
If calculations are needed, use the calculator API like this:
<api>[Calculator(expression)]</api> -> result
...
[Continue your reasoning until you reach the final answer]
...
</think>

<answer>
[Provide your final answer]
...
</answer>

Example 1:

<think> 
First, I need to calculate the price after the 15% discount. Discount amount = $80 × 0.15 
<api>[Calculator(80 * 0.15)]</api> -> 12 
Discounted price = $80 - $12 
<api>[Calculator(80 - 12)]</api> -> 68

Now I need to add the 8% sales tax to this discounted price. Tax amount = $68 × 0.08 
<api>[Calculator(68 * 0.08)]</api> -> 5.44 
Final price = $68 + $5.44 
<api>[Calculator(68 + 5.44)]</api> -> 73.44
So the final price is $73.44 
</think>

<answer>73.44</answer>

Example 2:
<think>
I have to solve this expression "18 + 12 x 3". let me use the calculator to solve this expression

<api> [Calculator(18 + 12 * 3)] </api> -> 54

I got 54 as an answer from the Calculator. So the answer is 54
</think>

<answer>54<answer>
"""




def load_and_combine_datasets(split="train") -> Dataset:
    """Load and combine the question and answer datasets."""
    question_dataset = load_dataset('openai/gsm8k', "main")[split]
    answer_dataset = load_dataset('Satyach/gsm8k')[split]
    
    # Ensure datasets have the same length
    assert len(question_dataset) == len(answer_dataset), "Datasets must have the same length"
    
    # Create dictionary of lists using list comprehensions
    combined_dict = {
        'prompt': [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': question}
            ] 
            for question in question_dataset['question'][:len(question_dataset)]
        ],
        'answer': [
            completion
            for completion in answer_dataset['calculator_answer'][:len(question_dataset)]
        ]
    }
    
    # Convert the dictionary of lists to a Dataset object
    return Dataset.from_dict(combined_dict)