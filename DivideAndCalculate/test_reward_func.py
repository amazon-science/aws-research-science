import pytest
import re

def strict_format_reward_func(completions, kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>[\s\S]?<api>[(.?)]</api>(?:\s->.?)?[\s\S]?</think>\s<answer>[\s\S]*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def test_strict_format_reward_func():
    # Test case 1: Valid format
    valid_completion = [[{
        "content": "<think><api>(test)</api> -> result</think> <answer>This is the answer</answer>"
    }]]
    assert strict_format_reward_func(valid_completion, {}) == [0.5]

    # Test case 2: Invalid format (missing tags)
    invalid_completion = [[{
        "content": "This is just plain text without proper formatting"
    }]]
    assert strict_format_reward_func(invalid_completion, {}) == [0.0]

    # Test case 3: Multiple completions
    multiple_completions = [
        [{"content": "<think><api>(test)</api></think> <answer>Answer 1</answer>"}],
        [{"content": "Invalid format"}],
        [{"content": "<think><api>(test2)</api> -> result</think> <answer>Answer 3</answer>"}]
    ]
    assert strict_format_reward_func(multiple_completions, {}) == [0.5, 0.0, 0.5]

    # Test case 4: Empty content
    empty_completion = [[{"content": ""}]]
    assert strict_format_reward_func(empty_completion, {}) == [0.0]

    # Test case 5: Wrong tag order
    wrong_order = [[{
        "content": "<answer>First</answer> <think><api>(test)</api></think>"
    }]]
    assert strict_format_reward_func(wrong_order, {}) == [0.0]

    # Test case 6: Missing closing tags
    missing_tags = [[{
        "content": "<think><api>(test)</api></think> <answer>No closing tag"
    }]]
    assert strict_format_reward_func(missing_tags, {}) == [0.0]

def test_strict_format_reward_func_edge_cases():
    # Test case 1: Nested tags
    nested_tags = [[{
        "content": "<think><api>(<api>nested</api>)</api></think> <answer>Answer</answer>"
    }]]
    assert strict_format_reward_func(nested_tags, {}) == [0.5]

    # Test case 2: Multiple line breaks and spaces
    multiline = [[{
        "content": """<think>
        <api>(test)</api>
        -> result
        </think> 
        <answer>
        Multiline
        answer
        </answer>"""
    }]]
    assert strict_format_reward_func(multiline, {}) == [0.5]

def test_strict_format_reward_func_exceptions():
    # Test case 1: Invalid input structure
    with pytest.raises(IndexError):
        strict_format_reward_func([[]], {})
    
    with pytest.raises(KeyError):
        strict_format_reward_func([[{"wrong_key": "content"}]], {})

if __name__ == "__main__":
    pytest.main(["-v"])
