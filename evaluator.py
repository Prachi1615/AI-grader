import json
import re

llm = llm

def evaluate_answer(answer: str, question_data: dict) -> dict:
    # Extract necessary details from question_data
    question_id = question_data.get("Question_ID")
    question = question_data.get("Question")
    answer_categories = question_data.get("Answer_Categories")
    ground_truth_answer = question_data.get("Ground truth answer")
    
    # Create the prompt
    prompt = f"""
    You are a Computer Science professor. You need to evaluate a student's answer based on the provided question, ground truth answer, and answer categories.

    Question: {question}
    Ground truth answer: {ground_truth_answer}
    
    Answer categories and rubrics:
    {answer_categories}
    
    Student's answer: {answer}
    
    Evaluate the student's answer based on the given rubrics and score it accordingly. Provide a score and a short reasoning (under 50 words).

    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{
        "Question_ID" : "{question_id}",
        "Score": float,
        "Rationale": "str"
    }}
    ```
    """
    
    # Get the evaluation from the LLM
    response = llm(prompt)
    
    # Extract the JSON object from the response using a regular expression
    match = re.search(r'```json(.*?)```', response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            raise ValueError("Unable to parse JSON from LLM response")
    else:
        raise ValueError("No JSON object found in LLM response")

# Example usage:
question_data = {
    "Question_ID": "1",
    "Question": "What is the time complexity of binary search?",
    "Answer_Categories": [
        ["Excellent", "Provides a clear and correct explanation of O(log n)", 10],
        ["Good", "Mentions O(log n) but lacks detailed explanation", 7],
        ["Fair", "Attempts to explain but is incorrect or incomplete", 5],
        ["Poor", "Incorrect answer or irrelevant information", 2]
    ],
    "Ground truth answer": "The time complexity of binary search is O(log n)."
}

answer = "The time complexity of binary search is O(log n). It divides the search interval in half each time."

# Evaluate the answer
result = evaluate_answer(answer, question_data)
print(result)
