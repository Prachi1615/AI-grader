from langchain.llms import OpenAI
from langchain_experimental.utilities import PythonREPL
import json

llm = llm

def evaluate(answer: str, evaluation_data: dict, ground_truth_answer: str):
        
    question_id = evaluation_data['Question_ID']
    question = evaluation_data['Question']
    question_type = evaluation_data['Question type']
    answer_categories = evaluation_data['Answer_Categories']

    if question_type.lower() == 'programming':
        repl = PythonREPL()
        code_execution_result = repl.run(answer)
        prompt = f"""
        You are a computer science professor evaluating a programming answer. 
        The question is: {question}
        The ground truth answer is: {ground_truth_answer}
        The answer provided by the student is: {answer}
        The result of the code execution is: {code_execution_result}

        Based on the following categories and their rubrics:
        {json.dumps(answer_categories, indent=2)}

        Evaluate the student's answer and return a markdown code snippet with a JSON object formatted to look like:
        ```json
        {{
          "Question_ID": "{question_id}",
          "Score": float,
          "Rationale": "str"
        }}
        ```
        """
    else:
        prompt = f"""
        You are a computer science professor evaluating a general question.
        The question is: {question}
        The ground truth answer is: {ground_truth_answer}
        The answer provided by the student is: {answer}

        Based on the following categories and their rubrics:
        {json.dumps(answer_categories, indent=2)}

        Evaluate the student's answer and return a markdown code snippet with a JSON object formatted to look like:
        ```json
        {{
          "Question_ID": "{question_id}",
          "Score": float,
          "Rationale": "str"
        }}
        ```
        """

    response = llm(prompt)
    # Extract the JSON part from the response
    start_idx = response.find("```json") + 7
    end_idx = response.find("```", start_idx)
    json_response = response[start_idx:end_idx].strip()

    # Parse JSON response
    result = json.loads(json_response)
    return result

# Example usage
answer = "def add(a, b): return a * b"
evaluation_data = {
    "Question_ID": "Q1",
    "Question": "Write a function to add two numbers.",
    "Question type": "Programming",
    "Answer_Categories": [
        ["Excellent", "Function correctly adds two numbers and handles edge cases", 10],
        ["Good", "Function correctly adds two numbers but does not handle edge cases", 8],
        ["Fair", "Function has minor issues but works", 5],
        ["Poor", "Function does not work correctly", 2]
    ]
}
ground_truth_answer = "def add(a, b): return a + b"

result = evaluate(answer, evaluation_data, ground_truth_answer)
print(result)
