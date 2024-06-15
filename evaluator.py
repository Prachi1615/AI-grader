import json
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import TransformChain
from langchain_experimental.utilities import PythonREPL

# Define the prompt template for general questions
general_prompt_template = """
You are a Computer Science professor. Based on the question, ground truth answer, and answer categories, assign a score and provide reasoning.
Question: {question}
Ground truth answer: {ground_truth}
Answer categories: {answer_categories}
Using the rubrics, evaluate the answer and provide the score and reasoning.
Answer: {answer}
Score:
Reasoning:
"""

# Define the prompt template for programming questions
programming_prompt_template = """
You are a programming expert. Evaluate the provided code based on the question, ground truth code, and answer categories, assign a score and provide reasoning.
Question: {question}
Ground truth code: {ground_truth}
Answer categories: {answer_categories}
Using the rubrics, evaluate the code and provide the score and reasoning.
Code:
{code}
Score:
Reasoning:
"""

def evaluate(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    question_id = data["Question_ID"]
    question_type = data["Question_type"]
    question = data["Question"]
    answer_categories = data["Answer_Categories"]
    ground_truth = data["Ground_truth_answer"]

    # Prepare the answer categories string
    answer_categories_str = "\n".join([f"{category[0]}: {category[1]} (Score: {category[2]})" for category in answer_categories])
    
    if question_type == "Programming":
        # Initialize PythonREPL
        repl = PythonREPL()

        # Execute the code
        execution_result = repl.run(ground_truth)
        
        # Format the prompt
        prompt = programming_prompt_template.format(
            question=question,
            ground_truth=ground_truth,
            answer_categories=answer_categories_str,
            code=execution_result
        )

        # Create an LLM chain for programming evaluation
        llm_chain = LLMChain(
            prompt=PromptTemplate.from_template(programming_prompt_template),
            llm="text-davinci-003"  # replace with your preferred LLM model
        )
    else:
        # Format the prompt for general questions
        prompt = general_prompt_template.format(
            question=question,
            ground_truth=ground_truth,
            answer_categories=answer_categories_str,
            answer=ground_truth  # assuming ground truth is the answer to evaluate
        )

        # Create an LLM chain for general evaluation
        llm_chain = LLMChain(
            prompt=PromptTemplate.from_template(general_prompt_template),
            llm="text-davinci-003"  # replace with your preferred LLM model
        )

    # Generate the evaluation
    response = llm_chain.run(prompt)
    response_lines = response.split("\n")
    score_line = next(line for line in response_lines if line.startswith("Score:"))
    reasoning_line = next(line for line in response_lines if line.startswith("Reasoning:"))

    score = score_line.split("Score:")[1].strip()
    reasoning = reasoning_line.split("Reasoning:")[1].strip()

    result = {
        "Question_ID": question_id,
        "Score": score,
        "Reasoning": reasoning[:50]  # Ensure reasoning is under 50 words
    }

    return result

# Example usage
json_file_path = 'input.json'
result = evaluate(json_file_path)
print(json.dumps(result, indent=4))
