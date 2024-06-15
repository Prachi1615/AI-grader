from openai import OpenAI
import os
import json
client = OpenAI()



def question_generation(pdf_content):
    tools = [
      {
        "type": "function",
        "function": {
          "name": "Question_generation",
          "description": "Extract the questions in json",
          "parameters": {
            "type": "object",
            "properties": {
                "all_questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {
                                "type": "integer",
                                "description": "This is a descriptive question generated to solve the problem"
                            },
                            "question": {
                                "type": "string",
                                "description": "This is the reason why the question is asked and how it solves the problem stated above"
                            },
                        },
                        "required": ["question_id", "question"]
                    }
                },
                
            },
            "required": ["all_questions"]
          },
        }
      }
    ]
    messages = [{"role": "user", "content": f"""Extract the question from the data provided. Make sure you use the context to generate the json
        context: {pdf_content}"""}]
    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=messages,
      tools=tools,
      tool_choice="auto"
    )
    return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


def answer_generation(pdf_content, slides_content):
    tools = [
      {
        "type": "function",
        "function": {
          "name": "Question_generation",
          "description": "Extract the questions in json",
          "parameters": {
            "type": "object",
            "properties": {
                "all_questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {
                                "type": "integer",
                                "description": "This is a descriptive question generated to solve the problem"
                            },
                            "question": {
                                "type": "string",
                                "description": "This is the reason why the question is asked and how it solves the problem stated above"
                            },
                            "answers": {
                                "type": "string",
                                "description": "This provides the correct solution of the question asked from the slide content only"
                            },
                        },
                        "required": ["question_id", "question", "answers"]
                    }
                },
                
            },
            "required": ["all_questions"]
          },
        }
      }
    ]
    messages = [{"role": "user", "content": f"""Anwer the questions in the questions section using the slides content provided and make sure you use 
    the pdf output to generate outputs.
        Questions: {pdf_content}
        slides content: {slides_content}
        """}]
    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=messages,
      tools=tools,
      tool_choice="auto"
    )
    return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


def student_answer_generation(question_context, student_pdf_outputs):
    tools = [
      {
        "type": "function",
        "function": {
          "name": "Question_generation",
          "description": "Extract the questions in json",
          "parameters": {
            "type": "object",
            "properties": {
                "all_questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {
                                "type": "integer",
                                "description": "This is a descriptive question generated to solve the problem"
                            },
                            "student_answer": {
                                "type": "string",
                                "description": "This is the extracted answer from the student submission for that particular question"
                            },
                        },
                        "required": ["question_id", "student_answer"]
                    }
                },
                
            },
            "required": ["all_questions"]
          },
        }
      }
    ]
    messages = [{"role": "user", "content": f"""Extract all the answers provided by the students the answers should correspond to the questions
    provided in the questions outputs use the same questions ids
    the pdf output to generate outputs.
    Questions: {question_context}
        student output: {student_pdf_outputs}
        """}]
    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=messages,
      tools=tools,
      tool_choice="auto"
    )
    return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
