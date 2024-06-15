import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
import io
import os
import json
# client = OpenAI()



import numpy as np
from utilities import question_generation,answer_generation,student_answer_generation
# MongoDB connection
mongo_client = pymongo.MongoClient("",
                                   tls=True,
    tlsAllowInvalidCertificates=True,
    serverSelectionTimeoutMS=5000)
db = mongo_client["knowledge_base"]
collection = db["kb"]
model = SentenceTransformer('all-MiniLM-L6-v2')

question_file=st.file_uploader("Upload the questions")
knowledge_base_file=st.file_uploader("Upload the Knowledge Source")
rubrik_file=st.file_uploader("Upload the rubric")

solution_file=st.file_uploader("Upload your assignment solution")



if question_file is not None:
    pdf_reader = PdfReader(question_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    que_gen=question_generation(text)
    que_str_o=str(que_gen['all_questions'])

if rubrik_file is not None:
    pdf_reader = PdfReader(rubrik_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()


if solution_file is not None:
    pdf_reader = PdfReader(solution_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    # student_answer_generation(que_str_o,text)
    

if knowledge_base_file is not None:
    

    pdf_reader = PdfReader(knowledge_base_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    # kb_output=answer_generation(que_str_o,text)

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    
    for chunk in chunks:
        # Vectorize each chunk
        vector = model.encode(chunk)
        
        
        # Ensure vector is in the form of a 2-element list for geospatial indexing
        vector_geo = {"type": "Point", "coordinates": vector[:2].tolist()}
        
        # Create a document to insert into MongoDB
        document = {
            "filename": knowledge_base_file.name,
            "chunk": chunk,
            "vector": vector_geo
        }
        
        # Insert the document into MongoDB
        collection.insert_one(document)
    
    st.success("PDF content has been chunked, vectorized, and stored in MongoDB Atlas.")
# questions_json = '''
#     [
#         {"question": "What is the model used?"},
#         {"question": "How do I create a Python function?"}
#     ]
#     '''

# questions = json.loads(questions_json)

# def generate_question_vectors(questions, model):
#     question_vectors = {}
#     for question in questions:
#         tokens = question.lower().split()  
#         vectors = []
#         for token in tokens:
#             if token in w2v_model:
#                 vectors.append(w2v_model[token])
#         if vectors:
#             question_vector = np.mean(vectors, axis=0)  
#             question_vectors[question] = question_vector.tolist()
#     return question_vectors
# question_vectors = generate_question_vectors(questions, model)

# def find_similar_vectors(query_vector):
#     results = collection.find(
#         {"vector": {"$exists": True}},
#         {"_id": 0, "vector": 1}
#     )
#     similar_vectors = []
#     for result in results:
#         vector = result['vector']
#         similarity = cosine_similarity([query_vector], [vector])[0][0]
#         similar_vectors.append({"vector": vector, "similarity": similarity})
#     return similar_vectors
# context_for_questions = {}
# for question, vector in question_vectors.items():
#     similar_vectors = find_similar_vectors(vector)
#     context_for_questions[question] = similar_vectors

# # Print or return the context in JSON format
# print(json.dumps(context_for_questions, indent=2))

grade = "A"
st.write("Final grade ",grade)



# Sample JSON data with questions
