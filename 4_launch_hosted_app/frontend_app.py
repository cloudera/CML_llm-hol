import os
import gradio
import pinecone
from typing import Any, Union, Optional
from pydantic import BaseModel
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import requests
import json
import cmlapi
import sys

client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
projects = client.list_projects(search_filter=json.dumps({"name": "Shared LLM Model for Hands on Lab"}))
project = projects.projects[0]
model = client.list_models(project_id=project.id)
selected_model = model.models[0]
MODEL_ACCESS_KEY = selected_model.access_key
MODEL_ENDPOINT = os.getenv("CDSW_API_URL").replace("https://", "https://modelservice.").replace("/api/v1", "/model?accessKey=") + MODEL_ACCESS_KEY

USE_PINECONE = False # Set this to avoid any Pinecone calls

EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"

if USE_PINECONE:
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    PINECONE_INDEX = os.getenv('PINECONE_INDEX')

    print("initialising Pinecone connection...")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    print("Pinecone initialised")

    print(f"Getting '{PINECONE_INDEX}' as object...")
    index = pinecone.Index(PINECONE_INDEX)
    print("Success")

    # Get latest statistics from index
    current_collection_stats = index.describe_index_stats()
    print('Total number of embeddings in Pinecone index is {}.'.format(current_collection_stats.get('total_vector_count')))

app_css = f"""
        .gradio-header {{
            color: white;
        }}
        .gradio-description {{
            color: white;
        }}

        #custom-logo {{
            text-align: center;
        }}
        .gr-interface {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        .gradio-header {{
            background-color: rgba(0, 0, 0, 0.5);
        }}
        .gradio-input-box, .gradio-output-box {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        h1 {{
            color: white; 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: large; !important;
        }}
"""

def main():
    # Configure gradio QA app 
    print("Configuring gradio app")
    demo = gradio.Interface(fn=get_responses,
                        title="Enterprise Custom Knowledge Base Chatbot with Llama2",
                        description="This AI-powered assistant uses Cloudera DataFlow (NiFi) to scrape a website's sitemap and create a knowledge base. The information it provides as a response is context driven by what is available at the scraped websites. It uses Meta's open-source Llama2 model and the sentence transformer model all-mpnet-base-v2 to evaluate context and form an accurate response from the semantic search. It is fine tuned for questions stemming from topics in its knowledge base, and as such may have limited knowledge outside of this domain. As is always the case with prompt engineering, the better your prompt, the more accurate and specific the response.",
                        inputs=[gradio.Radio(['Llama-2-7b'], label="Select Model", value="Llama-2-7b"), gradio.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"), gradio.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="250"), gradio.Textbox(label="Question", placeholder="Enter your question here.")],
                        outputs=[gradio.Textbox(label="Llama2 7B Model Response"), gradio.Textbox(label="Context Data Source(s)"), gradio.Textbox(label="Pinecone Match Score")],
                        allow_flagging="never",
                        css=app_css)

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
    print("Gradio app ready")

# Helper function for generating responses for the QA app
def get_responses(engine, temperature, token_count, question):
    if engine is "" or question is "" or engine is None or question is None:
        return "One or more fields have not been specified."
    if temperature is "" or temperature is None:
      temperature = 1
      
    if token_count is "" or token_count is None:
      token_count = 100
    
    if USE_PINECONE:
        context_chunk, sources, score = get_nearest_chunk_from_pinecone_vectordb(index, question)
        response = get_llama2_response_with_context(question, context_chunk, temperature, token_count)
        return response, sources, score
    else:
        # Essentially no context, for now
        context_chunk = "Cloudea is an Open Lakehouse Company"
        sources = ""
        score = ""
        response = get_llama2_response_with_context(question, context_chunk, temperature, token_count)
        return response, sources, score



# Get embeddings for a user question and query Pinecone vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_pinecone_vectordb(index, question):
    # Generate embedding for user question with embedding model
    retriever = SentenceTransformer(EMBEDDING_MODEL_REPO)
    xq = retriever.encode([question]).tolist()
    xc = index.query(xq, top_k=5,
                 include_metadata=True)
    
    matching_files = []
    scores = []
    for match in xc['matches']:
        # extract the 'file_path' within 'metadata'
        file_path = match['metadata']['file_path']
        # extract the individual scores for each vector
        score = match['score']
        scores.append(score)
        matching_files.append(file_path)

    # Return text of the nearest knowledge base chunk 
    # Note that this ONLY uses the first matching document for semantic search. matching_files holds the top results so you can increase this if desired.
    response = load_context_chunk_from_data(matching_files[0])
    sources = matching_files[0]
    score = scores[0]
    return response, sources, score
  
# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()

  
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llama2_response_with_context(question, context, temperature, token_count):

    question = "Answer this question based on the given context. Question: " + str(question)
    
    question_and_context = question + " Here is the context: " + str(context)

    try:
        # Following LLama's spec for prompt engineering
        llama_sys = f"<<SYS>>\n You are a helpful, respectful and honest assistant. If you are unsurae about an answer, truthfully say \"I don't know\".\n<</SYS>>\n\n"
        llama_inst = f"[INST]Use your own knowledge and additionally use the following information to answer the user's question: {context} [/INST]"
        question_and_context = f"{llama_sys} {llama_inst} [INST] User: {question} [/INST]"
        
        data={ "request": {"prompt":question_and_context,"temperature":temperature,"max_new_tokens":token_count,"repetition_penalty":1.0} }
        
        r = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers={'Content-Type': 'application/json'})
        
        # Logging
        print(f"Request: {data}")
        print(f"Response: {r.json()}")
        
        no_inst_response = str(r.json()['response'])[len(question_and_context)+2:]
            
        return no_inst_response
        
    except Exception as e:
        print(e)
        return e


if __name__ == "__main__":
    main()
    

