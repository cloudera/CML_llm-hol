import os
import gradio as gr
import pinecone
from typing import Any, Union, Optional
from pydantic import BaseModel
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import requests
import json
import time

from huggingface_hub import hf_hub_download


USE_PINECONE = False # Set this to avoid any Pinecone calls

EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
GEN_AI_MODEL_REPO = "TheBloke/Llama-2-13B-chat-GGUF"
GEN_AI_MODEL_FILENAME = "llama-2-13b-chat.Q5_0.gguf"

gen_ai_model_path = hf_hub_download(repo_id=GEN_AI_MODEL_REPO, filename=GEN_AI_MODEL_FILENAME)


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

## TO DO GET MODEL DEPLOYMENT
## Need to get the below prgramatically in the future iterations
MODEL_ACCESS_KEY = os.environ["CML_MODEL_KEY"]
MODEL_ENDPOINT = "https://modelservice.ml-8ac9c78c-674.se-sandb.a465-9q4k.cloudera.site/model?accessKey=" + MODEL_ACCESS_KEY


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
    # demo = gradio.Interface(fn=get_responses,
    #                     title="Enterprise Custom Knowledge Base Chatbot with Llama2",
    #                     description=,
    #                     inputs=[gradio.Radio(['Llama-2-7b'], label="Select Model", value="Llama-2-7b"), gradio.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"), gradio.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="250"), gradio.Textbox(label="Question", placeholder="Enter your question here.")],
    #                     outputs=[gradio.Textbox(label="Llama2 7B Model Response"), gradio.Textbox(label="Context Data Source(s)"), gradio.Textbox(label="Pinecone Match Score")],
    #                     allow_flagging="never",
    #                     css=app_css)

    DESC = "This AI-powered assistant uses Cloudera DataFlow (NiFi) to scrape a website's sitemap and create a knowledge base. The information it provides as a response is context driven by what is available at the scraped websites. It uses Meta's open-source Llama2 model and the sentence transformer model all-mpnet-base-v2 to evaluate context and form an accurate response from the semantic search. It is fine tuned for questions stemming from topics in its knowledge base, and as such may have limited knowledge outside of this domain. As is always the case with prompt engineering, the better your prompt, the more accurate and specific the response."
    
    
    # Create the Gradio Interface
    demo = gr.ChatInterface(
        fn=get_responses, 
        #examples=["What is Cloudera?", "What is Apache Spark?"], 
        title="Enterprise Custom Knowledge Base Chatbot with Llama2",
        description = DESC,
        additional_inputs=[gr.Radio(['Llama-2-7b'], label="Select Model", value="Llama-2-7b"), 
                           gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"),
                           gr.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="250")],
        css = app_css,
        retry_btn = None,
        undo_btn = None,
        clear_btn = None,
        autofocus = True
        )

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,   
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_READONLY_PORT')))
    print("Gradio app ready")

# Helper function for generating responses for the QA app
def get_responses(message, history, model, temperature, token_count):
    
    
    if USE_PINECONE:
        context_chunk, sources, score = get_nearest_chunk_from_pinecone_vectordb(index, question)
        response = get_llama2_response_with_context(question, context_chunk, temperature, token_count)
        return response, sources, score
    else:
        # Essentially no context, for now
        context_chunk = "Cloudea is an Open Lakehouse Company"
        sources = ""
        score = ""
        response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
        
        #response = f"System prompt: {system_prompt}\n Message: {message}."
        for i in range(len(response)):
            time.sleep(0.02)
            yield response[: i+1]
        
        #return response, sources, score
    


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
        #params = {
        #    "prompt": str(question_and_context)
        #}
        
        ## TO DO CONVERT TO USE CML MODEL
        # response = llama2_model(prompt=question_and_context, **params)

        # model_out = response['choices'][0]['text']
        # return model_out
        
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
    

