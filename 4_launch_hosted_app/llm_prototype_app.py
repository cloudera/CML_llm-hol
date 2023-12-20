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
from typing import Optional
import boto3
from botocore.config import Config

from huggingface_hub import hf_hub_download


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

## TO DO GET MODEL DEPLOYMENT
## Need to get the below prgramatically in the future iterations
MODEL_ACCESS_KEY = os.environ["CML_MODEL_KEY"]
MODEL_ENDPOINT = "https://modelservice.ml-8ac9c78c-674.se-sandb.a465-9q4k.cloudera.site/model?accessKey=" + MODEL_ACCESS_KEY

if os.environ.get("AWS_DEFAULT_REGION") == "":
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

    
## Setup Bedrock client:
def get_bedrock_client(
    assumed_role: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    bedrock_client = session.client(
        service_name="bedrock-runtime",
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


boto3_bedrock = get_bedrock_client(
      region=os.environ.get("AWS_DEFAULT_REGION", None))

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

    DESC = "This AI-powered assistant showcases the flexibility of Cloudera Machine Learning to work with 3rd party LLMs and Vector Databases."
    
    # Create the Gradio Interface
    demo = gr.ChatInterface(
        fn=get_responses, 
        #examples=["What is Cloudera?", "What is Apache Spark?"], 
        title="Enterprise Custom Knowledge Base Chatbot with Llama2",
        description = DESC,
        additional_inputs=[gr.Radio(['Local Llama 2 7B', 'AWS Bedrock Claude v2.1'], label="Select Foundational Model", value="Local Llama 2 7B"), 
                           gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"),
                           gr.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="250"),
                           gr.Radio(['None', 'Pinecone', 'Local Chroma'], label="Vector Database Choices", value="None")],
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
def get_responses(message, history, model, temperature, token_count, vector_db):
    
    #print("%s %s %s %s" % (model, temperature, token_count, vector_db))
    # AWS Bedrock Claude v2.1 0.5 50 Chroma
    
    if model == "Local Llama 2 7B":
        # Essentially no context, for now <--- this is a STUB to update with Chroma call
        context_chunk = "Cloudea is an Open Lakehouse Company"
        sources = ""
        score = ""
        response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
        
        #response = f"System prompt: {system_prompt}\n Message: {message}."
        for i in range(len(response)):
            time.sleep(0.02)
            yield response[: i+1]
    
    elif model == "AWS Bedrock Claude v2.1":
        # TODO: make AWS Bedrock call
        context_chunk = "Cloudea is an Open Lakehouse Company"
        sources = ""
        score = ""
        response = get_bedrock_response_with_context(message, context_chunk, temperature, token_count)
        
        for i in range(len(response)):
            time.sleep(0.02)
            yield response
    
    
#     if USE_PINECONE:
#         context_chunk, sources, score = get_nearest_chunk_from_pinecone_vectordb(index, question)
#         response = get_llama2_response_with_context(question, context_chunk, temperature, token_count)
#         return response, sources, score
#     else:
#         # Essentially no context, for now
#         context_chunk = "Cloudea is an Open Lakehouse Company"
#         sources = ""
#         score = ""
#         response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
        
#         #response = f"System prompt: {system_prompt}\n Message: {message}."
#         for i in range(len(response)):
#             time.sleep(0.02)
#             yield response[: i+1]
        
#         #return response, sources, score
    


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


def get_bedrock_response_with_context(question, context, temperature, token_count):
    instruction_text = """Human: You are a helpful, honest, and courteous assistant. If you don't know the answer, simply state I don't know the answer to that question. Answer the following question: {{USER_TEXT}}
                    Assistant:"""
    
    input_text = question

    # Replace instruction placeholder to build a complete prompt
    full_prompt = instruction_text.replace("{{USER_TEXT}}", input_text)
    
    # Model expects a JSON object with a defined schema
    body = json.dumps({"prompt": full_prompt,
             "max_tokens_to_sample":int(token_count),
             "temperature":0.6,
             "top_k":250,
             "top_p":1.0,
             "stop_sequences":[]
              })

    # Provide a model ID and call the model with the JSON payload
    modelId = 'anthropic.claude-v2:1'
    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept='application/json', contentType='application/json')
    response_body = json.loads(response.get('body').read())
    print("Model results successfully retreived")
    
    result = response_body.get('completion')
    #print(response_body)
    
    return result

    
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
    

