# Large Language Models with Cloudera
**Hands on Lab for Cloudera Machine Learning**

:wave: and welcome... This repository is under construction.

### Overview
The goal of this hands-on lab is to explore and interact with a real LLM application. Additionally, we gain experience in configuring components of the application for the given task and performance desired. In a real-world scenario, changing business requirements and technology advancements necessitate agility in interchanging these components. 

To begin we consider the LLM life cycle. This is a simplified view but helps us highlight some of the key components we need to consider when designing our LLM application. 

![Alt text](./images/LLM-APP-PROCESS.png)

From left to right we see the major phases. Under each phase we see some considerations that need to be made. 
After defining your use case requirements, which includes success criteria, four key decisions to be made are

1. Model Selection
2. Model Adaptation (Fine-tuning, RAG, More)
3. Vector Database selection (if using RAG)
4. Application Choice

This view considers not only the intial design of the LLM application, but one that allows for your application to adapt and evolve over time. For example, newer more performant models may be released that may benefit you application. This same might be true for new vector databases.  

Ultimately the lab aims to demonstrate the ease and flexibility in which users can build and modify end to end LLM applications.

This lab is broken up into the following 7 sections.

1. Exploring Amazon Bedrock through CML
2. Scrape and ingest data and populate Pinecone DB
3. Explore your data via Pinecone DB
4. Deploy a CML application
5. Switch Vector DB to Chroma DB
6. Langchain
7. Use a locally hosted LLama2 model.

:construction: 
A great resource in the meantime is [this version of the Hands on Lab](https://github.com/pdefusco/CML_LLM_HOL_Workshop/tree/main). 

#### LLMs with AWS Bedrock

In this first section, we'll interact with a model (Anthropic's Claude) via Amazon's Bedrock in a jupyter notebook environment from within CML.

We begin by setting up our bedrock client. At this point you should have already set up your AWS credentials as environment variables. 

In this section we'll provide instructions to the model - how we would like it to respond to our prompts. In this case we are asking it to provide a summary of input text. 


Changing the randomeness of the model's response. The key to generative AI is in it's ability to generate fresh new content. There are multiple 'knobs' we have available to modify how random (or perhaps creative) the model repsonse can be. Try playing with the following to see if you can see model reponse behaviour changing. If we consider the model's - next word prediction to a softmax out, we can modify how the model picks from this distribution: 
- Temperature - The value of 1 leaves the distribution unchanged. Higher values will flatten the distrubution, while lower values increase already higher weight predictions.
- Top k - Limits the model's selection of word responses to the top k most probable
- Top p - Limits the model's selection of the word responses to the top p percent of the distribution
  
Lastly try playing with changing the input text, or even in the instuctions - how you would like Claude to responde to your prompts.