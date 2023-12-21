# Large Language Models with Cloudera
**Hands on Lab for Cloudera Machine Learning**

:wave: and welcome... This repository is under construction.

### Overview
The goal of this hands-on lab is to explore and interact with a real LLM application. Additionally, we gain experience in configuring components of the application for the given task and performance desired. In a real-world scenario, changing business requirements and technology advancements necessitate agility in interchanging these components. 

To begin we consider the LLM life cycle. This is a simplified view but helps us highlight some of the key components we need to consider when designing our LLM application. 

![Alt text](./assets/LLM-APP-PROCESS.png)

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

### Exploring Amazon Bedrock through CML

In this first section, we'll interact with a model (Anthropic's Claude) via Amazon's Bedrock in a jupyter notebook environment from within CML.

We begin by setting up our bedrock client. At this point you should have already set up your AWS credentials as environment variables. 

#### Defining how Claude will work and respond
In this section we'll provide instructions to the model - how we would like it to respond to our prompts. In this case we are asking it to provide a summary of input text. Try playing with these settings by changing the input text, or even in the instuctions in how you would like Claude to responde to your prompts.

#### Model Parameters
The key to generative AI is in its ability to generate fresh new content. There are multiple 'knobs' we have available to modify how random (or perhaps creative) the model response can be. Try playing with the following to see if you can see model response behaviour changing. If we consider the model's - next word prediction to a softmax out, we can modify how the model picks from this distribution: 
- Temperature - The value of 1 leaves the distribution unchanged. Higher values will flatten the distribution, while lower values increase already higher weight predictions.
- Top k - Limits the model's selection of word responses to the top k most probable
- Top p - Limits the model's selection of the word responses to the top p percent of the distribution
  


### Scrape and ingest data and populate Pinecone DB

In this section we'll see the power of CML as we run job to scrape the data we'd would like to add to our knowledge base, and then a sectond job that populates the pincone vector database. The jobs have already been created for you at setup. Lets start by defining the HTML you'd like to gather data from.

**Scraping web data**
For this exercise html links are already provided. There are 5 links to various subdomains of :
https://docs.cloudera.com/machine-learning/cloud/
They are all located in file called html_links.txt. 
Anytime you point to a new location(s) you can update this file and then rerun the scraping job.

**Loading Pinecone**
In this exercise, we'll look at a number of ways to populate our vector database. One approach is to do this through a script - pinecone_vectordb_insert.py

Although effective, a more automated way of doing this is to create a job. So that say after each time you've scraped new data, you can in turn populate the vector db. A CML job can be created a number of ways, but an automated way to to do this through an API. We'll look at this now through a jupyter notebook. For this look at the file called 'create_pinecone.ipynb'. Go through the notebook and run the cells.

You've now succesfully created a new job and run it as well. By looking at your project's job section you should see a new job created that starts with "Populate Pinecone Vector DB ... "

### Explore your data via Pinecone DB

We will now get to explore our new knowlege base. We will use the following file: pinecone_vectordb_query.ipynb. You will see in the cell titled 'Create a question for semantic search' you can define your question based on your data. 

There are two functions we'll use to help us execute the query. 

- **get_nearest_chunk_from_pinecone_vectordb** - this function takes a user question and queries the Pinecone vector database to find the most relevant knowledge base content. This starts by embedding the question. Then we look for a hit on top 5 matches based on vector similarity. Finally a file path, mapping to original content is identified along with similarity score. 
- **load_context_chunk_from_data** - this function handles the responce once the filepath (or search result) has been idenfied with earlier function.

Try interacting with your vector db. You can ask it different questions about your data.

### Deploy a CML application

So far we have interacted with our models and vector database through a jupyter notebook. Now lets see how an a user might interact with an LLM solution through a CML application. CML supports a large number of solutions for deploying applications. In this lab we'll be deploying a gradio app to interact with the model. 

There are a number of ways to deploy an application within CML. In this lab we will deploy an application using the UI. We'll also explore how to do this programatically through the CML API. 

#### Deploying your application via the UI

On your project screen click on "Applications", click on new application (upper right corner). See figure below for how to fill in the fields.
Note the path for the script that will be running the app. Lastly you do not need a GPU for this instance, as this application will not house the model but will call the model for inference.
![alt-text](./assets/app-build-through-ui.png)

#### Application script
Let's take a minute to see what's powering this application before we see the application. Open the folder '3d_launch_hosted_app.py'. The model defines endpoint url and access key varibles (lines 42 and 43) which are then passed through to the bedrock client. 

You might notice this script shares some functions with the code we used earlier to query our pinecone database. The new response function also considers which model the user selects to complete the response. This highlights the power of modularity in CML.

#### Deploying your application through the API. 
Next let's look at how an application can be deployed programatically. 
The notebook first sets up the conatainer runtime paramters for the application - the python version, GPU (if required), and editor. After this is complete the application build request is exectuted. Here we define the resources required, based on expected usage. Most importantly we define the script running the application.






### Switch Vector DB to Chroma DB

