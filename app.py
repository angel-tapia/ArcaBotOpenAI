import streamlit as st
import os
import urllib
import requests
import random
from collections import OrderedDict
from IPython.display import display, HTML
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings

from prompts import COMBINE_QUESTION_PROMPT, COMBINE_PROMPT
from utils import model_tokens_limit, num_tokens_from_docs

# Demo Datasource Blob Storage. Change if using your own data
DATASOURCE_SAS_TOKEN = "?sv=2022-11-02&ss=bfqt&srt=c&sp=rwdlacupiytfx&se=2023-07-11T00:26:26Z&st=2023-05-15T16:26:26Z&spr=https&sig=Ji%2BfdvZT3%2BrVZyOPDQQPCbi9ea8Hl2tI7%2FPFZBPwXU4%3D"

# Don't mess with these unless you really know what you are doing
AZURE_SEARCH_API_VERSION = '2021-04-30-Preview'
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"

# Change these below with your own services credentials

AZURE_SEARCH_ENDPOINT = "https://serv-search-arca.search.windows.net"
AZURE_SEARCH_KEY = "Eo9mrr9KzPzVK7t8pIzekEMWh9wAZi6EgnnjOrDuSuAzSeAzHEUZ" # Make sure is the MANAGEMENT KEY no the query key
AZURE_OPENAI_ENDPOINT = "https://acopenai01.openai.azure.com/"
AZURE_OPENAI_API_KEY = "c095b67b814549a791596262dc299584"

headers = {'Content-Type': 'application/json','api-key': AZURE_SEARCH_KEY}

index1_name = "cogsrch-index-files"
index2_name = "cogsrch-index-csv"
indexes = [index1_name]




def generate_response(QUESTION):
    agg_search_results = []

    for index in indexes:
        url = AZURE_SEARCH_ENDPOINT + '/indexes/'+ index + '/docs'
        url += '?api-version={}'.format(AZURE_SEARCH_API_VERSION)
        url += '&search={}'.format(QUESTION)
        url += '&select=*'
        url += '&$top=5'  # You can change this to anything you need/want
        url += '&queryLanguage=en-us'
        url += '&queryType=semantic'
        url += '&semanticConfiguration=my-semantic-config'
        url += '&$count=true'
        url += '&speller=lexicon'
        url += '&answers=extractive|count-3'
        url += '&captions=extractive|highlight-false'

        resp = requests.get(url, headers=headers)
        print(url)
        print(resp.status_code)

    search_results = resp.json()
    agg_search_results.append(search_results)

    content = dict()
    ordered_content = OrderedDict()

    for search_results in agg_search_results:
        for result in search_results['value']:
            if result['@search.rerankerScore'] > 0: # Filter results that are at least 25% of the max possible score=4
                content[result['id']]={
                                        "title": result['title'],
                                        "chunks": result['pages'],
                                        "language": result['language'], 
                                        "caption": result['@search.captions'][0]['text'],
                                        "score": result['@search.rerankerScore'],
                                        "name": result['metadata_storage_name'], 
                                        "location": result['metadata_storage_path']                  
                                    }
        
    #After results have been filtered we will Sort and add them as an Ordered list\n",
    for id in sorted(content, key= lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        url = ordered_content[id]['location'] + DATASOURCE_SAS_TOKEN
        title = str(ordered_content[id]['title']) if (ordered_content[id]['title']) else ordered_content[id]['name']
        score = str(round(ordered_content[id]['score'],2))

    os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
    os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
    os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION
    os.environ["OPENAI_API_TYPE"] = "azure"

    # Create our LLM model
    # Make sure you have the deployment named "gpt-35-turbo" for the model "gpt-35-turbo (0301)". 
    # Use "gpt-4" if you have it available.
    MODEL = "gpt-35-turbo" # options: gpt-35-turbo, gpt-4, gpt-4-32k
    llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=500)

    # Now we create a simple prompt template
    prompt = PromptTemplate(
        input_variables=["question", "language"],
        template='Answer the following question: "{question}". Give your response in {language}',
    )

    #print(prompt.format(question=QUESTION, language="French"))

    # And finnaly we create our first generic chain
    #chain_chat = LLMChain(llm=llm, prompt=prompt)
    #return chain_chat({"question": QUESTION, "language": "Spanish"})

    # Iterate over each of the results chunks and create a LangChain Document class to use further in the pipeline
    docs = []
    for key,value in ordered_content.items():
        for page in value["chunks"]:
            docs.append(Document(page_content=page, metadata={"source": value["location"]}))

    if(len(docs)>0):
        tokens_limit = model_tokens_limit(MODEL) # this is a custom function we created in app/utils.py
        num_tokens = num_tokens_from_docs(docs) # this is a custom function we created in app/utils.py
        print("Custom token limit for", MODEL, ":", tokens_limit)
        print("Combined docs tokens count:",num_tokens)
            
    else:
        print("NO RESULTS FROM AZURE SEARCH")

    if num_tokens > tokens_limit:
        # Select the Embedder model
        if len(docs) < 50:
            # OpenAI models are accurate but slower, they also only (for now) accept one text at a time (chunk_size)
            embedder = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1) 
        else:
            # Bert based models are faster (3x-10x) but not as great in accuracy as OpenAI models
            # Since this repo supports Multiple languages we need to use a multilingual model. 
            # But if English only is the requirement, use "multi-qa-MiniLM-L6-cos-v1"
            # The fastest english model is "all-MiniLM-L12-v2"
            embedder = HuggingFaceEmbeddings(model_name = 'distiluse-base-multilingual-cased-v2')
        
        print(embedder)
        
        # Create our in-memory vector database index from the chunks given by Azure Search.
        # We are using FAISS. https://ai.facebook.com/tools/faiss/
        db = FAISS.from_documents(docs, embedder)
        top_docs = db.similarity_search(QUESTION, k=4)  # Return the top 4 documents
        
        # Now we need to recalculate the tokens count of the top results from similarity vector search
        # in order to select the chain type: stuff (all chunks in one prompt) or 
        # map_reduce (multiple calls to the LLM to summarize/reduce the chunks and then combine them)
        
        num_tokens = num_tokens_from_docs(top_docs)
        print("Token count after similarity search:", num_tokens)
        chain_type = "map_reduce" if num_tokens > tokens_limit else "stuff"
        
    else:
        # if total tokens is less than our limit, we don't need to vectorize and do similarity search
        top_docs = docs
        chain_type = "stuff"
        
    print("Chain Type selected:", chain_type)
    
    if chain_type == "stuff":
        chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                        prompt=COMBINE_PROMPT)
    elif chain_type == "map_reduce":
        chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                        question_prompt=COMBINE_QUESTION_PROMPT,
                                        combine_prompt=COMBINE_PROMPT,
                                        return_intermediate_steps=True)

    response = chain({"input_documents": top_docs, "question": QUESTION, "language": "Spanish"})
    
    answer = response['output_text']

    return answer

def main():
    st.title('Conversación Chatbot ArcaContinental OpenAI')

    user_input = st.text_input('Enter your message')
    if st.button('Send'):
        st.write('Usuario:', user_input)
        response = generate_response(user_input)
        st.write('Bot:', response)

if __name__ == '__main__':
    main()