import sys
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
import pinecone
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from sentence_transformers.util import semantic_search
import requests
from pathlib import Path
from time import sleep
import pandas as pd
import torch
import os
import random
import string
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Negotiation AI Assistant", layout="wide")
st.subheader("Your Negotiation AI Assistant")

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


st.sidebar.markdown(
    """
    <style>
    .blue-underline {
        text-decoration: bold;
        color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
        }
    </style>
    """, unsafe_allow_html=True
)

file_path = os.path.join(os.getcwd(), "GTY.pdf")

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))  
    
random_string = generate_random_string(20)    
final_ss_contents=""
texts=""
wechat_image= "WeChatCode.jpg"

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
model_id = os.getenv('model_id')
hf_token = os.getenv('hf_token')
repo_id = os.getenv('repo_id')

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def get_embeddings(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

chain = load_qa_chain(llm=llm, chain_type="stuff")

PINECONE_API_KEY = "b664743d-f2ff-4842-a168-30e8fdbdefda"
PINECONE_ENVIRONMENT = "gcp-starter"
PINECONE_INDEX_NAME = "myindex-allminilm-l6-v2-384"

pinecone.init(api_key=PINECONE_API_KEY, environment = PINECONE_ENVIRONMENT) 
#pinecone.list_indexes()

index = pinecone.Index("myindex-allminilm-l6-v2-384")

with st.sidebar:
    st.subheader("Real world negotiation skills.")    
    try:        
        with st.spinner("Preparing materials for you..."):
            doc_reader = PdfReader(file_path)
            raw_text = ''
            for i, page in enumerate(doc_reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            text_splitter = CharacterTextSplitter(        
                separator = "\n",
                chunk_size = 1000,
                chunk_overlap  = 200, #striding over the text
                length_function = len,
            )
            temp_texts = text_splitter.split_text(raw_text)
            texts = temp_texts
            st.write("Materials ready.")  
            st.write("Wait a while for the AI Assistant to be ready to Chat.")               
            st.write("Disclaimer: This app is for information purpose only. NO liability could be claimed against whoever associated with this app in any manner.")    
            st.subheader("Enjoy NEGOTIATION Chatting!")            
            st.sidebar.markdown("Contact: [binqiang.liu@foxmail.com](mailto:binqiang.liu@foxmail.com)")
            st.sidebar.markdown('WeChat: <span class="blue-underline">pat2win</span>, or scan the code below.', unsafe_allow_html=True)
            st.image(wechat_image)
            st.sidebar.markdown('<span class="blue-underline">Life Enhancing with AI.</span>', unsafe_allow_html=True)                
    except Exception as e:
        st.write("Unknow error.")
        print("Unknow error.")
        st.stop()

initial_user_query = st.text_input("Enter your question here:\n")
if initial_user_query!="":
    with st.spinner("AI Working...Please wait a while to Cheers!"):
        user_query=[initial_user_query]
        q_initial_embedding = get_embeddings(user_query)
        q_final_embeddings = pd.DataFrame(q_initial_embedding) 
        q_embedding_for_pinecone = []
        row = q_final_embeddings.loc[0] 
        q_embedding_for_pinecone = list(row.values)
#index.upsert(db_embedding_saveto_pinecone)
#index.describe_index_stats()
        ss_results=index.query(vector=q_embedding_for_pinecone, top_k=5, include_values=False)
        hits = ss_results['matches'] 
        ss_contents = []
        for i in range(len(hits)):
            temp_ss_content = texts[int(hits[i]['id'])]
            ss_contents.append(temp_ss_content)
        temp_ss_contents=str(ss_contents)
        final_ss_contents = temp_ss_contents.replace('\\n', '') 
        print("Contexts the AI Assistant extracts from the original materials which have already embedded and saved in Pinecone:\n"+final_ss_contents)
#        file_path = "tempfile.txt"
#        with open(file_path, "w", encoding="utf-8") as file:
#            file.write(final_ss_contents)
#        loader = TextLoader("tempfile.txt", encoding="utf-8")
#        loaded_documents = loader.load()
        i_file_path = random_string + ".txt"
        with open(i_file_path, "w", encoding="utf-8") as file:
            file.write(final_ss_contents)
        loader = TextLoader(i_file_path, encoding="utf-8")
        loaded_documents = loader.load()        
        temp_ai_response=chain.run(input_documents=loaded_documents, question=initial_user_query)
        final_ai_response=temp_ai_response.partition('<|end|>')[0]
        i_final_ai_response = final_ai_response.replace('\n', '')
        st.write("AI Response:")
        st.write(i_final_ai_response)
        print("AI Response:")
        print(i_final_ai_response)
        print("Have more questions? Go ahead and continue asking your AI assistant :)")
