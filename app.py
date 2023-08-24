import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import requests
from pathlib import Path
from time import sleep
import torch
import os
from dotenv import load_dotenv
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
model_id = os.getenv('model_id')
hf_token = os.getenv('hf_token')
repo_id = os.getenv('repo_id')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
model_id = os.environ.get('model_id')
hf_token = os.environ.get('hf_token')
repo_id = os.environ.get('repo_id')

st.set_page_config(page_title="Your Negotiation AI Assistant")
css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

file_path = os.path.join(os.getcwd(), "GTY.pdf")

texts=""

with st.sidebar:
    st.subheader("Real word negotiation skills.")
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
            st.write("Please wait a few seconds.")
            st.write("You are just about to start your NEGOTIATION journey!")
    except Exception as e:
        st.write("Unknow error.")
        print("Unknow error.")
        st.stop()

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def get_embeddings(input_str_texts):
    response = requests.post(api_url, headers=headers, json={"inputs": input_str_texts, "options":{"wait_for_model":True}})
    return response.json()

initial_embeddings=get_embeddings(texts)

db_embeddings = torch.FloatTensor(initial_embeddings) 

question = st.text_input("Enter any question on NEGOTIATION!")

if question !="":         
    st.write("Your question: "+question)
    print("Your question: "+question)
    print()
else:
#    st.write("Please enter your question first.")
    print("Please enter your question first.")
    st.stop()

q_embedding=get_embeddings(question)
final_q_embedding = torch.FloatTensor(q_embedding)

from sentence_transformers.util import semantic_search
hits = semantic_search(final_q_embedding, db_embeddings, top_k=5)

for i in range(len(hits[0])):
    print(texts[hits[0][i]['corpus_id']])
    print()

page_contents = []
for i in range(len(hits[0])):
    page_content = texts[hits[0][i]['corpus_id']]
    page_contents.append(page_content)

print(page_contents)
print()
temp_page_contents=str(page_contents)
print()
final_page_contents = temp_page_contents.replace('\\n', '') 
print(final_page_contents)
print()
print("AI Working...Please wait a while...Cheers!")
print()

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

chain = load_qa_chain(llm=llm, chain_type="stuff")

with st.spinner("AI Working...Please wait a while to Cheers!"):
    file_path = "tempfile.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(final_page_contents)

    loader = TextLoader("tempfile.txt", encoding="utf-8")
    loaded_documents = loader.load()

    temp_ai_response=chain.run(input_documents=loaded_documents, question=question)
    final_ai_response=temp_ai_response.partition('<|end|>')[0]
    i_final_ai_response = final_ai_response.replace('\n', '')
    print("AI Response:")
    print(i_final_ai_response)
    print("Have more questions? Go ahead and continue asking your AI assistant : )")

    st.write("AI Response:")
    st.write(i_final_ai_response)
#    st.write("---")
#    st.write("Have more questions? Go ahead and continue asking your AI assistant : )")
