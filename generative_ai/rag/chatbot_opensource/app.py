import streamlit as st
import os
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title('🔥Welcome to PDF-bot!🔥')
    st.markdown("""
    ## About
    This web interface lets you ask any question to your PDF.
    
    This app is built using below tools:
    
    - [OpenAI](https://openai.com)
    - [LangChain](https://docs.langchain.com)
    - [Streamlit](https://streamlit.io)
    
    Just upload your pdf here and extract information from it.
    """)
    add_vertical_space(3)
    st.write("## 😜 Enjoy...!!!")
    st.markdown("""
    Made with 💕 by Data Scientist
    """)


def main():
    st.header("🎧Chat with PDF🎧")
    
    pdf = st.file_uploader('Please upload you PDF here', type='pdf')
    
    if pdf:
        store_name = pdf.name[:-4]
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text=text)
        
        embeddings = GPT4AllEmbeddings()
        vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
        
        query = st.text_input('Ask question about your pdf file: ❓')
        st.write("Here is the answer 🔊")

        if query:
            docs = vectorstore.similarity_search(query=query, k=3)

            llm = Ollama(model="llama2")
            chain = load_qa_chain(llm=llm, chain_type='stuff')

            response = chain.run(input_documents=docs, question=query)
            st.write(response)
            # st.write(f"💲💲 The summary of querying the answer is {cb} 💲💲")

if __name__=='__main__':
    main()
