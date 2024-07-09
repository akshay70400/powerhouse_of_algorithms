import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template 
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS  ## pinecone/chroma/faiss: databases which stores vectors in pickle format
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models  import ChatOpenAI


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text, pdf_reader

def get_text_chunks(raw_text):
    text_splitter=  CharacterTextSplitter(
        separator='\n',
        chunk_size=10,
        chunk_overlap=5,
        length_function=len,
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks
    
def get_vector_store(text_chunks):
    embeddings=OpenAIEmbeddings(model='text-embedding-ada-002')
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')  
    # Instructor embeddings completes the embedding on your local machine
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)    
    return vectorstore
    
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_pdf_summary(pdf_reader):
    total_summary = ""
    for num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[num].extract_text()
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role":"system", "content":"You are a helpful search assistant"},
                {"role":"user", "content":f"Summarize this {page_text} in maximum 200 words"},
            ],
        )
        page_summary = response['choices'][0]['message']['content']
        total_summary += page_summary + '\n'
                     
    with open('pdf_summary_500.txt', 'w+') as summary_file:
        summary_file.write(total_summary)

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with multiple PDFs', page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    # to make your varibales persistent during the entire lifecycle of application
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple PDFs")
    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_user_input(user_question)
    
    st.write(user_template.replace("{{MSG}}", "Hey Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    
    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader("Upload your pdfs here and click on process", accept_multiple_files=True)
        
        if st.button('Process'):
            with st.spinner('Wait a moment..!'):
                # get the pdfs
                raw_text, pdf_reader = get_pdf_text(pdf_docs)
                # get the texh chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vector_store(text_chunks)
                st.write('Vectorstore successfully created')
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                get_pdf_summary(pdf_reader)
                with open('pdf_summary_500.txt', 'rb') as pdf_summary_file:
                    pdf_summary = pdf_summary_file.read()
                    st.subheader(f'Here is your summary:')
                    st.write(pdf_summary)

if __name__=='__main__':
    main()
