from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterSplitter
import pinecone
from langchain.vectorestores import Pinecone

def load_documents(directory):
    directory = './docs'
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterSplitter(
        chunck_size, chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

def get_pinecone_vectorstore():
    pinecone.init(
        api_key='',
        environmet='',
    )
    index_name = 'langchain_pinecone_db_demo'
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
