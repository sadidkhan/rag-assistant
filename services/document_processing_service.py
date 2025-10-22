
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

class DocumentProcessingService:
    def __init__(self):
        self.persist_directory = "chroma_db"
        embedding_model="all-minilm"
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_directory)
    
    def load_pdf_docs(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(len(docs))
        print(docs[1])
        #docs = docs[0:3]
        return docs
    
    def split_docs(docs):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50, 
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs

    def split_docs_by_token(docs):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50, 
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs

    
    def preprocess(self, file_path):
        docs = self.load_pdf_docs(file_path)
        split_docs = self.split_docs_by_token(docs)
        document_ids = self.vectorstore.add_documents(documents=split_docs)
        # retriever = self.vectorstore.as_retriever()
        # return retriever

        
