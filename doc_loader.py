# @title Load the pdf document {"vertical-output":true,"display-mode":"form"}
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_docs(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(len(docs))
    print(docs[1])
    #docs = docs[0:3]
    return docs

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)
    print(len(split_docs))
    print(split_docs[1])
    return split_docs



# Example usage:
file_path = "./data/nke-10k-2023.pdf"
docs = load_pdf_docs(file_path)