
import os
from typing import Dict, Iterable, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

class ChatService:
    def __init__(self, model_name="llama3.1"):
        self.llm = ChatOllama(model=model_name)
        self.embed_model = "all-minilm"
        self.persist_dir = "chroma_db"
        self.collection_name = "example_collection"
        self.pdf_path = "./data/nke-10k-2023.pdf"
        self.embeddings = OllamaEmbeddings(model=self.embed_model)
        self._vectorstore: Optional[Chroma] = None  # cache handle
        

    def format_history(self, messages: List[Dict[str, str]], system_prompt: str = "") -> List[BaseMessage]:
        formatted: List[BaseMessage] = []
        if system_prompt:
            formatted.append(SystemMessage(content=system_prompt))
        for m in messages:
            role = m.get("role")
            if role == "user":
                formatted.append(HumanMessage(content=m["content"]))
            elif role in ("assistant", "ai"):
                formatted.append(AIMessage(content=m["content"]))
        return formatted

    async def chat(self, messages: List[Dict[str, str]], system_prompt: str = "") -> str:
        lc_messages = self.format_history(messages, system_prompt)
        response = await self.llm.ainvoke(lc_messages)
        return response.content

    # Streaming version for SSE/WS
    def chat_stream(self, messages: List[Dict[str, str]], system_prompt: str = "") -> Iterable[str]:
        lc_messages = self.format_history(messages, system_prompt)
        for chunk in self.llm.stream(lc_messages):
            # chunk is a AIMessageChunk; concat its content increments
            if getattr(chunk, "content", None):
                yield chunk.content
    
    def load_pdf_docs(self, file_path: Optional[str] = None):
        path = file_path or self.pdf_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()          # one Document per page
        return docs

    def split_docs(self, docs):
        # Simple, reliable char-based splitter. Adjust sizes for your model context.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(docs)

    def split_docs_by_token(docs):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50, 
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs

    
    def build_index(self, file_path: Optional[str] = None):
        docs = self.load_pdf_docs(file_path)
        chunks = self.split_docs(docs)
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        self._vectorstore = vs
        return vs

    def _open_or_build(self) -> Chroma:
        # Try open an existing collection; if not found, build
        if self._vectorstore is not None:
            return self._vectorstore
        vs = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        # Edge case: empty collection on first run — build
        if (len(vs.get()["ids"]) == 0):
            vs = self.build_index(self.pdf_path)
        self._vectorstore = vs
        return vs
    
    def get_retriever(self):
        vs = self._open_or_build()
        retriever = vs.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.2,   # tune this; 0.2–0.4 often works
            },
        )
        return retriever

    # Optional: LangChain tool wrapper (rename for your corpus)
    def get_retriever_tool(self):
        from langchain.tools.retriever import create_retriever_tool
        retriever = self.get_retriever()
        return create_retriever_tool(
            retriever,
            name="retrieve_company_10k",
            description="Search and return relevant sections from the uploaded 10-K filings.",
        )
    
    def rag_answer(self, question: str) -> str:
        retriever = self.get_retriever()
        docs = retriever.invoke(question)  # returns list[Document]
        context = "\n\n".join([f"[p{d.metadata.get('page', '?')}] {d.page_content}" for d in docs])

        system = (
            "You are a helpful assistant. Use the CONTEXT to answer the QUESTION. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"CONTEXT:\n{context}\n"
        )

        messages = [{"role": "user", "content": question}]
        return self.chat(messages, system_prompt=system)

    def rag_stream(self, question: str) -> Iterable[str]:
        retriever = self.get_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join([f"[p{d.metadata.get('page', '?')}] {d.page_content}" for d in docs])

        system = (
            "You are a helpful assistant. Use the CONTEXT to answer the QUESTION. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"CONTEXT:\n{context}\n"
        )
        messages = [{"role": "user", "content": question}]
        yield from self.chat_stream(messages, system_prompt=system)

        
