import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import tempfile
import shutil

load_dotenv()

CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4


llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
)


vectorstore = None


@asynccontextmanager
def lifespan(app: FastAPI):
    global vectorstore

    if os.path.exists(CHROMA_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR, embedding_function=embeddings
        )
    yield


app = FastAPI(title="Rag_api", version="2.1")


class QueryRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]


class McpRequest(BaseModel):
    query: str


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR, embedding_function=embeddings
        )
    return vectorstore


def ensure_vectorstore():
    vs = get_vectorstore()
    try:
        if vs._collection.count() == 0:
            raise HTTPException(
                400, "Vectorstore is empty, please upload documents first."
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400, detail="No indexed documents found. upload a file first."
        )
    return vs


@app.post("/upload_documents", response_model=dict)
async def upload_documents(file: UploadFile = File(...)):

    if not file.filename.endswith((".txt")):
        raise HTTPException(400, "Only txt file supported for mow")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc = Document(page_content=text, metadata={"source": file.filename})
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = file.filename
            chunk.metadata["chunk_id"] = i

        vs = get_vectorstore()
        vs.add_documents(chunks)

        return {
            "status": "success",
            "chunks": len(chunks),
            "source": file.filename,
            "db_path": CHROMA_DIR,
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/rag-ask", response_model=RAGResponse)
async def rag_ask(request: QueryRequest):

    vs = ensure_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    prompt = ChatPromptTemplate.from_template("""
            Answer the question based only on the following context:
            Context: {context}
            Question: {input}
            Answer:
            """)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    result = retrieval_chain.invoke({"input": request.question})
    answer = result["answer"]
    context_docs = result["context"]
    sources = [doc.metadata.get("source", "unknown") for doc in context_docs]

    return RAGResponse(answer=answer, sources=list(set(sources)))


@app.post("/mcp/query")
async def mcp_query(request: McpRequest):
    vs = ensure_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(request.query)

    return {
        "tool": "rag_search",
        "query": request.query,
        "results": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content[:500],
            }
            for doc in docs
        ],
    }


@app.get("/health")
def health():
    db_exist = os.path.exists(CHROMA_DIR)
    loaded = ensure_vectorstore()
    doc_count = 0

    if loaded:
        try:
            doc_count = vectorstore._collection.count()
        except Exception:
            pass

    return {
        "status": "RAG API v2.1 ready!",
        "vectorstore": loaded is not None,
        "chroma_db": db_exist,
        "document_count": doc_count,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
