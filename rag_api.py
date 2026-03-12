import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import shutil

load_dotenv()
app=FastAPI(title="Rag_api", version="2.0")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0,)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

class QueryRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    sources: list[str]

vectorstore = None

def create_rag_chain():
    """Modern LCEL RAG chain"""
    if vectorstore is None:
        raise ValueError("no vectorstore loaded!")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template("""
            Answer the question based only on the following context:
            Context: {context}
            Question: {question}
            Answer:
            """)
    chain = ( 
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        |StrOutputParser()
    )

    return chain

@app.post("/upload_documents", response_model= dict)
async def upload_documents(file: UploadFile = File(...)):
    global vectorstore

    if not file.filename.endswith(('.txt')):
        raise HTTPException(400, "Only txt file supported for mow")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        shutil.copyfileobj(file.file,tmp)
        tmp_path = tmp.name
    try:
        with open(tmp_path, 'r', encoding='utf-8') as f:
            text = f.read()

        doc = Document(page_content=text, metadata={"source": file.filename})
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks=splitter.split_documents([doc])

        vectorstore = Chroma.from_documents(
            chunks, embeddings,persist_directory="./chroma_db"
        )
        return {
            "status": "success",
            "chunks": len(chunks),
            "source": file.filename,
            "db_path": "./chroma_db"
        }
    finally:
        os.unlink(tmp_path)

app.post("/rag-ask", response_model=RAGResponse)
async def rag_ask(request: QueryRequest):
    chain = create_rag_chain()
    answer = chain.invoke(request.question)

    retriver = vectorstore.as_retriever(search_kwargs={"k": 1})
    sources = retriver.invoke(request.question)
    source_names = [doc.metadata.get("source", "unknown") for doc in sources]

    return RAGResponse(
        answer=answer,
        sources=list(set(source_names))
    )

app.post("/health")
def health():
    return{
        "status": "RAG API v2 ready!",
        "vectorstore": vectorstore is not None,
        "chroma_db": os.path.exists("./chroma_db")
    }

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8081)


