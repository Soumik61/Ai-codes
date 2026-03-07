from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv(override=True)
app = FastAPI()
open_api_key = os.environ["OPENAI_API_KEY"] 

llm = ChatOpenAI(openai_api_key = open_api_key, model="gpt-5.2", temperature=0)
store = {}

class ChatRequest(BaseModel):
    session_id:str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.session_id not in store:
        store[request.session_id] = ChatMessageHistory()
    history = store[request.session_id]
    message = history.messages + [HumanMessage(content=request.message)]
    response = await llm.ainvoke(message)

    history.add_user_message(request.message)
    history.add_ai_message(response.content)

    return {"response": response.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
