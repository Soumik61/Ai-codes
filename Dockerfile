FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8080"] 
