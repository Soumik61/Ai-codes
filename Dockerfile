FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
ENV OPENAI_API_KEY="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"
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8080"] 
