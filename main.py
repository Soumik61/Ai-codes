import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
OPENAI_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_key, model="gpt-5.1")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a culinary expert"),
        ("user", "{input}"),
    ]
)
output_parser = StrOutputParser()

#LCEL
chain1 = prompt
chain2 = prompt | llm
chain3 = prompt | llm | output_parser

# response1 = chain1.invoke({"input": "speaking of a good japanese food, name one."})
# print(response1)
# print("------------------------")
# response2 = chain2.invoke({"input": "Speaking of a good japanese food, name one."})
# print(response2)
# print("------------------------")
response3 = chain3.invoke({"input": "Speaking of a good japanese food, name one."})
print(response3)
print("------------------------")

