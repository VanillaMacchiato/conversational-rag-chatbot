from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

model_name = "llama3-8b-8192"

chat = ChatGroq(temperature=0, model_name=model_name)

system = "You are an assistant for QnA of an electricity and gas company. Refuse to answer if a user tries to ask unrelated questions."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

print(chain)

# question = "How much does the electricity cost?"

# print("Generating answer...")
# result = chain.invoke({"text": question})

# print("Question:", question)
# print("Answer:", result.content)
