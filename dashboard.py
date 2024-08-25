import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
import os


def get_response(chain, prompt):
    for chunk in chain.stream(prompt):
        yield chunk

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.title("PowerGrid Energy Solutions - QnA Chatbot")

model_name = "llama-3.1-8b-instant"
# system = "You are an assistant for question-answering of an electricity and gas company. Say that you don't know if the user asks unrelated questions to gas and electricity."
human = "{text}"
# prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

qdrant_host_url = os.getenv("QDRANT_HOST_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

llm = ChatGroq(temperature=0, model_name=model_name, streaming=True)
embedding = JinaEmbeddings(model_name='jina-embeddings-v2-base-en')
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embedding,
    url=qdrant_host_url,
    api_key=qdrant_api_key,
    collection_name="chatbot"
)
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "You are an assistant for question-answering of an electricity and gas company. Refuse to answer if the user asks unrelated questions to gas and electricity. Give concise answer based on the question and the context if available. The user doesn't know the context so you will be explaining."}, {"role": "assistant", "content": "Hi! I'm PowerGrid Assistant. How may I help you?"}]

    st.session_state["langchain_history"] = []

    st.session_state["st_history"] = []

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    history = st.session_state["messages"]
    history = [(d["role"], d["content"]) for d in history]

    history = ChatPromptTemplate.from_messages(history)
    # chain = history | llm
    

    prompt_template = PromptTemplate.from_template("You are an assistant for question-answering of an electricity and gas company. Refuse to answer if the user asks unrelated questions to gas and electricity. Give concise answer based on the question and the context if available. Context: {context}\n\nQuestion: {question}")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template 
        | llm
    )

    response = st.chat_message("assistant").write_stream(get_response(rag_chain, prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})

