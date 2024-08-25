import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.runnables import RunnablePassthrough, Runnable
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import MessagesPlaceholder
from langchain.globals import set_debug

set_debug(True)

qdrant_host_url = os.getenv("QDRANT_HOST_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

st.set_page_config(
    page_title="The Power Company Energy Solutions - QnA Chatbot",
)


def get_response(chain, prompt):
    for chunk in chain.stream(prompt):
        yield chunk

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.title("The Power Company Energy Solutions - QnA Chatbot")

model_name = "llama-3.1-8b-instant"

llm = ChatGroq(temperature=0, model_name=model_name, streaming=True)
embedding = JinaEmbeddings(model_name='jina-embeddings-v2-base-en')
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embedding,
    url=qdrant_host_url,
    api_key=qdrant_api_key,
    collection_name="chatbot"
)
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{"role": "assistant", "content": "Hi! I'm The Power Company Assistant. How may I help you?"}]

for msg in st.session_state.chat_history:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])


contextualize_rag_prompt = "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. Chat history:"
rag_system_prompt = "Given a question, provide an answer based on the context. If the question is not in context and not related to gas and electricity, just refuse to answer. \nContext: {context}.\n\nQuestion: {question}"

if prompt := st.chat_input():
    chat_history = st.session_state.chat_history
    chat_history_formatted = [(chat["role"], chat["content"]) for chat in chat_history]

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_rag_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "<Latest user question> {input}")
        ]
    )

    contextualize_chain = contextualize_prompt | llm

    refined_prompt = contextualize_chain.invoke({"chat_history": chat_history_formatted, "input": prompt})

    print("refined_prompt", refined_prompt.content)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_messages([("system", rag_system_prompt)])
        | llm
    )

    response = rag_chain.invoke(refined_prompt.content)

    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
    st.chat_message("assistant").write(response.content)