import dotenv
import os
from langchain_qdrant import QdrantVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

filename = "faq_data.txt"
doc = TextLoader(file_path=filename).load()

print("FAQ data loaded")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(doc)

qdrant_host_url = os.getenv("QDRANT_HOST_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

embedding = JinaEmbeddings(model_name='jina-embeddings-v2-base-en')
print("Embeddings loaded")

qdrant = QdrantVectorStore.from_documents(
    documents=splits,
    embedding=embedding,
    url=qdrant_host_url,
    api_key=qdrant_api_key,
    collection_name="chatbot"
)

print("FAQ data uploaded to Qdrant")
