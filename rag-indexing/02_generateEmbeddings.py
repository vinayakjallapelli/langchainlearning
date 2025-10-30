import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

os.getenv("OPENAI_API_KEY")


# load the document
loader = PyPDFLoader("./Learning_LangChain.pdf")
doc = loader.load()

# split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(doc)

# embed the document
model = OpenAIEmbeddings()
embeddings = model.embed_documents([chunk.page_content for chunk in chunks])

print(embeddings)