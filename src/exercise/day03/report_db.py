# pdf -> text

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS

def loadPdfToDB():
    loader = PyPDFLoader("data.pdf")
    pages = loader.load_and_split()

    # text -> snippets
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)

    # embeding
    embeddings = HuggingFaceEmbeddings() # sentence-transformers/all-mpnet-base-v2

    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        ).from_loaders([loader])

    # save vector db

    index.vectorstore.save_local("faiss-nj-pdf-data")

    return index
