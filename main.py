from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, LlamaCppEmbeddings
import os
from getpass import getpass


def sentenceTransformer():
    print("Loading Sentence transformer")
    embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-V2')
    return embeddings

def openai():
    print("Loading openai model")
    OPENAI_API_KEY = getpass()
    embeddings = OpenAIEmbeddings(openai_api_key= os.environ["OPENAI_API_KEY"])
    return embeddings

def huggingface():
    print("Loading Huggingface model")
    return HuggingFaceEmbeddings()

def llamacpp():
    print("Loading Llama cpp")
    return LlamaCppEmbeddings(model_path="ggml-model-q4_0.bin")


class semantic_search:
    def __init__(self, directory, model = 'sentence_transformer') -> None:
        self.directory = directory
        self.persist_directory = 'chromadb'
        if model == 'sentence_transformer':
            self.embeddings = sentenceTransformer()
        elif model == 'openai':
            self.embeddings = openai()
        elif model == 'huggingface':
            self.embeddings = huggingface()
        elif model == "llamacpp":
            self.embeddings = llamacpp()
            
    def load_docs(self):
        
        documents = []
        for file in os.listdir(self.directory):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(self.directory, file)
                loader = PyPDFLoader(pdf_path, extract_images=True)
                documents.extend(loader.load())
                print(f"Loaded {file}")
            else:
                loader = DirectoryLoader(self.directory, show_progress=True, use_multithreading=True)
                documents.extend(loader.load())
                print(f"Loaded {file}")
        return documents
    
    def split_docs(self, chunk_size = 1000, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(self.load_docs())
        print(f"Created {len(docs)} chunks")
        return docs
    
    def store_in_db(self):
        
        docs = self.split_docs()
        db = Chroma.from_documents(docs, self.embeddings, persist_directory = self.persist_directory)
        db.persist()
        
    def search(self, query):
        db = Chroma(persist_directory = self.persist_directory, embedding_function = self.embeddings)
        matching_docs = db.similarity_search_with_score(query, k=2)
        print(matching_docs)
    
    
    
directory = 'test_data'
query = 'patching'
s = semantic_search(directory, "sentence_transformer")
s.store_in_db()
s.search(query)
