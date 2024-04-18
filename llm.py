import langchain
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch 
from langchain.vectorstores import Chroma as chromadb

# Create a Chroma DB instance

chroma_client = chromadb.Client()

# Create a collection to store the documents
collection_name = "documents"
db = chroma_client.create_collection(collection_name)

# Define an embedding function to create vector representations of documents
def embedding_function(document):
    # Convert the document text to lowercase and remove punctuation
    text = document['text'].lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = text.split()

    # Create a vector representation of the tokens
    vector = []
    for token in tokens:
        # Use an LLM to generate a semantic representation of the token
        token_vector = langchain.get_embedding(token)
        vector.extend(token_vector)

    return vector

# Define a function to answer questions using an LLM
def answer_question(question, document):
    # Load a question answering model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

    # Encode the question and document
    encoded_question = tokenizer(question, return_tensors="pt")
    encoded_document = tokenizer(document, return_tensors="pt")

    # Generate an answer to the question
    with torch.no_grad():
        output = model(**encoded_question, **encoded_document)

    # Extract the answer from the model's output
    answer = output.start_logits.argmax(-1).item()
    answer_text = document[answer:answer + output.end_logits.argmax(-1).item()]

    return answer_text

# Add documents to the Chroma DB
documents = [
    {"id": 1, "text": "This is a document about artificial intelligence."},
    {"id": 2, "text": "This is a document about machine learning."},
    {"id": 3, "text": "This is a document about natural language processing."},
]

for document in documents:
    db.add(collection_name, document['id'], document['text'], embedding_function)

# Perform a semantic search query
query = "What are the applications of artificial intelligence?"

# Search for documents that are semantically similar to the query
results = db.search(collection_name, query, embedding_function, top_k=10)

# Enhance the search results by answering questions using the LLM
for result in results:
    document_text = result['metadatas']['text']
    answer_text = answer_question(query, document_text)

    print(f"Document ID: {result['id']}")
    print(f"Document Text: {document_text}")
    print(f"Answer: {answer_text}")