import ollama
import re
import gradio as gr
from concurrent.futures import ThreadPoolExecutor

from chromadb.config import Settings
from chromadb import Client
from langchain.vectorstores import Chroma

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

# Initialize Ollama embeddings using DeepSeek-R1
embedding_function = OllamaEmbeddings(model="deepseek-r1")

# Initialize text splitter and vector store retriever
text_splitter = None
retriever = None


# Parallelize embedding generation
def generate_embedding(chunk):
    return embedding_function.embed_query(chunk.page_content)

# Combine document chunks into a single context string
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def process_pdf(file_path):
    if file_path is None:
        return None, None

    # Load the document using PyMuPDFLoader
    loader = PyMuPDFLoader(file_path)
    data = loader.load()

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)

    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(generate_embedding, chunks))

    # Initialize Chroma client and create/reset the collection
    client = Client(Settings())
    
    if "custom_collection" in [c.name for c in client.list_collections()]:
        client.delete_collection(name="custom_collection")
    
    collection = client.create_collection(name="custom_collection")

    # Add documents and embeddings to Chroma
    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[{'id': idx}],
            embeddings=[embeddings[idx]],
            ids=[str(idx)]  # Ensure IDs are strings
        )

    # Reinitialize retriever using Ollama embeddings for queries
    vectorstore = Chroma(collection_name="custom_collection", client=client, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever()

    return text_splitter, retriever

def retrieve_context(question, retriever):
    # Retrieve relevant document chunks
    retrieved_docs = retriever.invoke(question)
    # Combine the retrieved content into a single context string
    context = combine_docs(retrieved_docs)
    return context


def query_ollama(question, context):
    # Format the input prompt
    formatted_prompt = f"Question: {question}\n\nContext: {context}"

    # Query DeepSeek-R1 using Ollama
    response = ollama.chat(
        model="deepseek-r1",
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    response_content = response["message"]["content"]

    # Remove content between <think> and </think> tags to remove thinking output
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    return final_answer

def rag_chain(question, retriever):
    # Retrieve context and generate an answer using RAG
    context = retrieve_context(question, retriever)
    answer = query_ollama(question, context)
    return answer

def ask_question(file_path, reload_context, question):
    global text_splitter, retriever
    
    # Context reload required
    if reload_context or ((retriever is None or text_splitter is None) and file_path is not None):
        text_splitter, retriever = process_pdf(file_path)

    # Execute RAG query
    if retriever is not None and text_splitter is not None:
        result = rag_chain(question, retriever)
    else:
        result = query_ollama(question, "")

    return {result}

# Set up the Gradio interface
interface = gr.Interface(
    fn=ask_question,
    inputs=[
        gr.File(label="Upload PDF (optional)"),
        gr.Checkbox(label="Reload context", value=False),
        gr.Textbox(label="Ask a question"),
    ],
    outputs="text",
    title="Ask questions about your PDF",
    description="Use DeepSeek-R1 to answer your questions about the uploaded PDF document.",
)

interface.launch()