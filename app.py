import os
import tempfile
import streamlit as st

import chromadb
import ollama

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile


# 🧠 System Prompt
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


# 🧾 Process PDF and Split
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks."""

    # Create temp file and write PDF content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Save path

    # Load and split the PDF after the file is closed
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()

    # Delete temp file after it's no longer in use
    os.unlink(temp_file_path)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)



# 📚 Initialize or Get ChromaDB Collection
def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


# ➕ Add Processed Chunks to Vector DB
def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("✅ Data added to the vector store!")


# 🔍 Search Vector DB
def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


# 💬 Call LLM to Get Final Answer
def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    response_found = False
    for chunk in response:
        if not chunk["done"]:
            response_found = True
            yield chunk["message"]["content"]
    if not response_found:
        yield "⚠️ No response received from the model."


# 📊 CrossEncoder Re-Ranking
def re_rank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    if not documents:
        return "", []

    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)

    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]] + "\n\n"
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


# 🖥️ Streamlit App UI - Enhanced
st.set_page_config(page_title="📖 RAG PDF Q&A", layout="wide")


# 🔧 Sidebar Upload Section
with st.sidebar:
    st.title("📚 Upload Your Document")
    st.markdown("Easily extract insights from your PDF files using a smart AI assistant.")
    uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

    if uploaded_file:
        st.success("✅ PDF Uploaded Successfully!", icon="📂")
        process = st.button("⚡ Process & Index File")

        if process:
            file_name_clean = uploaded_file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, file_name_clean)
            st.info("📁 Document chunks added to vector DB.")

# 🎯 Main App Interface
st.title(" Smart Document QA System")
st.markdown("Ask questions based on the content of your uploaded document.")

prompt = st.text_area("🔎 Enter your question below:", placeholder="e.g., What are the key findings from this report?")
ask = st.button("🎯 Ask the AI")

# Layout split for better UX
col1, col2 = st.columns([2, 1])

if ask and prompt:
    with st.spinner("🔍 Retrieving context and generating answer..."):
        results = query_collection(prompt)
        context_chunks = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, context_chunks)

    with col1:
        st.subheader("📜 Answer")
        st.write_stream(call_llm(context=relevant_text, prompt=prompt))

    with col2:
        with st.expander("🔍 Retrieved Chunks"):
            st.write(results)

        with st.expander("⭐ Top Ranked Chunks"):
            st.write(relevant_text_ids)
            for idx in relevant_text_ids:
                st.markdown(f"**Chunk {idx}:**")
                st.code(context_chunks[idx][:500], language="text")

# Optional: Footer
st.markdown("---")
st.markdown(
    "<center><small>🚀 Built with Streamlit, LangChain, Ollama & ChromaDB</small></center>",
    unsafe_allow_html=True
)
