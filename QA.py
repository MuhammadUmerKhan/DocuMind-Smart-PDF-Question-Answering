import streamlit as st
import os
import subprocess
from PyPDF2 import PdfReader  # For extracting text from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into manageable chunks
from langchain.embeddings import HuggingFaceEmbeddings  # Converts text into vector embeddings
from langchain.vectorstores import FAISS  # Efficient vector-based search storage
from langchain.chains import RetrievalQA  # Handles retrieval-augmented question answering
from langchain.llms import CTransformers  # Runs lightweight LLMs locally
from langchain.prompts import PromptTemplate  # Defines structured prompts for the LLM

# App configuration
st.set_page_config(
    page_title="DocuMind AI - Smart PDF Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model Configuration - Defines the path and settings for the TinyLlama model
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

MODEL_CONFIG = {
    "model": MODEL_PATH,
    "model_type": "llama",
    "config": {
        "max_new_tokens": 1024,  # Controls response length
        "temperature": 0.25,  # Controls randomness in output
        "context_length": 2048,  # Defines max input size
        "gpu_layers": 0  # Uses CPU for inference
    }
}

# Ensure Model is Available
def download_model():
    """Automatically downloads the model if it's missing."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        st.warning("Model file not found. Downloading... ⏳")
        subprocess.run(["wget", "-O", MODEL_PATH, MODEL_URL], check=True)
        st.success("Model downloaded successfully! ✅")

def process_pdf(pdf):
    """Extracts text from a PDF file and validates if it's readable."""
    try:
        pdf_reader = PdfReader(pdf)
        text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        return text if text.strip() else None  # Returns None if text extraction fails
    except Exception as e:
        st.error(f"PDF Processing Error: {str(e)}")
        return None

def create_knowledge_base(text):
    """Splits text and converts it into a searchable vector database."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    chunks = text_splitter.split_text(text)  # Splits document into meaningful sections
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Converts text to embeddings
    return FAISS.from_texts(chunks, embeddings)  # Stores embeddings in FAISS for fast retrieval

def load_llm():
    """Loads the LLM after ensuring it's downloaded."""
    download_model()
    return CTransformers(**MODEL_CONFIG)  # Loads the TinyLlama model

def main():
    """Streamlit app main function."""
    st.title("📄 Smart Document Analysis")  # UI title
    
    uploaded_file = st.file_uploader("📂 Upload PDF Document", type=["pdf"])  # Upload section
    
    if uploaded_file:
        with st.spinner("🔍 Analyzing document..."):
            text = process_pdf(uploaded_file)  # Extracts text from the uploaded PDF
            if not text:
                st.error("❌ Unreadable or empty PDF content")
                return
            
            knowledge_base = create_knowledge_base(text)  # Converts extracted text into a searchable knowledge base
            llm = load_llm()  # Loads the LLM for answering questions
            
            # Sets up Retrieval-Augmented Generation (RAG) pipeline
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=knowledge_base.as_retriever(search_kwargs={'k': 3}),  # Fetches top-3 relevant chunks
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template="""<|system|>
You are an expert document analysis assistant. Follow these rules:
1. Answer strictly based on the document content
2. Be concise but comprehensive
3. Structure answers with bullet points when appropriate
4. Highlight key terms in **bold**
5. If uncertain, state \"The document doesn't specify\"

Context: {context}</s>
<|user|>
{question}</s>
<|assistant|>
""",
                        input_variables=["context", "question"]
                    )
                },
                return_source_documents=True  # Enables displaying sources for transparency
            )
            
            st.success("✅ Document analysis complete! Ask your questions below.")
            
            if query := st.text_input("🔎 Enter your question:"):  # Accepts user query
                with st.spinner("💡 Generating expert answer..."):
                    result = qa_chain({'query': query})  # Passes query to RAG system
                    answer = postprocess_answer(result['result'])  # Post-processes the output
                    
                    st.subheader("📝 Expert Analysis")
                    st.markdown(answer)  # Displays the final answer
                    
                    with st.expander("📌 View Supporting Passages"):
                        for doc in result['source_documents']:  # Displays retrieved passages
                            st.caption(doc.page_content)
                            st.divider()

def postprocess_answer(text):
    """Formats AI-generated answers for better readability."""
    text = text.strip()
    if not text.endswith(('.', '!', '?')):  # Ensures proper sentence termination
        text += '...'
    return text.replace('**', '**')  # Ensures Markdown bold formatting

if __name__ == "__main__":
    main()  # Runs the Streamlit app
