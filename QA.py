import streamlit as st
import os
import subprocess
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate

# App configuration
st.set_page_config(
    page_title="DocuMind AI - Smart PDF Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model Configuration
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

MODEL_CONFIG = {
    "model": MODEL_PATH,
    "model_type": "llama",
    "config": {
        "max_new_tokens": 1024,
        "temperature": 0.25,
        "context_length": 2048,
        "gpu_layers": 0
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
    """Extract text from PDF with validation."""
    try:
        pdf_reader = PdfReader(pdf)
        text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        return text if text.strip() else None
    except Exception as e:
        st.error(f"PDF Processing Error: {str(e)}")
        return None

def create_knowledge_base(text):
    """Convert text into an optimized knowledge base."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

def load_llm():
    """Load LLM model after ensuring it's downloaded."""
    download_model()
    return CTransformers(**MODEL_CONFIG)

def main():
    st.title("📄 Smart Document Analysis")
    
    uploaded_file = st.file_uploader("📂 Upload PDF Document", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("🔍 Analyzing document..."):
            text = process_pdf(uploaded_file)
            if not text:
                st.error("❌ Unreadable or empty PDF content")
                return
            
            knowledge_base = create_knowledge_base(text)
            llm = load_llm()
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=knowledge_base.as_retriever(search_kwargs={'k': 3}),
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
                return_source_documents=True
            )
            
            st.success("✅ Document analysis complete! Ask your questions below.")
            
            if query := st.text_input("🔎 Enter your question:"):
                with st.spinner("💡 Generating expert answer..."):
                    result = qa_chain({'query': query})
                    answer = postprocess_answer(result['result'])
                    
                    st.subheader("📝 Expert Analysis")
                    st.markdown(answer)
                    
                    with st.expander("📌 View Supporting Passages"):
                        for doc in result['source_documents']:
                            st.caption(doc.page_content)
                            st.divider()

def postprocess_answer(text):
    """Format and enhance AI-generated answers."""
    text = text.strip()
    if not text.endswith(('.', '!', '?')):
        text += '...'
    return text.replace('**', '**')  # Markdown bold formatting

if __name__ == "__main__":
    main()