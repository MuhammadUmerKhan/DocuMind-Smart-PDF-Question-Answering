# 📝 DocuMind AI: Smart PDF Question Answering System

![PDF Analysis](https://blog.apify.com/content/images/size/w1200/2023/11/Extract-PDF-documents-for-question-answering-from-a-PDF-1.png)

In today's data-driven world, extracting meaningful insights from documents is crucial for businesses, researchers, and individuals. This project focuses on building an intelligent PDF question-answering system that leverages state-of-the-art language models to provide accurate, context-aware answers from uploaded PDF documents. 📚✨

This repository provides a complete solution for document analysis, including text extraction, semantic understanding, and interactive question answering. By combining advanced NLP techniques with efficient document processing, we aim to make document analysis faster, smarter, and more accessible.

## 💖 Table of Contents
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [Usage Instructions](#usage-instructions)
- [Running the Project](#running-the-project)
- [Live Demo](#live-demo)
- [Future Plans](#future-plans)
- [License](#license)

---

## ❓ Problem Statement

Extracting meaningful information from PDF documents is often time-consuming and error-prone. Traditional methods rely on manual searching or keyword-based approaches, which fail to capture the context and nuances of the content. This is especially challenging for large documents or technical papers.

This project aims to develop an intelligent system that can:
- **Automatically extract and process text** from PDFs 📝
- **Understand the semantic context** of the document 🧠
- **Provide accurate, context-aware answers** to user queries 🔍

---

## 🛠️ Methodology

1. **Document Processing:**
   - Extract text from PDFs using **PyPDF2** 📝➡️📂
   - Split text into meaningful chunks using **RecursiveCharacterTextSplitter** ✂️
   - Handle edge cases like empty pages or unreadable text 🧠

2. **Semantic Understanding:**
   - Generate embeddings using **Sentence Transformers (MiniLM)** 🧠
   - Create a searchable knowledge base with **FAISS** 🔍

3. **Question Answering:**
   - Use **TinyLlama-1.1B** (quantized) for efficient, CPU-friendly inference 🦥
   - Implement **RetrievalQA** chain for context-aware answers 🎯
   - Optimize prompts for better answer quality ✨

4. **User Interface:**
   - Build an interactive web app using **Streamlit** 🌐
   - Provide real-time feedback and visualizations 📊

---

## 🚀 Key Features

- **📂 PDF Upload & Processing:** Supports any standard PDF document.
- **🔍 Context-Aware Answers:** Provides accurate, document-specific responses.
- **💡 Interactive Interface:** User-friendly web app with real-time feedback.
- **🖥️ CPU-Friendly:** Optimized for deployment on low-resource systems.
- **📝 Markdown Support:** Answers are formatted for better readability.

---

## 🚀 Usage Instructions

### 📂 Clone the Repository
```bash
git clone https://github.com/MuhammadUmerKhan/PDF-Question-and-Answering-System.git
cd PDF-Question-and-Answering-System
```

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### ⬇️ Download the Model
Download the quantized TinyLlama model:
```bash
mkdir -p models
wget -O models/tinyllama.gguf https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

---

## 🏃‍♂️ Running the Project

### 🌐 Start the Streamlit App
```bash
streamlit run app.py
```

Open your browser and navigate to:
```
http://localhost:8501/
```

---

## 🏆 Key Findings

- **TinyLlama-1.1B** provides a good balance between accuracy and efficiency for CPU-based systems.
- **FAISS** enables fast semantic search, even for large documents.
- **Prompt engineering** significantly improves answer quality and relevance.
- The system performs well on technical documents, research papers, and reports.

---

## 🛡️ Future Plans

- If **high computation resources** become available, we plan to explore **more advanced LLMs** with enhanced contextual understanding and reasoning capabilities.
- Potential upgrades include models like **LLaMA 3, Mistral, or GPT-4** for **more accurate and sophisticated question-answering**.
- **Multi-document support** for cross-document analysis 📚
- **Table and image extraction** for richer insights 🎨
- **Integration with cloud storage** for seamless document access ☁️

---

## 💖 Conclusion

This project demonstrates how advanced NLP techniques can be used to build intelligent document analysis systems. By combining efficient text processing, semantic understanding, and interactive interfaces, we’ve created a tool that makes PDF analysis faster, smarter, and more accessible.

---

💡 **Feel free to contribute, raise issues, or suggest improvements!**

📌 **License:** MIT License 🔓

---

### Screenshots

1. **Upload Interface**  
   ![Upload Interface](https://github.com/MuhammadUmerKhan/PDF-Question-and-Answering-System/blob/main/imgs/ss2.png)

2. **Question Answering**  
   ![Question Answering](https://github.com/MuhammadUmerKhan/PDF-Question-and-Answering-System/blob/main/imgs/ss1.png)

3. **Supporting Evidence**  
   ![Supporting Evidence](https://github.com/MuhammadUmerKhan/PDF-Question-and-Answering-System/blob/main/imgs/ss3.png)

---
