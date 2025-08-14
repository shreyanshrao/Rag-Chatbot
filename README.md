# 📚 RAG-Based Chatbot with Google Gemini 1.5

A Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **Google's Gemini 1.5 model**, and **Streamlit** for an interactive UI.  
The bot can answer domain-specific questions by retrieving relevant context from a vector database and generating detailed responses.

---

## 🚀 Features
- **Google Gemini 1.5** (`gemini-1.5-pro` or `gemini-1.5-flash`) for high-quality responses.
- **Retrieval-Augmented Generation** to ground answers in real data.
- **Vector store integration** for storing and searching embeddings.
- **Streamlit frontend** for easy interaction.
- **Python-based** for flexibility and rapid development.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **LangChain**
- **langchain-google-genai**
- **Google Generative AI API**
- **Streamlit**
- **FAISS / Chroma** (or your chosen vector store)

**How It Works (My Process)**
Data Preparation

Collected domain-specific documents (PDFs, text files, or scraped content).

Cleaned and preprocessed text using LangChain’s document loaders.

Split text into chunks (e.g., 500–1000 characters) to make retrieval more efficient.

Embedding Generation

Used GoogleGenerativeAIEmbeddings from langchain-google-genai to convert text chunks into vector embeddings.

Stored these embeddings in a vector store (FAISS/Chroma) for fast similarity search.

Retrieval-Augmented Generation (RAG) Pipeline

Built a retriever using the vector store.

For every user query:

Retrieve top relevant chunks using similarity search.

Pass retrieved context + query to the Gemini 1.5 model.

Generate a grounded, context-aware answer.
- **dotenv** for API key management

---
