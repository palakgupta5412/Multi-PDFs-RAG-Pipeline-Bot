# Multi PDF RAG Chatbot

This project extends the RAG architecture to support **multiple PDF documents**.

The chatbot can retrieve information from an entire collection of documents and provide context-aware answers along with source citations.

---

## 🚀 Features

* Chat with multiple PDF documents
* Sentence-aware text chunking
* Semantic search using embeddings
* Persistent FAISS vector database
* Source citation (document + page number)
* Query rewriting for improved retrieval
* Streaming responses from LLM
* Efficient document loading

---

## 🧠 Concepts Implemented

This project implements a full **multi-document RAG pipeline** including:

* Multi-file document ingestion
* Sentence-based chunking
* Embedding generation
* Vector similarity search
* Retrieval augmentation
* Context construction
* LLM answer generation

---

## 🛠 Tech Stack

* **Python**
* **Groq API**
* **Sentence Transformers**
* **FAISS**
* **PyPDF**
* **NLTK**
* **NumPy**

---

## 📂 Project Structure

```
MultiPDF-RAG/
│
├── docs/
│   ├── notes1.pdf
│   ├── notes.pdf
│   └── notes3.pdf
│
├── multipdfrag.py
├── faiss_index.bin
├── chunks.pkl
└── README.md
```

---

## ⚙️ Setup

```
pip install pypdf sentence-transformers faiss-cpu groq numpy nltk
```

---

## ▶️ Run the chatbot

```
python multipdfrag.py
```

---

## 🔍 Retrieval Process

```
Multiple PDFs
 ↓
Text Extraction
 ↓
Sentence Chunking
 ↓
Embedding Generation
 ↓
FAISS Vector Database
 ↓
Similarity Search
 ↓
Context Assembly
 ↓
LLM Response Generation
```

---

## 🧩 Key Functionalities

### 📄 Multi-document ingestion

Loads all PDFs inside a folder automatically.

### ✂️ Sentence-based chunking

Improves retrieval accuracy compared to character chunking.

### 🧠 Semantic search

Uses embeddings to find meaning-based matches.

### 📚 Source citation

Displays which document and page the answer came from.

### ⚡ Persistent vector database

Embeddings are stored so they don't need to be recomputed.

### 🔄 Query rewriting

Improves search queries for better retrieval.

### 💬 Streaming responses

Displays the answer progressively like modern AI assistants.

---

## 📌 Key Learnings

* Building scalable RAG systems
* Managing multi-document knowledge bases
* Vector similarity search
* Retrieval optimization techniques
* Designing modular AI pipelines

---

## 🔮 Future Improvements

* Hybrid retrieval (BM25 + embeddings)
* Support for Excel / CSV documents
* Web interface with React
* AI agents for document analysis
* Deployment as a web application

---

## 👩‍💻 Author

**Palak Gupta**
