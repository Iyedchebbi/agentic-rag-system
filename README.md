# 🧠 Agentic RAG System (CV AI Assistant)

An AI-powered Agentic RAG system that reads a CV, splits it into chunks, performs semantic search using FAISS, and generates intelligent answers using an LLM via OpenRouter.

---

## 🚀 Features
- Load and process CV text file
- Split documents into smart chunks
- Semantic search using FAISS vector database
- Embeddings powered by NVIDIA Llama Nemotron Embed VL 1B v2
- Answer generation using GPT OSS 120B (OpenRouter)
- Agent-style retrieval + reasoning pipeline
- Secure API key handling using .env
- Interactive terminal-based chatbot (CLI)

---

## 🧰 Tech Stack
- Python
- LangChain
- FAISS (Vector Database)
- OpenRouter API
  - NVIDIA Llama Nemotron Embed VL 1B v2 (Embeddings)
  - GPT OSS 120B (LLM)
- python-dotenv
- Requests

---

## 📁 Project Structure
project/
 ├── app.py
 ├── cv.txt
 ├── requirements.txt
 ├── .env
 ├── .env.example
 ├── .gitignore
 └── README.md

---

## ⚙️ Installation

1. Clone repo:
git clone https://github.com/your-username/your-repo.git

2. Install dependencies:
pip install -r requirements.txt

3. Create .env:
OPENROUTER_API_KEY=your_api_key_here

4. Run:
python app.py

---

## 💬 How It Works
User question → FAISS retrieval → context → GPT OSS 120B → answer

---

## 🧠 Models
- NVIDIA Llama Nemotron Embed VL 1B v2
- GPT OSS 120B (OpenRouter)
