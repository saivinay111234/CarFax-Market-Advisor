# 🚗 Car Market Intelligence Agent using RAG + CrewAI + Gemini

A powerful AI Agent application that helps users analyze car listings and answer vehicle-related queries using **RAG (Retrieval-Augmented Generation)**, **ChromaDB**, and **Gemini LLM**, built with **CrewAI** and **Streamlit**.

---

## 📌 Key Features

- 🔍 **RAG Retrieval**: Retrieves relevant cars from a local ChromaDB based on user queries.
- 🤖 **Agent Reasoning**: Uses CrewAI agents to autonomously choose between tools (RAG or Knowledge LLM) based on query type.
- 💡 **LLM Summarization**: Contextual responses are refined and summarized by Google Gemini.
- 🧠 **Intelligent Tool Selection**: No hardcoded logic—agent decides the best tool dynamically.
- 📄 **Dual Output View**: Displays raw context and agent's final insights side-by-side.
- 🖥️ **Streamlit UI**: Simple and intuitive frontend to interact with the AI agent.

---

## 🚀 Demo

| Context (RAG Output) | Agent Insights |
|----------------------|----------------|
| Car listings from ChromaDB | Summarized market insights and recommendations from the AI Agent |

---

## 🏗️ Tech Stack

- 🧠 **CrewAI** – Multi-agent reasoning framework
- 🌐 **Google Gemini API** – LLM backend
- 🧱 **ChromaDB** – Vector database for storing car listings
- 🤖 **SentenceTransformers** – Embedding function for semantic search
- 📊 **Streamlit** – Web frontend
- 🐍 **Python** – Core language

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/car-intelligence-agent.git
cd car-intelligence-agent
