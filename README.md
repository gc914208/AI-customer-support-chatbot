ApexAssist - AI Customer Support Bot

**ApexAssist** is a fast, intelligent customer support chatbot designed for e-commerce and business support. It answers customer queries instantly and can automatically pull information from custom company documents (like FAQs, shipping rules, and return policies).

---

## 🎯 How to Use ApexAssist

### 💬 1. General Support Chat (Standard Mode)
- Simply open the app and type any general support question into the chat box (e.g., *"How can customer support help me today?"*).
- Responses stream back instantly using **Groq (Llama 3.3)**.

### 📄 2. Company Knowledge Search (RAG Mode)
To make the bot answer using **specific company information**:
1. Open the **Sidebar** on the left.
2. Upload a support document (`.pdf` or `.txt`) — *you can try uploading the included `apex_wear_faq.txt` file!*
3. Once processed, ask questions about products, shipping, returns, or warranty:
   - *"What is the return window for items?"*
   - *"How should I wash my Apex Wear hoodie?"*
   - *"What is the standard shipping time to Canada?"*
4. ApexAssist will answer grounded **strictly** in your uploaded document's details!

---

## ✨ Key Features

- **⚡ Instant Streaming Responses:** Powered by Groq LPU hardware for near-zero delay.
- **📚 Smart Document Ingestion (RAG):** Upload any company PDF/TXT file to give the bot immediate domain knowledge.
- **🔄 Flexible Modes:** Switches automatically between general assistance and document-grounded answers.
- **🧹 Easy Controls:** Clear chat history or reset uploaded documents with a single click in the sidebar.

---

## 🛠️ How It Works (Tech Stack)

- **Frontend Interface:** [Streamlit](https://streamlit.io/)
- **LLM Engine:** [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`)
- **Embeddings & Search:** [Google Gemini API](https://ai.google.dev/) (`models/gemini-embedding-001`) with **FAISS Vector Database**
- **Orchestration:** [LangChain](https://www.langchain.com/)

---

## 🔑 Setup & Configuration

### Prerequisites & API Keys
This app uses Streamlit Secrets for configuration (`.streamlit/secrets.toml` locally, or via **Streamlit Cloud Settings**):

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here" # Needed for document embeddings
Running Locally
Clone the repository:

Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY
Install required packages:

Bash
pip install -r requirements.txt
Run the application:

Bash
streamlit run app.py
