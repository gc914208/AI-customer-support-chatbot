ApexAssist - AI Customer Support Bot

ApexAssist is a high-speed AI customer support assistant built with **Streamlit**, **Groq (Llama-3.3-70b)**, and **LangChain**. It functions both as a standard fast support chatbot and as a **Retrieval-Augmented Generation (RAG)** engine when provided with company-specific knowledge base files.

---

## ✨ Features

- **🚀 Dual Mode Operation:**
  - **Standard Mode:** Answers general support inquiries instantly using `llama-3.3-70b-versatile` via Groq.
  - **RAG Mode:** Dynamically activated when files are uploaded, grounding answers strictly in custom knowledge base documents.
- **⚡ Ultra-Fast Responses:** Powered by Groq's LPU hardware for near-instant streaming answers.
- **📄 Document Support:** Accepts `.pdf` and `.txt` support documents.
- **🔍 Vector Retrieval:** Uses Google Gemini embeddings (`models/gemini-embedding-001`) with **FAISS** for fast similarity searching.
- **💬 Interactive Chat UI:** Includes custom CSS loading indicators, streaming response output, and chat history controls.

---

## 🛠️ Tech Stack

- **Frontend/Framework:** [Streamlit](https://streamlit.io/)
- **LLM Engine:** [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`)
- **Embeddings:** [Google Gemini API](https://ai.google.dev/) (`models/gemini-embedding-001`)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Database:** [FAISS CPU](https://github.com/facebookresearch/faiss)

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Run the Streamlit App
Bash
streamlit run app.py
🔑 Environment Variables & Secrets
To run the application, configure your API keys in Streamlit Secrets (.streamlit/secrets.toml locally, or in Streamlit Cloud Settings):

Ini, TOML
GROQ_API_KEY = "gsk_YourGroqApiKeyHere"
GEMINI_API_KEY = "YourGeminiApiKeyHere"  # Required only when using RAG document search
📖 How to Use
Ask General Questions: Type directly into the chat box for instant general responses.

Enable Custom Knowledge (RAG Mode):

Expand the Sidebar.

Upload your company support documents (.pdf or .txt).

Enter your GEMINI_API_KEY (if not pre-configured in secrets).

Ask specific questions about your products, shipping, returns, or policies!

Reset: Use the Clear Chat History or Clear Uploaded Documents buttons in the sidebar at any time.

📁 Repository Structure
Plaintext
├── app.py              # Main Streamlit application entry point
├── requirements.txt    # Python package dependencies
├── README.md           # Project documentation
└── apex_wear_faq.txt   # Sample company knowledge base file

---

### How to Add It to GitHub in 3 Steps:

1. Open your repository on **GitHub**.
2. Click **Add file** ➔ **Create new file**.
3. Name the file **`README.md`**, paste the text above, and click **Commit changes...** at the top right.
