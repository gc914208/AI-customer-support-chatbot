# 🤖 ApexAssist — AI Customer Support Bot (RAG Application)

ApexAssist is a high-performance, context-aware AI customer support agent. Built using **Streamlit**, **LangChain**, and **Groq**, it delivers low-latency streaming responses and utilizes a dynamic **Retrieval-Augmented Generation (RAG)** pipeline to answer customer queries grounded in custom documentation.

---

## ✨ Features

- **🚀 Lightning-Fast Inference:** Powered by Groq's Llama 3.3 70B model with real-time response streaming.
- **📄 Custom Knowledge Base (RAG):** Upload `.pdf` or `.txt` support documentation on the fly to provide product-specific answers without model hallucination.
- **⚡ Local Vector Store:** Uses open-source Hugging Face sentence transformers (`all-MiniLM-L6-v2`) and **FAISS** for fast, zero-API-cost local vector embeddings.
- **🔄 Dual-Mode Architecture:** Seamlessly toggles between general conversational support and strict, document-grounded context retrieval when files are uploaded.
- **💼 Production-Ready UX:** Clean UI with session state management, chat history reset, and sidebar controls.

---

## 🛠️ Tech Stack

- **Frontend / Framework:** [Streamlit](https://streamlit.io/)
- **LLM Orchestration:** [LangChain](https://www.langchain.com/)
- **Inference Engine:** [Groq](https://groq.com/) (`llama-3.3-70b-versatile`)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
- **Document Loaders:** PyPDF, TextLoader

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11 or 3.12** installed on your system.
- A free **Groq API Key** (get one at [console.groq.com](https://console.groq.com/)).

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/apex-ai-customer-support.git](https://github.com/your-username/apex-ai-customer-support.git)
   cd apex-ai-customer-support
Create a virtual environment:

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Configure Secrets:
Create a .streamlit/secrets.toml file in the root directory and add your Groq key:

Ini, TOML
GROQ_API_KEY = "gsk_your_groq_api_key_here"
(Alternatively, you can enter the key directly in the app's sidebar UI during execution).

Run the Streamlit app:

Bash
streamlit run app.py
🌐 Cloud Deployment (Streamlit Cloud)
Push your code to a GitHub repository.

Go to share.streamlit.io and create a new app pointing to your repository and app.py.

In Advanced Settings, ensure the Python version is set to 3.11 or 3.12.

In Secrets, paste your API key:

Ini, TOML
GROQ_API_KEY = "gsk_your_groq_api_key_here"
Deploy!

📁 Project Structure
Plaintext
├── .streamlit/
│   └── secrets.toml      # Local secrets configuration (Git ignored)
├── app.py                # Main Streamlit application entry point
├── requirements.txt      # Python dependencies
├── runtime.txt           # Forces Python 3.11 on Streamlit Cloud
└── README.md             # Project documentation
🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

📄 License
This project is open-source under the MIT License.
