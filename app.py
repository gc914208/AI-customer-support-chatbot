import os
import tempfile
import streamlit as st

# LangChain Imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ApexAssist | AI Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ApexAssist — AI Customer Support")
st.caption("Powered by Groq Llama-3.3-70B and RAG")

# -----------------------------------------------------------------------------
# Sidebar Configuration & Keys
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Retrieve Groq API key only
    groq_api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    
    st.divider()
    st.header("📄 Knowledge Base")
    st.caption("Optional: Upload files with companies information for more accurate customer support answers with RAG.")
    uploaded_files = st.file_uploader(
        "Upload support docs (.pdf, .txt)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Check for Groq Key
if not groq_api_key:
    st.warning("⚠️ Please provide your **Groq API Key** in `.streamlit/secrets.toml` or via the sidebar to continue.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# -----------------------------------------------------------------------------
# Vectorstore & Document Processing (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Processing Knowledge Base...")
def build_vectorstore(_uploaded_files):
    """Processes uploaded PDFs/TXTs and creates a FAISS vector store using local embeddings."""
    all_docs = []
    
    for uploaded_file in _uploaded_files:
        suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")
            
            docs = loader.load()
            all_docs.extend(docs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)
    
    # Embed locally using open-source Hugging Face model (No API key required!)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

retriever = None
if uploaded_files:
    try:
        vectorstore = build_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.sidebar.success(f"Loaded {len(uploaded_files)} document(s) into vector store.")
    except Exception as e:
        st.sidebar.error(f"Error processing documents: {str(e)}")

# -----------------------------------------------------------------------------
# Chat Memory & Session Setup
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you with Apex products and support today?"}
    ]

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------------------------------------------------------
# Core LLM & RAG Logic
# -----------------------------------------------------------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=300,
    streaming=True
)

if user_input := st.chat_input("Type your question here..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # RAG Mode (When files are uploaded)
        if retriever:
            system_prompt = (
                "You are ApexAssist, an expert customer support agent for Apex. "
                "Answer concise, factual support questions using ONLY the context below:\n\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": user_input})
            full_response = response.get("answer", "I couldn't find relevant information in the uploaded documentation.")
            response_placeholder.markdown(full_response)
        
        # Standard Mode (No files uploaded)
        else:
            system_prompt = "You are ApexAssist, an empathetic and helpful AI customer support assistant, for a e-commerce company named Apex wear. Answer concisely,"
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            chain = prompt | llm
            for chunk in chain.stream({"input": user_input}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
