import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page Config
st.set_page_config(
    page_title="Apex - Customer Support",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for Loading Dots
st.markdown("""
<style>
.loading-dots {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background-color: #f0f2f6;
    border-radius: 12px;
    margin-bottom: 10px;
}
.loading-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #007bff;
    animation: bounce 1.4s infinite ease-in-out both;
}
.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }
.loading-dots span:nth-child(3) { animation-delay: 0s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1.0); }
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 ApexAssist Support Bot (Powered by Groq)")
st.write("Ask any customer support question below.")

# Retrieve Keys
groq_api_key = st.secrets.get("GROQ_API_KEY", None)
gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if not groq_api_key:
        groq_api_key = st.text_input("Enter Groq API Key", type="password")
        
    st.markdown("---")
    st.subheader("📄 Optional: Upload Document for RAG")
    st.caption("Upload a file if you want the bot to answer from custom knowledge.")
    uploaded_files = st.file_uploader(
        "Upload support file (.pdf, .txt)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.subheader("App Management")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🗑️ Clear Uploaded Documents", use_container_width=True):
        st.session_state.vector_store = None
        st.success("Cleared documents! Bot reset to standard mode.")
        st.rerun()

# Optional RAG Ingestion (Only runs if a user uploads a file)
if uploaded_files:
    if not gemini_api_key:
        gemini_api_key = st.sidebar.text_input("Enter Gemini API Key (Required for embeddings)", type="password")
    
    if gemini_api_key and st.session_state.vector_store is None:
        with st.spinner("Processing documents for custom knowledge..."):
            all_docs = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_filepath = tmp_file.name
                
                try:
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_filepath)
                    else:
                        loader = TextLoader(tmp_filepath, encoding="utf-8")
                    
                    docs = loader.load()
                    split_docs = text_splitter.split_documents(docs)
                    all_docs.extend(split_docs)
                finally:
                    os.remove(tmp_filepath)
            
            if all_docs:
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                st.session_state.vector_store = FAISS.from_documents(all_docs, embeddings)
                st.success("Document loaded! Bot will now use this document to answer.")

# Render Chat History
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

# User Query Handler
if user_query := st.chat_input("How can I help you today?"):
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)
        
    with st.chat_message("assistant"):
        if not groq_api_key:
            st.error("Please provide your Groq API Key in Streamlit Secrets or sidebar.")
        else:
            try:
                # Fast Groq LLM (Llama-3.3-70b gives high quality & high speed)
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=groq_api_key,
                    temperature=0.1,
                    max_tokens=300,
                    streaming=True
                )
                
                loader_placeholder = st.empty()
                loader_placeholder.markdown(
                    '<div class="loading-dots"><span></span><span></span><span></span></div>', 
                    unsafe_allow_html=True
                )
                
                # PATH A: RAG Mode (If a document was uploaded)
                if st.session_state.vector_store is not None:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
                    system_prompt = (
                        "You are 'ApexAssist', an empathetic customer support AI.\n"
                        "Answer concise, factual support questions using ONLY the context below:\n\n{context}"
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
                    
                    def stream_rag():
                        first_chunk = True
                        for chunk in chain.stream({"input": user_query}):
                            if "answer" in chunk:
                                if first_chunk:
                                    loader_placeholder.empty()
                                    first_chunk = False
                                yield chunk["answer"]
                    
                    answer = st.write_stream(stream_rag())

                # PATH B: Standard Chat Mode (No document needed!)
                else:
                    system_prompt = "You are 'ApexAssist', an empathetic and helpful AI customer support assistant. Answer concisely."
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    chain = prompt | llm
                    
                    def stream_standard():
                        first_chunk = True
                        for chunk in chain.stream({"input": user_query}):
                            if first_chunk:
                                loader_placeholder.empty()
                                first_chunk = False
                            yield chunk.content
                    
                    answer = st.write_stream(stream_standard())
                
                st.session_state.chat_history.append(("assistant", answer))
                
            except Exception as e:
                loader_placeholder.empty()
                st.error(f"Error: {str(e)}")
