import os
import tempfile
import streamlit as st

# LangChain Imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
st.caption("Powered by Groq Llama-3.3-70B, Google Embeddings, and RAG")

# -----------------------------------------------------------------------------
# Sidebar Configuration & Keys
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Retrieve API keys from Streamlit secrets or sidebar input
    groq_api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    google_api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Google AI API Key", type="password")
    
    st.divider()
    st.header("📄 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF support manuals/docs",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Check for API Keys
if not groq_api_key or not google_api_key:
    st.warning("⚠️ Please provide both your **Groq API Key** and **Google AI API Key** in `.streamlit/secrets.toml` or via the sidebar to continue.")
    st.stop()

# Set environmental variables for LangChain integrations
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

# -----------------------------------------------------------------------------
# Vectorstore & Document Processing (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Processing Knowledge Base...")
def build_vectorstore(_pdf_files):
    """Processes uploaded PDFs and creates a FAISS vector store."""
    all_docs = []
    
    for pdf_file in _pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    
    # Embed and index with FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

retriever = None
if uploaded_files:
    try:
        vectorstore = build_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        st.sidebar.success(f"Loaded {len(uploaded_files)} PDF(s) into vector database.")
    except Exception as e:
        st.sidebar.error(f"Error processing PDFs: {str(e)}")

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
    temperature=0.2,
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
        
        # If knowledge base is loaded, use RAG Retrieval Chain
        if retriever:
            system_prompt = (
                "You are ApexAssist, an expert customer support agent for Apex. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you do not know the answer based on the context, politely inform the user. "
                "Keep your response clear, professional, and concise.\n\n"
                "Context:\n{context}"
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
        
        # Direct LLM fallback when no PDFs are uploaded
        else:
            system_prompt = "You are ApexAssist, an expert AI customer support assistant. Answer helpful questions concisely."
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
