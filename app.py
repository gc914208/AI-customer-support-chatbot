import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page Configuration
st.set_page_config(
    page_title="ApexAssist - Gemini RAG Support Bot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for Animated Three-Dots Loading Indicator
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

st.title("🤖 ApexAssist AI Support Bot (Gemini RAG)")
st.write("Upload knowledge base documents (.pdf or .txt) and chat with your Gemini-powered customer support bot.")

# Retrieve API key automatically from Streamlit Secrets or sidebar fallback
gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)

# Initialize Session States
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Fallback input if Secrets key isn't provided
    if not gemini_api_key:
        gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
    else:
        st.success("🔑 Gemini API Key loaded permanently from secrets!")

    st.subheader("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Training Files (.pdf, .txt)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    # Reset Actions
    st.markdown("---")
    st.subheader("App Management")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
        
    if st.button("🗑️ Clear Database Documents", use_container_width=True):
        st.session_state.vector_store = None
        st.success("Database cleared! You can upload new files.")
        st.rerun()

# Process Uploaded Files
if uploaded_files and gemini_api_key:
    if st.session_state.vector_store is None:
        with st.spinner("Processing and indexing documents with Gemini embeddings..."):
            all_docs = []
            # High speed optimization: smaller chunk size
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
                try:
                    os.environ["GOOGLE_API_KEY"] = gemini_api_key
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                    st.session_state.vector_store = FAISS.from_documents(all_docs, embeddings)
                    st.success(f"Successfully indexed {len(all_docs)} text chunks!")
                except Exception as e:
                    st.error(f"Failed to initialize embeddings: {str(e)}")

# Chat Interface
st.subheader("💬 Chat with ApexAssist")

# Display historical messages
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

# User Input Handling
if user_query := st.chat_input("Ask a customer support question..."):
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)
        
    with st.chat_message("assistant"):
        if not gemini_api_key:
            st.error("Please provide your Gemini API Key in the sidebar or set it in Streamlit Secrets.")
        elif st.session_state.vector_store is None:
            st.warning("Please upload at least one knowledge base document (.pdf or .txt) to activate RAG.")
        else:
            try:
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
                
                # Fast Model & Generation Setup
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash",
                    temperature=0.1,
                    max_output_tokens=250,
                    streaming=True
                )
                
                # Fetch only top 2 relevant context chunks for low latency
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
                
                system_prompt = (
                    "You are 'ApexAssist', an empathetic and professional AI customer support assistant.\n"
                    "Answer the user's question concisely using ONLY the provided context below. If you do not know the answer "
                    "or if it is not in the context, say exactly:\n"
                    "'I'm sorry, but I don't have that information on hand. Let me connect you to a live support agent to help you further.'\n"
                    "Do not make up facts, URLs, or policies.\n\n"
                    "Context:\n{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Render 3-Dots Animation Placeholder
                loader_placeholder = st.empty()
                loader_placeholder.markdown(
                    '<div class="loading-dots"><span></span><span></span><span></span></div>', 
                    unsafe_allow_html=True
                )
                
                # Token Streaming Generator
                def stream_response():
                    first_chunk = True
                    for chunk in rag_chain.stream({"input": user_query}):
                        if "answer" in chunk:
                            if first_chunk:
                                loader_placeholder.empty()  # Removes dots on first token
                                first_chunk = False
                            yield chunk["answer"]
                
                answer = st.write_stream(stream_response())
                st.session_state.chat_history.append(("assistant", answer))
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
