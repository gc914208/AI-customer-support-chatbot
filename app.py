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

st.title("🤖 ApexAssist AI Support Bot (Gemini RAG)")
st.write("Upload knowledge base documents (.pdf or .txt) and chat with your Gemini-powered customer support bot.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.subheader("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Training Files (.pdf, .txt)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

# Initialize Session States
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process Uploaded Files
if uploaded_files and gemini_api_key:
    if st.session_state.vector_store is None:
        with st.spinner("Processing and indexing documents with Gemini embeddings..."):
            all_docs = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            
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
                    # Set API key in environment for LangChain components
                    os.environ["GOOGLE_API_KEY"] = gemini_api_key
                    # Updated to the new supported model: models/gemini-embedding-001
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

# User Input
if user_query := st.chat_input("Ask a customer support question..."):
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)
        
    with st.chat_message("assistant"):
        if not gemini_api_key:
            st.error("Please provide your Gemini API Key in the sidebar configuration.")
        elif st.session_state.vector_store is None:
            st.warning("Please upload at least one knowledge base document (.pdf or .txt) to activate RAG.")
        else:
            with st.spinner("Generating answer using Gemini..."):
                try:
                    os.environ["GOOGLE_API_KEY"] = gemini_api_key
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.5-flash",
                        temperature=0.2,
                    )
                    
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    
                    system_prompt = (
                        "You are 'ApexAssist', an empathetic and professional AI customer support assistant.\n"
                        "Answer the user's question using ONLY the provided context below. If you do not know the answer "
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
                    
                    response = rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    
                    st.write(answer)
                    st.session_state.chat_history.append(("assistant", answer))
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
