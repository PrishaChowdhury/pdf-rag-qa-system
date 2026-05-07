import streamlit as st
from PyPDF2 import PdfReader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


# Configure page
st.set_page_config(page_title="PDF RAG", layout="centered")
st.title("📄 PDF Question Answering")

# API Key input
GOOGLE_API_KEY = "AIzaSyBkQfOLLFl3tKZmpmQqEtbJ_BDKAJ9AnZY"  # Add your API key here

if not GOOGLE_API_KEY:
    st.warning("Please add your Google API key in the code")
    st.stop()

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Tech Stack:**
    - LLM: Gemini Pro
    - Embeddings: all-MiniLM-L6-v2
    - Vector DB: FAISS
    - Framework: LangChain
    """)
    
    if "chunks_count" in st.session_state:
        st.metric("Document Chunks", st.session_state.chunks_count)

# File upload
pdf_file = st.file_uploader("Upload PDF", type="pdf")

if pdf_file:
    # Extract text
    pdf_reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    
    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    st.session_state.chunks_count = len(chunks)

    # Create embeddings and vector store
    with st.spinner("Processing PDF..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'} )
        vectorstore = FAISS.from_texts(chunks, embeddings)
        st.session_state.vectorstore = vectorstore

 # Setup LLM and QA chain
    llm = ChatGoogleGenerativeAI(
        model= "gemini-2.5-flash", 
        google_api_key=GOOGLE_API_KEY, 
        temperature=0.3
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    st.session_state.qa_chain = qa_chain
    
    st.success(f"✅ PDF processed! ({len(chunks)} chunks created)")

# Question answering
if "qa_chain" in st.session_state:
    st.divider()
    question = st.text_input("💬 Ask a question about your document:")
    
    if question:
        with st.spinner("Generating answer..."):
            response = st.session_state.qa_chain.invoke({"query": question})
            
            st.write("**Answer:**")
            st.write(response["result"])
            
            # Show source chunks
            with st.expander("📚 View source chunks"):
                for i, doc in enumerate(response["source_documents"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()