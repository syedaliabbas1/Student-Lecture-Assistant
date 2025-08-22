# app.py
import streamlit as st
import tempfile
import os
import base64
from datetime import datetime
from rag import RAGPipeline   # <- your RAG class

# Document loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


# ================================
# Function to add background image
# ================================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Call background function (adjust filename if needed)
add_bg_from_local("bg-image2.jpg")


# ================================
# RAG Initialization
# ================================
if "rag" not in st.session_state:
    rag = RAGPipeline()
    rag.setup_environment()
    rag.initialize_components()
    rag.create_basic_rag_graph()
    st.session_state.rag = rag

# Manage conversations
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

# Track uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()

# Auto-create first chat
if not st.session_state.conversations:
    chat_id = str(datetime.now().timestamp())
    st.session_state.conversations[chat_id] = {"title": "Chat 1", "messages": []}
    st.session_state.active_chat = chat_id


# ================================
# Sidebar
# ================================
st.sidebar.title("📘 Lecture Notes Assistant")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your lecture notes", type=["pdf", "txt", "docx"])

if uploaded_file:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}_{hash(uploaded_file.getvalue())}"
    
    if file_id not in st.session_state.uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif uploaded_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                else:
                    st.sidebar.error("Unsupported file format")
                    st.stop()

                docs = loader.load()
                st.session_state.rag.load_and_process_documents(docs=docs)
                st.session_state.uploaded_files.add(file_id)
                st.sidebar.success(f"✅ Notes '{uploaded_file.name}' uploaded and processed")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {str(e)}")
            finally:
                if os.path.exists(file_path):
                    os.unlink(file_path)
    else:
        st.sidebar.info(f"📄 '{uploaded_file.name}' already processed")


# Show processed files
if st.session_state.uploaded_files:
    st.sidebar.subheader("📚 Processed Files")
    for i, file_id in enumerate(st.session_state.uploaded_files, 1):
        file_name = file_id.split('_')[0]
        st.sidebar.text(f"{i}. {file_name}")

# Clear documents
if st.sidebar.button("🗑️ Clear All Documents") and st.session_state.uploaded_files:
    st.session_state.uploaded_files.clear()
    st.sidebar.warning("Document tracking cleared. Upload files again to re-process.")

# Button for new chat
if st.sidebar.button("➕ New Chat"):
    chat_id = str(datetime.now().timestamp())
    st.session_state.conversations[chat_id] = {
        "title": f"Chat {len(st.session_state.conversations)+1}",
        "messages": []
    }
    st.session_state.active_chat = chat_id

# Chat history
st.sidebar.subheader("💬 Chat History")
for chat_id, chat_data in st.session_state.conversations.items():
    if st.sidebar.button(chat_data["title"], key=chat_id):
        st.session_state.active_chat = chat_id


# ================================
# Main UI
# ================================
st.title("🔎 Student Lecture Assistant")
st.caption("Upload your lecture notes and ask questions about them.")



chat = st.session_state.conversations[st.session_state.active_chat]

# Display chat history
for role, content in chat["messages"]:
    with st.chat_message(role):
        st.markdown(content)

# Input box
if user_input := st.chat_input("Ask a question about your notes..."):
    if not st.session_state.uploaded_files:
        st.warning("Please upload some lecture notes first!")
        st.stop()
    
    chat["messages"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag.query(user_input)
                answer = result["answer"]
                st.markdown(answer)
                chat["messages"].append(("assistant", answer))
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                chat["messages"].append(("assistant", error_msg))
