import logging
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.simple_rag import create_simple_rag
from src.utils import fetch_ollama_models

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()


def reinitialize_simple_rag():
    """
    Reinitialize the SimpleRAG instance with updated parameters.
    """
    if "temp_dir" in st.session_state and st.session_state.temp_dir:
        st.session_state.simple_rag = create_simple_rag(
            documents_dir=st.session_state.temp_dir,
            api_key=st.session_state.api_key,
            api_url=st.session_state.api_url,
            huggingface_token=st.session_state.huggingface_token,
            model=st.session_state.model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            max_tokens=st.session_state.max_tokens,
        )
        logging.info("Reinitialized SimpleRAG instance")


def app():
    """
    Streamlit app for document retrieval and response generation.
    """
    st.set_page_config(page_title="Document Retrieval and Response Generation")
    st.title("📄 Document Retrieval and Response Generation")
    st.markdown(
        "Upload your text documents and generate responses based on your queries."
    )

    # Initialize session state keys
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "api_url" not in st.session_state:
        st.session_state.api_url = None
    if "huggingface_token" not in st.session_state:
        st.session_state.huggingface_token = None

    # API Key and URL Input
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    if api_key and api_url and huggingface_token:
        st.session_state.api_key = api_key
        st.session_state.api_url = api_url
        st.session_state.huggingface_token = huggingface_token
        st.success(
            "API key, URL, and Hugging Face token are set as environment variables."
        )
        logging.info(
            "API key, URL, and Hugging Face token are set as environment variables."
        )
    else:
        st.subheader("API Configuration")
        st.session_state.api_key = st.text_input(
            "Enter your API key:", type="password", placeholder="Your API Key"
        )
        st.session_state.api_url = st.text_input(
            "Enter your API URL:", placeholder="https://api.example.com"
        )
        st.session_state.huggingface_token = st.text_input(
            "Enter your Hugging Face token:",
            type="password",
            placeholder="Your Hugging Face Token",
        )

    # Fetch models from Ollama API
    if st.session_state.api_key and st.session_state.api_url:
        models = fetch_ollama_models("http://localhost:11434", st.session_state.api_key)
    else:
        models = []

    # Add GUI elements for payload parameters
    model = st.selectbox(
        "Model", options=models, key="model", on_change=reinitialize_simple_rag
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        key="temperature",
        on_change=reinitialize_simple_rag,
    )
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        key="top_p",
        on_change=reinitialize_simple_rag,
    )
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=100,
        max_value=5000,
        value=3000,
        key="max_tokens",
        on_change=reinitialize_simple_rag,
    )

    # File Upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Select text files to upload:",
        accept_multiple_files=True,
        type=["txt"],
        key="file_uploader",
    )

    if uploaded_files:
        # Ensure no duplicate files are uploaded
        unique_files = {file.name: file for file in uploaded_files}.values()
        st.session_state.uploaded_files = unique_files
        st.success(f"{len(unique_files)} file(s) selected for processing.")
        logging.info(f"{len(unique_files)} file(s) selected for processing.")

    # Process Documents Button
    if st.button("Process Documents"):
        if not uploaded_files:
            st.error("Please provide all required inputs.")
            logging.error("No files uploaded or API details missing.")
            return

        temp_dir = tempfile.mkdtemp()
        st.session_state.temp_dir = temp_dir

        for uploaded_file in st.session_state.uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.session_state.simple_rag = create_simple_rag(
            documents_dir=temp_dir,
            api_key=st.session_state.api_key,
            api_url=st.session_state.api_url,
            huggingface_token=st.session_state.huggingface_token,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        st.session_state.documents_processed = True
        st.success("Documents processed successfully!")
        logging.info("Documents processed successfully!")

    # Query Input and Response Generation
    if (
        "documents_processed" in st.session_state
        and st.session_state.documents_processed
    ):
        st.subheader("Query Input")
        user_query = st.text_input(
            "Enter your query:", key="query", placeholder="What do you want to know?"
        )

        if user_query:
            with st.spinner("Searching for relevant documents..."):
                retrieved_content, titles = st.session_state.simple_rag.retrieve(
                    user_query
                )
            st.write("### Retrieved Titles:")
            st.write(titles, unsafe_allow_html=True)  # Allow HTML for better formatting
            with st.spinner("Generate response..."):
                response = st.session_state.simple_rag.generate_response(
                    user_query, retrieved_content
                )
            st.write("### Response:")
            st.write(
                response, unsafe_allow_html=True
            )  # Allow HTML for better formatting


if __name__ == "__main__":
    app()
