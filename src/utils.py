import requests
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)


def fetch_ollama_models(api_url, api_key):
    """
    Fetch available models from the Ollama API.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    logging.info(f"Fetching models from Ollama API at {api_url}")
    response = requests.get(f"{api_url}/api/tags", headers=headers)
    if response.status_code == 200:
        models = response.json().get("models", [])
        logging.info(f"Fetched {len(models)} models")
        return [model["name"] for model in models]
    else:
        logging.error("Failed to fetch models from Ollama API.")
        st.error("Failed to fetch models from Ollama API.")
        return []
