# Document Retrieval and Response Generation

This project retrieves relevant documents from a collection of text files based on user queries. It generates embeddings for the documents and uses these embeddings to find the most pertinent documents. The app then generates a response based on the content of the retrieved documents using an external API.

## Future Work

We plan to extend the functionality of this project to support more document types, including PDFs and images. This will involve integrating additional libraries and models to handle these formats and generate embeddings accordingly.

## Project Structure

The project is structured as follows:

- `main.py`: Contains the main logic of the project.
- `app.py`: Contains the Streamlit app for a web-based interface.
- `src/`: Contains the source code for the project.
  - `document_processor.py`: Contains the `DocumentProcessor` class for processing text documents.
  - `endpoint.py`: Contains the `EndPoint` class for sending requests to an external API.
  - `qdrant_index.py`: Contains the `QdrantIndex` class for indexing and searching embeddings using Qdrant.
  - `simple_rag.py`: Contains the `SimpleRAG` class for retrieving and generating responses.
  - `text_embedder.py`: Contains the `TextEmbedder` class for generating embeddings for text.
  - `utils.py`: Contains utility functions, such as fetching models from the Ollama API.

## Setup

### Prerequisites

- Python 3.13
- `uv` package manager

### Installation

1. **Setup Environment**:
    ```sh
    uv sync
    ```

2. **Set Environment Variables**:
    Update the `.env` file in the root directory and add the following environment variables:
    ```env
    API_KEY=your_api_key
    API_URL=your_api_url
    HUGGINGFACE_TOKEN=your_huggingface_token
    ```

## Usage

### Command Line Interface

1. **Run the Script**: Execute the `main.py` script with the `--documents_dir` argument pointing to the directory containing your text files:
    ```sh
    uv run main.py --documents_dir=path/to/your/documents
    ```

2. **Enter Query**: When prompted, enter your query. The script will search for the most relevant documents and generate a response based on their content.

### Streamlit App

1. **Run the Streamlit App**: Execute the `app.py` script using Streamlit:
    ```sh
    uv run streamlit run app.py
    ```

2. **Upload Documents and Enter Query**: Use the web interface to upload text files, process documents, and enter your query. The app will search for the most relevant documents and generate a response based on their content.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.