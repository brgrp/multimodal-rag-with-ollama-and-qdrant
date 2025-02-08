import argparse
import logging
import os

from dotenv import load_dotenv

from src.simple_rag import create_simple_rag

logging.basicConfig(level=logging.WARNING)

# Load environment variables from .env file
load_dotenv()


def main():
    """
    Main function to run the document search and response generation.
    """
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    required_env_vars = ["API_KEY", "API_URL", "HUGGINGFACE_TOKEN"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(
            f"Required environment variables are not set: {', '.join(missing_vars)}"
        )
        raise EnvironmentError(
            f"Required environment variables are not set: {', '.join(missing_vars)}"
        )

    parser = argparse.ArgumentParser(
        description="Document search and response generation"
    )
    parser.add_argument(
        "--documents_dir",
        type=str,
        required=True,
        help="Directory containing text files",
    )
    args = parser.parse_args()

    simple_rag = create_simple_rag(
        documents_dir=args.documents_dir,
        api_key=api_key,
        api_url=api_url,
        huggingface_token=huggingface_token,
    )

    if not simple_rag:
        logging.error("Failed to create SimpleRAG instance.")
        exit(1)

    user_query = input("Enter your query: ")
    try:
        retrieved_content, retrieved_titles = simple_rag.retrieve(user_query)
        response = simple_rag.generate_response(user_query, retrieved_content)
        if response:
            logging.info("Response: %s", response)
            print("Response:", response)
        logging.info("Retrieved Titles: %s", retrieved_titles)
        print("Retrieved Titles:", retrieved_titles)
    except Exception as e:
        logging.error(f"Failed to process query. Error: {e}")


if __name__ == "__main__":
    main()
