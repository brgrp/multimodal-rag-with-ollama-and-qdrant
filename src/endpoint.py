from abc import ABC, abstractmethod
import logging
import requests
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)


class IRequestSender(ABC):
    @abstractmethod
    def send(
        self, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a request to the specified endpoint.

        :param endpoint: The endpoint URL
        :param headers: The request headers
        :param payload: The request payload
        :return: The response from the endpoint
        """
        pass


class RequestSender(IRequestSender):
    def send(
        self, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a request to the specified endpoint.

        :param endpoint: The endpoint URL
        :param headers: The request headers
        :param payload: The request payload
        :return: The response from the endpoint
        """
        try:
            logging.info(f"Sending request to {endpoint}")
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                logging.error("Failed to parse the response as JSON.")
                raise SystemExit("Failed to parse the response as JSON.")
        except requests.RequestException as e:
            logging.error(f"Failed to make the request. Error: {e}")
            raise SystemExit(f"Failed to make the request. Error: {e}")


class EndPoint:
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        request_sender: IRequestSender = RequestSender(),
    ):
        """
        Initialize the EndPoint class.

        :param api_key: API key for the endpoint
        :param endpoint: The endpoint URL
        :param request_sender: Instance of IRequestSender
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        self.request_sender = request_sender
        logging.info(f"Initialized EndPoint with endpoint: {endpoint}")

    def send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a payload to the endpoint.

        :param payload: The request payload
        :return: The response from the endpoint
        """
        logging.info("Sending payload to endpoint")
        return self.request_sender.send(self.endpoint, self.headers, payload)


if __name__ == "__main__":
    # Usage
    api_key = "your_api_key"
    endpoint_url = "http://localhost:11434/api/chat"
    request_sender = RequestSender()
    endpoint = EndPoint(api_key, endpoint_url, request_sender)

    payload = {
        "model": "llama3.2",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant named DocumentFinder",
            },
            {
                "role": "user",
                "content": "Who let the dog out?",
            },
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800,
        "stream": False,
    }
    response = endpoint.send(payload)
    print(response["message"]["content"])
