"""
This module contains the PredictorTemplate class, which serves as a base class
for creating predictors that utilize OpenAI's API to generate OME XML from raw metadata.
"""

import os
import time
from typing import Optional, List, Dict, Any

import xml.etree.ElementTree as ET
from openai import OpenAI

import metagpt.utils.utils as utils

class PredictorTemplate:
    """
    A template for creating a new predictor. A predictor utilizes one or several assistants
    to predict the OME XML from the raw metadata.
    """

    def __init__(self):
        self.client = OpenAI()
        self.run: Optional[Any] = None
        self.pre_prompt: Optional[str] = None
        self.max_iter: int = 15
        self.run_iter: int = 0
        self.response: Optional[str] = None
        self.messages: List[Any] = []
        self.assistants: List[Any] = []
        self.threads: List[Any] = []
        self.vector_stores: List[Any] = []
        self.message: Optional[str] = None
        self.out_path: Optional[str] = None
        self.token_in_cost: float = 5/1e6
        self.token_out_cost: float = 15/1e6
        self.attempts: int = 0
        self.max_attempts: int = 3
        self.name: str = self.__class__.__name__
        self.model: str = "gpt-4o-mini"
        self.in_tokens: int = 0
        self.out_tokens: int = 0
        self.prompt: Optional[str] = None
        self.file_paths: List[str] = []
        self.temperature: float = 0.0
        self.last_error: Optional[Exception] = None
        self.last_response: Optional[str] = None
        self.last_error_msg: str = self._generate_last_error_msg()

    def _generate_last_error_msg(self) -> str:
        return f"""
        In a previous attempt you predicted {str(self.last_response)} and got the following
        error from the validator: {str(self.last_error)}. Please try the prediction again
        and make sure to not make the same mistake again.
        """

    def predict(self) -> Dict[str, Any]:
        """
        Predict the OME XML from the raw metadata.

        Returns:
            Dict[str, Any]: The predicted OME XML and related information.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def add_attempts(self, i: float = 1) -> None:
        """
        Add an attempt to the attempt counter. Normalized by the number of assistants.

        Args:
            i (float): The number of attempts to add. Defaults to 1.
        """
        self.attempts += i / len(self.assistants)

    def init_thread(self) -> None:
        """Initialize a new thread with the initial prompt and message."""
        self.in_tokens += utils.num_tokens_from_string(self.prompt)
        self.in_tokens += utils.num_tokens_from_string(self.message)
        self.thread = self.client.beta.threads.create(messages=[
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": "."},
            {"role": "user", "content": self.message}
        ])
        self.threads.append(self.thread)

    def subdivide_raw_metadata(self) -> None:
        """Subdivide the raw metadata into appropriate chunks."""
        # This method is currently a placeholder and should be implemented if needed
        pass

    def read_ome_as_string(self, path: str) -> str:
        """
        Read the OME XML as a string from a file.

        Args:
            path (str): The path to the OME XML file.

        Returns:
            str: The contents of the OME XML file as a string.
        """
        with open(path, "r") as f:
            return f.read()

    def read_ome_as_xml(self, path: str) -> str:
        """
        Read the OME XML file and return the root element as a string.

        Args:
            path (str): The path to the OME XML file.

        Returns:
            str: The root element of the OME XML as a string.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode')

    def export_ome_xml(self) -> None:
        """Export the OME XML response to a file."""
        if self.out_path and self.response:
            with open(os.path.join(self.out_path, f"final_{time.time()}.ome.xml"), "w") as f:
                f.write(self.response)

    def read_raw_metadata(self) -> str:
        """
        Read the raw metadata from the file.

        Returns:
            str: The contents of the raw metadata file.

        Raises:
            FileNotFoundError: If the path_to_raw_metadata is not set or the file doesn't exist.
        """
        if not hasattr(self, 'path_to_raw_metadata'):
            raise FileNotFoundError("path_to_raw_metadata is not set")
        with open(self.path_to_raw_metadata, "r") as f:
            return f.read()

    def generate_message(self, msg: Optional[str] = None) -> Any:
        """
        Generate a new message in the thread.

        Args:
            msg (Optional[str]): The message content. If None, uses self.message.

        Returns:
            Any: The created message object.
        """
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=msg or self.message
        )
        self.messages.append(message)
        return message

    def get_response(self) -> None:
        """
        Predict the OME XML from the raw metadata and handle the response.
        """
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )

        while self.run.status in ["in_progress", "queued"]:
            print(f"Polling for run completion... Status: {self.run.status}")
            time.sleep(5)
            self.run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id,
            )

        print(f"Run status: {self.run.status}")
        if self.run.status == "failed":
            print("Run failed")
            print(self.run)
            return

        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        self.response = messages.data[0].content[0].text.value
        self.export_ome_xml()
        validation = self.validate(self.response)
        print(f"Validation result: {validation}")

    def validate(self, ome_xml: str) -> Optional[Exception]:
        """
        Validate the OME XML against the OME XSD.

        Args:
            ome_xml (str): The OME XML string to validate.

        Returns:
            Optional[Exception]: The exception if validation fails, None otherwise.
        """
        try:
            self.xsd_schema.validate(ome_xml)
            return None
        except Exception as e:
            return e

    def clean_assistants(self) -> None:
        """Clean up the assistants, threads, and vector stores."""
        try:
            for a in self.assistants:
                self.client.beta.assistants.delete(assistant_id=a.id)
            self.assistants.clear()

            for t in self.threads:
                self.client.beta.threads.delete(thread_id=t.id)
            self.threads.clear()

            for v in self.vector_stores:
                self.client.beta.vector_stores.delete(vector_store_id=v.id)
            self.vector_stores.clear()
        except Exception as e:
            print(f"There was an issue when cleaning up the assistants: {e}")

    def get_cost(self) -> Optional[float]:
        """
        Calculate the cost of the prediction based on token usage.

        Returns:
            Optional[float]: The calculated cost, or None if there was an error.
        """
        try:
            return self.out_tokens * self.token_out_cost + self.in_tokens * self.token_in_cost
        except Exception as e:
            print(f"Error calculating cost: {e}")
            return None

    def init_vector_store(self) -> None:
        """Initialize the vector store and upload file batches."""
        self.vector_store = self.client.beta.vector_stores.create(
            name=f"Vector Store for {self.name}",
        )
        with utils.file_context_manager(self.file_paths) as file_streams:
            file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=self.vector_store.id, files=file_streams
            )
        self.vector_stores.append(self.vector_store)