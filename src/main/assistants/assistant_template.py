import os
import warnings

from openai import OpenAI


class AssistantTemplate:
    """
    This class is a template for all assistants. It contains the basic structure and methods that all assistants should
    have.
    """

    def __init__(self, ome_xsd_path: str = None, client: OpenAI = None):

        self.ome_xsd_path = ome_xsd_path
        self.client = client

        self.name = None
        self.model = "gpt-4-turbo-preview"  # this always links to the most recent (gpt4) model
        self.description = None
        self.instructions = None
        self.tools = [{"type": "file_search"}]
        self.vector_store_id = None
        self.assistant = None

    def create_assistant(self, assistant_id_path: str = None):
        """
        Define the assistant that will help the user with the task
        :return:
        """
        print(f"- - - Creating {self.name}- - -")

        if assistant_id_path is None:
            assistant_id_path = f"./assistant_ids/{self.name}_assistant_id.txt"

        try:
            with open(assistant_id_path, "r") as f:
                assistant_id = f.read()
            self.assistant = self.client.beta.assistants.retrieve(assistant_id)
            # update the assistant prompt
            self.assistant = self.client.beta.assistants.update(
                assistant_id=self.assistant.id,
                instructions=self.instructions,
                description=self.description,
            )

        except Exception as e:
            # create a new on if the assistant does not exist
            warnings.warn(e)
            self.new_assistant()

        print("Successfully retrieved assistant")
        return self.assistant

    def new_assistant(self):
        """
        Define the assistant that will help the user with the task
        :return:
        """
        # try to get file from the client else create it
        if self.vector_store_id is None:
            # Create a vector store caled "Financial Statements"
            vector_store = self.client.beta.vector_stores.create(name="OME XML schema")
            self.vector_store_id = vector_store.id

            # Ready the files for upload to OpenAI
            file_paths = ["/home/aaron/PycharmProjects/MetaGPT/raw_data/ome_xsd.txt"]
            file_streams = [open(path, "rb") for path in file_paths]

            # Use the upload and poll SDK helper to upload the files, add them to the vector store,
            # and poll the status of the file batch for completion.
            file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
            )

            # You can print the status and the file counts of the batch to see the result of this operation.
            print(file_batch.status)
            print(file_batch.file_counts)

        self.assistant = self.client.beta.assistants.create(
            description=self.description,
            instructions=self.instructions,
            name=self.name,
            model=self.model,
            tools=self.tools,
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store_id]}},
            response_model=self.response_model
        )
        self.save_assistant_id()

    def save_assistant_id(self):
        """
        Save the assistant id to a file.
        :return:
        """
        with open(f"./assistant_ids/{self.name}_assistant_id.txt", "w") as f:
            f.write(self.assistant.id)
