import os
import warnings

from openai import OpenAI


class AssistantTemplate:
    """
    This class is a template for all assistants. It contains the basic structure and methods that all assistants should
    have.
    """

    def __init__(self, ome_xsd_path, client):

        self.ome_xsd_path = ome_xsd_path
        self.client = client

        self.name = None
        self.model = "gpt-4-turbo-preview"  # this always links to the most recent (gpt4) model
        self.description = None
        self.instructions = None

        self.assistant = None
        self.file_id = "file-nLWAvxK87WnJgTmOCsxIpLiY"

    def create_assistant(self, assistant_id_path=None):
        """
        Define the assistant that will help the user with the task
        :return:
        """
        print(f"- - - Creating {self.name}- - -")

        if assistant_id_path is None:
            assistant_id_path = f"./assistant_ids/{self.name}_assistant_id.txt"

        with open(assistant_id_path, "r") as f:
            assistant_id = f.read()

        try:
            self.assistant = self.client.beta.assistants.retrieve(assistant_id)

        except Exception as e:
            # create a new on if the assistant does not exist
            warnings.warn(e.message)
            self.new_assistant()

        # update the assistant prompt
        self.assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant.id,
            instructions=self.instructions,
            description=self.description,
            file_ids=[self.file_id],
        )
        print("Successfully retrieved assistant")
        return self.assistant

    def new_assistant(self):
        """
        Define the assistant that will help the user with the task
        :return:
        """
        # try to get file from the client else create it
        if self.file_id is None:
            file = self.client.files.create(
                file=open(self.ome_xsd_path, "rb"),
                purpose="assistants"
            )
            self.file_id = file.id

        self.assistant = self.client.beta.assistants.create(
            description=self.description,
            instructions=self.instructions,
            name=self.name,
            model=self.model,
            tools=[{"type": "retrieval"}],
            file_ids=[self.file_id]
        )
        with open(f"./assistant_ids/{self.name}_assistant_id.txt", "w") as f:
            f.write(self.assistant.id)
