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
        self.pre_prompt = None

        self.assistant = None

    def create_assistant(self, assistant_id_path=None):
        """
        Define the assistant that will help the user with the task
        :return:
        """
        print(f"- - - Creating {self.name}- - -")
        print(f"Assistant ID path: {assistant_id_path}")
        if assistant_id_path is not None:
            try:
                with open(assistant_id_path, "r") as f:
                    assistant_id = f.read()
                print("Trying to retrieve assistant from ID: " + assistant_id)
                self.assistant = self.client.beta.assistants.retrieve(assistant_id)
                # update the assistant prompt
                self.assistant = self.client.beta.assistants.update(
                    id=assistant_id,
                    instructions=self.pre_prompt,
                )
                print("Successfully retrieved assistant")
            except:
                print(f"Error retrieving assistant, creating new {self.name}")
                self.new_assistant()

        else:
            try:
                print(os.getcwd())
                path = f"./assistant_ids/{self.name}_assistant_id.txt"
                with open(path, "r") as file:
                    id = file.read()
                self.assistant = self.client.beta.assistants.retrieve(id)
                # update the assistant prompt
                self.assistant = self.client.beta.assistants.update(
                    id=self.assistant.id,
                    instructions=self.pre_prompt,
                )
                print(f"Retrieved {self.name}")
            except:
                print(f"Error retrieving assistant, creating new {self.name}")
                self.new_assistant()

        return self.assistant

    def new_assistant(self):
        """
        Define the assistant that will help the user with the task
        :return:
        """
        file = self.client.files.create(
            file=open(self.ome_xsd_path, "rb"),
            purpose="assistants"
        )
        self.assistant = self.client.beta.assistants.create(
            instructions=self.pre_prompt,
            name=self.name,
            model=self.model,
            tools=[{"type": "retrieval"}],
            file_ids=[file.id]
        )
        with open(f"./assistant_ids/{self.name}_assistant_id.txt", "w") as f:
            f.write(self.assistant.id)
