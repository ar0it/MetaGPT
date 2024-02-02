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

        self.name = __name__
        self.model = "gpt-4-turbo-preview"  # this always links to the most recent (gpt4) model
        self.pre_prompt = (f"You are {self.name}, an AI assistant specialized in curating metadata for images. Your "
                           f"task is"
                           "to transform raw, unstructured metadata into well-formed XML, adhering to the OME XML"
                           "standard. You have access to the OME XSD for reference. Furthermore you will be provided"
                           "with an OME-XML starting points generated by Bioformats. The starting point might contain"
                           "some of the information from the raw data but could also be incomplete. Your responses"
                           "should be exclusively in XML format, aligning closely with the standard. Strive for"
                           "completeness and validity. Rely on structured annotations only when necessary.")

        self.assistant = None

    def create_assistant(self, assistant_id_path=None):
        """
        Define the assistant that will help the user with the task
        :return:
        """
        print(f"- - - Creating {self.name}- - -")
        if assistant_id_path is not None:
            with open(assistant_id_path, "r") as f:
                assistant_id = f.read()
            try:
                print("Trying to retrieve assistant from ID: " + assistant_id)
                self.assistant = self.client.beta.assistants.retrieve(assistant_id)
                print("Successfully retrieved assistant")
            except:
                print(f"Assistant not found, creating new {self.name}")
                self.new_assistant()

        else:
            print(f"Assistant not found, creating new {self.name}")
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
        with open("../{self.name}_assistant_id.txt", "w") as f:
            f.write(self.assistant.id)
