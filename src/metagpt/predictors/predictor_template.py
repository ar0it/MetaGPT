import os
# import autogen
import argparse
from openai import OpenAI
import xml.etree.ElementTree as ET
from lxml import etree
import time
import xmlschema
import instructor


class PredictorTemplate:
    """
    A template for creating a new predictor. A predictor utilizes one or several assistants to predict the OME XML from
    the raw metadata.
    """
    def __init__(self):
        self.client = OpenAI() #instructor.patch(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.run = None
        self.pre_prompt = None
        self.response = None
        self.messages = []
        self.assistants = []
        self.threads = []
        self.vector_stores = []
        self.message = None
        self.out_path = None
        self.token_in_cost = 5/1e6
        self.token_out_cost = 15/1e6
        self.attempts = 0
        self.max_attempts = 3
        self.name = self.__class__.__name__
        self.model = "gpt-4o-mini"
        
    def predict(self)->dict:
        """
        Predict the OME XML from the raw metadata
        """

        raise NotImplementedError

    def add_attempts(self, i:float=1):
        """
        Add an attempt to the attempt counter. Normalized by the number of assistants.
        """
        self.attempts += i/len(self.assistants)

    def init_thread(self):
        """
        Initialize the thread
        """
        self.thread = self.client.beta.threads.create()

    def subdivide_raw_metadata(self):
        """
        Subdivide the raw metadata into appropriate chunks
        """
        self.raw_metadata = self.raw_metadata

    def read_ome_as_string(self, path):
        """
        Read the OME XML as a string
        """
        with open(path, "r") as f:
            ome_xml = f.read()

        return ome_xml

    def read_ome_as_xml(self, path):
        """
        This method reads the ome xml file and returns the root element.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        # convert the root to a string
        ome_string = str(ET.tostring(root))
        return ome_string

    def export_ome_xml(self):
        """
        Export the OME XML to a file
        """
        with open(self.out_path + f"final_{time.time()}.ome.xml", "w") as f:
            f.write(self.response)

    def read_raw_metadata(self):
        """
        Read the raw metadata from the file
        """
        with open(self.path_to_raw_metadata, "r") as f:
            raw_metadata = f.read()
        return raw_metadata

    def generate_message(self, msg=None):
        """
        Generate the prompt from the raw metadata
        """
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=msg
        )
        self.messages.append(message)
        return message

    def get_response(self):  # TODO: Change name
        """
        Predict the OME XML from the raw metadata
        """
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )

        while self.run.status == "in_progress" or self.run.status == "queued":
            print("Polling for run completion...")
            print(self.run.status)
            self.run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id,
                #instructions= self.assistant.instructions,
            )

            time.sleep(5)

        print(self.run.status)
        if self.run.status == "failed":
            print("Run failed")
            print(self.run)
            return None

        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        self.response = messages.data[0].content[0].text.value
        self.export_ome_xml()
        validation = self.validate(self.response)
        print(validation)

        """
        
        
        if validation is not None:
            print("Validation failed")
            msg = "There seems to be an issue with the OME XML you provided. Please fix this error:\n" + str(validation)
            print(msg)
            self.generate_message(msg=msg)
            self.run_message()
        """

    def validate(self, ome_xml) -> Exception:
        """
        Validate the OME XML against the OME XSD
        :return:
        """
        try:
            self.xsd_schema.validate(ome_xml)
        except Exception as e:
            return e

    def clean_assistants(self):
        """
        Clean up the assistants
        """
        try:
            for a in self.assistants:
                self.client.beta.assistants.delete(assistant_id=a.id)
            self.assistants = []
            for t in self.threads:
                self.client.beta.threads.delete(thread_id=t.id)
            self.threads = []
            for v in self.vector_stores:
                self.client.beta.vector_stores.delete(vector_store_id=v.id)
            self.vector_stores = []
        except Exception as e:
            print("There was an issue when cleaning up the assistants", e)
    def get_cost(self, run):
        """
        Get the cost of the prediction
        """
        try:
            return run.usage.completion_tokens * self.token_out_cost + run.usage.prompt_tokens * self.token_in_cost
        except Exception as e:
            return 0


