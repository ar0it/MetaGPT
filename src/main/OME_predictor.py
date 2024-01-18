"""
Class to predict the OME XML from the given raw metadata text.
"""

import os
import openai
import autogen
import argparse
from openai import OpenAI
import xml.etree.ElementTree as ET
import time


class XMLPredictor:
    """
    Class to predict the OME XML from the given raw metadata text.
    """

    def __init__(self,
                 path_to_raw_metadata="/home/aaron/PycharmProjects/MetaGPT/raw_data/raw_Metadata_Image8.txt",
                 out_path="/home/aaron/PycharmProjects/MetaGPT/out/ome_xml.ome.xml",
                 path_to_ome_starting_point=None,
                 ome_xsd_path="/home/aaron/PycharmProjects/MetaGPT/raw_data/test.txt"):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """

        self.run = None
        self.assistant_id = None
        self.pre_prompt = None
        self.ome_xsd_path = ome_xsd_path
        self.path_to_raw_metadata = path_to_raw_metadata
        self.path_to_ome_xml = None
        self.client = OpenAI()
        self.raw_metadata = self.read_raw_metadata()
        self.ome_xml = None
        self.out_path = out_path
        self.messages = []
        if path_to_ome_starting_point is not None:
            self.ome_starting_point = self.read_ome_xml(path_to_ome_starting_point)
        self.assistant = None
        self.thread = None
        self.message = None
        self.model = "gpt-4-1106-preview"  # "gpt-3.5-turbo"
        self.main()

    def main(self):
        """
        Predict the OME XML from the raw metadata
        """
        print("- - - Initializing assistant - - -")
        with open("assistant_id.txt", "r") as f:
            self.assistant_id = f.read()
            try:
                print("Trying to retrieve assistant: " + self.assistant_id)
                self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
                print("Successfully retrieved assistant")
            except:
                print("Assistant not found, creating new assistant")
                self.init_assistant()
        print("- - - Generating Thread - - -")
        self.thread = self.client.beta.threads.create()
        print("- - - Generating Prompt - - -")
        self.generate_message(msg=self.raw_metadata)
        print("- - - Predicting OME XML - - -")
        self.ome_xml = self.run_message()
        print("- - - Exporting OME XML - - -")
        self.export_ome_xml()
        print("- - - Shut down assistant - - -")

    def subdivide_raw_metadata(self):
        """
        Subdivide the raw metadata into appropriate chunks
        """
        self.raw_metadata = self.raw_metadata

    def read_ome_xml(self, path):
        """
        This method reads the ome xml file and returns the root element.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        return root

    def export_ome_xml(self):
        """
        Export the OME XML to a file
        """
        with open(self.out_path, "w") as f:
            f.write(self.ome_xml)

    def read_raw_metadata(self):
        """
        Read the raw metadata from the file
        """
        print(self.path_to_raw_metadata)
        with open(self.path_to_raw_metadata, "r") as f:
            raw_metadata = f.read()
        return raw_metadata

    def init_assistant(self):
        """
        Define the assistant that will help the user with the task
        :return:
        """
        self.pre_prompt = ("You are a microscopy expert who is specialized at curating metadata for images. You will "
                           "get raw unstructured metadata and are supposed to correctly identify and transcribe the "
                           "metadata to the ome xml standard. Remember you have the OME XSD accessible via retrieval, "
                           "make use of it to follow the standard. Only ever respond with the XML. Try to be as "
                           "complete as"
                           "possible and only use structured annotations when absolutely necessary.")

        file = self.client.files.create(
            file=open(self.ome_xsd_path, "rb"),
            purpose="assistants"
        )
        print(file)
        self.assistant = self.client.beta.assistants.create(
            instructions=self.pre_prompt,
            name="OME XML Assistant",
            model=self.model,
            tools=[{"type": "retrieval"}],
            file_ids=[file.id]
        )
        with open("assistant_id.txt", "w") as f:
            f.write(self.assistant.id)

    def generate_message(self, msg):
        """
        Generate the prompt from the raw metadata
        """
        self.message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=msg
        )

    def make_assistant(self):
        """
        Create the assistant
        :return:
        """
        self.init_assistant()

    def run_message(self):
        """
        Predict the OME XML from the raw metadata
        """
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )

        while self.run.status == "in_progress" or self.run.status == "queued":
            print(self.run.status)
            self.run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id
            )
            print("Polling for run completion...")
            time.sleep(5)

        print(self.run.status)
        if self.run.status == "failed":
            print("Run failed")
            print(self.run)
            return None

        assistant_response = None
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        print(messages.data[0].content[0].text.value)
        assistant_response = messages.data[0].content[0].text.value

        return assistant_response

    def get_ome_xml(self):
        """
        Get the OME XML
        :return:
        """
        return self.ome_xml


def parse_args():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser(description="Predict the OME XML from the raw metadata")
    parser.add_argument("--path_in", type=str, help="Path to the raw metadata file")
    parser.add_argument("--path_out", type=str, help="Path to the OME XML file")
    return parser.parse_args()


if __name__ == "__main__":
    path = parse_args().path_in
    xml_predictor = XMLPredictor()

# %%
