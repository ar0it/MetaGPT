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
                 path_to_ome_xml=None,
                 path_to_ome_starting_point=None,
                 ome_xsd_path="/home/aaron/PycharmProjects/MetaGPT/raw_data/ome.xsd"):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """
        self.run = None
        self.pre_prompt = None
        self.ome_xsd_path = ome_xsd_path
        self.path_to_raw_metadata = path_to_raw_metadata
        self.path_to_ome_xml = None
        self.client = OpenAI()
        if path_to_ome_xml is None:
            self.path_to_ome_xml = os.path.join(os.path.dirname(self.path_to_raw_metadata), "ome_xml.ome.xml")
        self.raw_metadata = self.read_raw_metadata()
        self.ome_xml = None
        self.messages = []
        self.ome_starting_point = self.read_ome_xml(path_to_ome_starting_point)
        self.assistant = None
        self.thread = None
        self.message = None
        self.model = "gpt-3.5-turbo"
        self.main()

    def main(self):
        """
        Predict the OME XML from the raw metadata
        """
        print("- - - Initializing assistant - - -")
        self.init_assistant()
        print("- - - Generating Prompt - - -")
        self.generate_message(msg=self.raw_metadata)
        print("- - - Predicting OME XML - - -")
        pred = self.run_message()
        print("- - - Exporting OME XML - - -")
        self.export_ome_xml()
        print("- - - Shut down assistant - - -")
        self.assistant.delete()

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
        with open(self.path_to_ome_xml, "w") as f:
            f.write(self.ome_xml)

    def read_raw_metadata(self):
        """
        Read the raw metadata from the file
        """
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

        self.assistant = self.client.beta.assistants.create(
            instructions=self.pre_prompt,
            name="OME XML Assistant",
            model=self.model,
            tools=[{"type": "retrieval"}],
            file_ids=[file.id]
        )

        self.thread = self.client.beta.threads.create()

    def generate_message(self, msg):
        """
        Generate the prompt from the raw metadata
        """
        self.message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=msg
        )

    def run_message(self):
        """
        Predict the OME XML from the raw metadata
        """
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )

        while self.run.status != "complete":
            self.run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id
            )
            print("Polling for run completion...")
            time.sleep(1)

        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        print(messages)
        return messages

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
    xml_predictor = XMLPredictor(path_to_raw_metadata=path)
    print(xml_predictor.raw_metadata)

#%%
