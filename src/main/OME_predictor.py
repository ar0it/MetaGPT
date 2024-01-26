"""
Class to predict the OME XML from the given raw metadata text.
"""

import os
import openai
import autogen
import argparse
from openai import OpenAI
import xml.etree.ElementTree as ET
from lxml import etree
import time
import xmlschema


class XMLPredictor:
    """
    Class to predict the OME XML from the given raw metadata text.
    """

    def __init__(self,
                 path_to_raw_metadata="/home/aaron/PycharmProjects/MetaGPT/raw_data/raw_Metadata_Image8.txt",
                 out_path="/home/aaron/PycharmProjects/MetaGPT/out/ome_xml.ome.xml",
                 path_to_ome_starting_point=None,
                 ome_xsd_path="/home/aaron/PycharmProjects/MetaGPT/raw_data/ome_xsd.txt"):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """
        self.xml_schema_path = "/home/aaron/PycharmProjects/MetaGPT/raw_data/ome_xsd.txt"
        self.xsd_schema = xmlschema.XMLSchema(self.xml_schema_path)
        self.run = None
        self.assistant_id = None
        self.pre_prompt = None

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
        self.model = "gpt-4-turbo-preview"  # "gpt-3.5-turbo"
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
        self.run_message()

        # print("- - - Exporting OME XML - - -")
        # self.export_ome_xml()
        # print("- - - Shut down assistant - - -")

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
        self.pre_prompt = ("You are OMEGPT, an AI assistant specialized in curating metadata for images. Your task is"
                           "to transform raw, unstructured metadata into well-formed XML, adhering to the OME XML"
                           "standard. You have access to the OME XSD for reference. Your responses should be"
                           "exclusively in XML format, aligning closely with the standard. Strive for completeness and"
                           "rely on structured annotations only when essential.")

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

        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        ome_xml = messages.data[0].content[0].text.value
        ome_xml = ome_xml.split("</OME>")[0].split("<OME")[1]
        self.ome_xml = "<OME" + ome_xml + "</OME>"
        print(self.ome_xml)
        validation = self.validate()
        if validation is not None:
            print("Validation failed" + str(validation))
            msg = "There seems to be an issue with the OME XML you provided. The error I get is: " + str(validation)
            self.generate_message(msg=msg)
            self.run_message()

    def validate(self) -> Exception:
        """
        Validate the OME XML against the OME XSD
        :param xml_path:
        :param xsd_path:
        :return:
        """

        try:
            self.xsd_schema.validate(self.ome_xml)
        except Exception as e:
            return e

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
