"""
Class to predict the OME XML from the given raw metadata text.
"""

import os
import openai
import autogen
import argparse

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]


class XMLPredictor:
    """
    Class to predict the OME XML from the given raw metadata text.
    """

    def __init__(self, path_to_raw_metadata, path_to_ome_xml=None):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """
        self.pre_prompt = None
        self.path_to_raw_metadata = path_to_raw_metadata
        self.path_to_ome_xml = None
        if path_to_ome_xml is None:
            self.path_to_ome_xml = os.path.join(os.path.dirname(path_to_raw_metadata), "ome_xml.ome.xml")
        self.raw_metadata = self.read_raw_metadata()
        self.ome_xml = None
        self.messages = []
        self.main()

    def main(self):
        """
        Predict the OME XML from the raw metadata
        """
        print("- - - Gnerating Prompt - - -")
        self.generate_promt()
        print("- - - Predicting OME XML - - -")
        self.predict_ome_xml()
        print("- - - Exporting OME XML - - -")
        self.export_ome_xml()

    def subdivide_raw_metadata(self):
        """
        Subdivide the raw metadata into appropriate chunks
        """
        self.raw_metadata = self.raw_metadata

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
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


        assistant = openai.client.beta.assistants.create(
            instruction="You are a microscopy expert who is specialized at curating metadata for images. You will ",
            name="OME XML Assistant",
            model="gpt-4-1106-preview",
            tools=[{"type": "code_interpreter"}]
        )
        file = client.files.create(
            file=open("path/to/file.txt", "rb"),
            purpose="assistants"
        )
        thread = client.assistants.create(
            message=[
                {
                    "role": "user",
                    "content": "I need help with my code",
                    "file_ids": [file.id]
                }
            ]
        )


    def generate_promt(self):
        """
        Generate the prompt from the raw metadata
        """

        self.pre_prompt = ("You are a microscopy expert who is specialized at curating metadata for images. You will "
                           "get raw unstructured metadata and are supposed to correctly identify and transcribe the "
                           "metadata to the ome xml standard. Only ever respond with the XML. Try to be as complete as "
                           "possible and only use structured annotations when absolutely necessary.")
        self.messages.append({"role": "system",
                              "content": self.pre_prompt})
        self.messages.append({"role": "user",
                              "content": self.raw_metadata})

    def predict_ome_xml(self):
        """
        Predict the OME XML from the raw metadata
        """
        prediction = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  # either gpt-3.5-turbo-16k or gpt-4, latter is way better but more expensive
            messages=self.messages,
            temperature=0,
            # How much the model deviates from the most likely answer ~creativity 0 = not creative, 1 = very creative
            max_tokens=5000,  # This is the amount of tokens including the task and handin text/initial messages etc.
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\"\"\""]
        )
        self.ome_xml = prediction["choices"][0]["message"]["content"]
        print(self.ome_xml)

    def getOMEXML(self):
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
