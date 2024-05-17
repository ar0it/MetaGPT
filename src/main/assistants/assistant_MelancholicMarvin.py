from src.main.assistants.assistant_template import AssistantTemplate
from pydantic import BaseModel, Field
import time
from marvin.beta import Application
from marvin.beta.assistants import EndRun
import marvin
import ome_types


class MelancholicMarvin:

    def __init__(self):
        self.name = "OME_Predictor"
        self.prompt = (f"You are {self.name}, an AI-Assistant specialized in curating and predicting metadata for "
                       f"images. Your")
        f"task is"
        "to transform raw, unstructured metadata into well-formed XML, adhering to the OME XML"
        "standard. You have access to the OME XSD. Your responses"
        "should be exclusively in OME-XML format, aligning closely with the standard. Strive for"
        "completeness and validity. Rely on structured annotations only when necessary. DO NOT RESPONDE AT ALL ONLY UPDATE THE STATE."
        self.state = self.ResponseModel()
        self.assistant = Application(
            name=self.name,
            instruction=self.prompt,
            state=self.state,
        )

    def run_assistant(self, msg, thread=None):
        self.assistant.say(msg)
        print(self.state)
        ome_types.from_xml(self.state.ome_xml)

    class ResponseModel(BaseModel):
        """
        This class defines the
        """
        ome_xml: str = Field(None, description="The OME XML curated from the raw metdata.")
#%%
