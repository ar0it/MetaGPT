from pydantic import BaseModel, Field
from marvin.beta import Application
from marvin.beta.assistants import EndRun
from ome_types import OME

class MelancholicMarvin:

    def __init__(self):
        self.name = "OME-GPT"
        self.prompt = (f"You are {self.name}, an AI-Assistant specialized in curating and predicting metadata for "
                   f"images. Your task is to take raw, unstructured metadata as input and store it in your given "
                   f"state which is structured by the OME XSD schema. Strive for completeness and validity. Rely on "
                   f"structured annotations only when necessary. DO NOT RESPOND AT ALL. ONLY UPDATE THE STATE. "
                   f"You might get a response from the system with potential validation errors. If you do, please "
                   f"correct them and try again. If you are unable to correct the errors, please ask for help. "
                   f"Do not get stuck in a recursive loop!")
        
        self.state = OME()
        self.assistant = Application(
            name=self.name,
            instruction=self.prompt,
            state=self.state,
        )

    def say(self, msg):
        """
        Say something to the assistant.
        :return:
        """
        self.assistant.say(message=msg, max_completion_tokens=100000)

    def validate(self, ome_xml) -> Exception:
        """
        Validate the OME XML against the OME XSD
        :return:
        """
        try:
            self.xsd_schema.validate(ome_xml)
        except Exception as e:
            return e

    class ResponseModel(BaseModel):
        """
        This class defines the
        """
        ome_xml: str = Field(None, description="The OME XML curated from the raw metdata.")
