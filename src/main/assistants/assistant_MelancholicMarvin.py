from src.main.assistants.assistant_template import AssistantTemplate
from pydantic import BaseModel, Field
import time
from marvin.beta import Application
from marvin.beta.assistants import EndRun
import marvin
import ome_types
from src.main.OME_validator import OME_Validator


class MelancholicMarvin:

    def __init__(self, schema: str = None):
        self.name = "OME_Predictor"
        self.prompt = (f"You are {self.name}, an AI-Assistant specialized in curating and predicting metadata for "
                       f"images. Your"
                       f"task is"
                       "to transform raw, unstructured metadata into well-formed XML, adhering to the OME XML"
                       "standard. You have access to the OME XSD. Your responses"
                       "should be exclusively in OME-XML format, aligning closely with the standard. Strive for"
                       "completeness and validity. Rely on structured annotations only when necessary. DO NOT RESPONDE AT ALL"
                       "ONLY UPDATE THE STATE. You might get a response from the system with potential valdiation errors."
                       "If you do, please correct them and try again. If you are unable to correct the errors, please ask for help.")
        self.state = self.ResponseModel()
        self.xsd_schema = schema
        self.assistant = Application(
            name=self.name,
            instruction=self.prompt,
            state=self.state,
        )

    def run_assistant(self, msg, thread=None):
        not_valid = True
        while not_valid:
            self.assistant.say(msg)
            response = self.assistant.state
            print(response.ome_xml)
            validation_error = self.validate(response.ome_xml)
            if validation_error is None:
                not_valid = False
            else:
                msg = f"Validation error: {validation_error}. Please correct the error and try again."
        return self.assistant.state

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
