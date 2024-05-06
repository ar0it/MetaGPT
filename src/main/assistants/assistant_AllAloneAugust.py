from src.main.assistants.assistant_template import AssistantTemplate
from pydantic import BaseModel, Field


class AllAloneAugust(AssistantTemplate):
    """
    This assistant is the most basic assistant approach and attempts to translate the raw meta-data all alone :(.
    """

    def __init__(self, ome_xsd_path, client):
        super().__init__(ome_xsd_path, client)
        self.name = "AllAloneAugust"
        self.pre_prompt = (f"You are {self.name}, an AI assistant specialized in curating metadata for images. Your "
                           f"task is"
                           "to transform raw, unstructured metadata into well-formed XML, adhering to the OME XML"
                           "standard. You have access to the OME XSD for reference via retrieval. Your responses"
                           "should be exclusively in XML format, aligning closely with the standard. Use the provided"
                           "function for better reliability. Strive for"
                           "completeness and validity. Rely on structured annotations only when necessary.")
        self.response_model = self.AugustResponseModel
        self.assistant = self.create_assistant()

    class AugustResponseModel(BaseModel):
        ome_xml: str = Field(description="The valid OME XML generated from the raw metadata.")
