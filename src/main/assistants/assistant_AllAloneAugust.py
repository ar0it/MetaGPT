from src.main.assistants.assistant_template import AssistantTemplate


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
                           "standard. You have access to the OME XSD for reference. Furthermore you will be provided"
                           "with an OME-XML starting points generated by Bioformats. The starting point might contain"
                           "some of the information from the raw data but could also be incomplete. Your responses"
                           "should be exclusively in XML format, aligning closely with the standard. Strive for"
                           "completeness and validity. Rely on structured annotations only when necessary.")