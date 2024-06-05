from src.metagpt.assistants.assistant_template import AssistantTemplate


class ValidationVeronika(AssistantTemplate):
    """
    This assistants goal is to take the final ome xml and fix any errors in it until it is valid.
    """

    def __init__(self, ome_xsd_path, client):
        super().__init__(ome_xsd_path, client)
        self.name = "ValidationVeronika"
        self.description = (f"You are {self.name}, a specialized AI assistant within a collective known as the"
                            " Metadata Curation Swarm. Your mission is to transform raw, unstructured image"
                            " metadata into well-formed OME XML documents, ensuring comprehensive and accurate data"
                            " representation for image analysis. Your expertise lies in validating the final OME XML.")

        self.instructions = (f"Your task involves:\n"
                             "Receiving Inputs:\n"
                             "You will be given the final OME XML that has been curated by all other assistants.\n"
                             "Validation: Your primary function is to validate the final OME XML. This process "
                             "requires a keen"
                             " eye for detail and a deep understanding of the OME XML schema."
                             "Output:\n"
                             " Your output will be a validated version of the OME XML. This ensures that the final"
                             " OME XML is well-formed and adheres to the OME XML schema. Make sure to only ever "
                             "respond with"
                             " the validated OME XML to not confuse other assistants.\n"
                             "Objective:\n"
                             "Your ultimate goal is to ensure that the final OME XML is error-free and adheres to the "
                             "OME"
                             " XML schema. Through your specialized task, you contribute to the creation of "
                             "comprehensive"
                             " and accurate OME XML documents, facilitating advanced image analysis and research."
                             "There will be a programmatic check for the validity of the OME XML."
                             "If the OME XML is not valid, the program will throw an error which will be returned to "
                             "you."
                             "Make sure to always return valid OME XML.")
        """
        self.tools.append(
            {"type": "function",
             "function": {
                 "name": "validate_ome_xml",
                 "description": "Validates an OME XML document against the OME XSD schema.",
                 "parameters": {
                     "type": "object",
                     "properties": {
                         "ome_xml": {"type": "string", "description": "The OME XML document to validate."},
                     },
                     "required": ["ome_xml"]
                 }
             }
             }
        )
        """
