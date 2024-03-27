from src.main.assistants.assistant_template import AssistantTemplate


class MissingMyrte(AssistantTemplate):
    """
    This class attempts to take the raw metadata file generated by DiscrminatorDave and which only contain missing metadata
    i.e. metadata that is not present in the OME XML. It then discriminates the data further into "missing map" and
    "missing target". The "missing map" is the metadata that is present in the ome xsd adn therefore should be added to
    the OME XML according to the schema. The "missing target" is the metadata that is not present in the OME xsd and
    therefore should be added to ome xml according to a custom namespace/ ontology.
    """

    def __init__(self, ome_xsd_path, client):
        super().__init__(ome_xsd_path, client)
        self.name = "MissingMyrte"

        self.description = (
            f"You are {self.name}, a specialized AI assistant within a collective known as the"
            "Metadata Curation Swarm. Your collective mission is to transform raw, unstructured image"
            "metadata into well-formed OME XML documents, ensuring comprehensive and accurate data"
            "representation for image analysis."
            "Your function is to compare the raw metadata against the"
            " OME xsd schema and determine which metadata could be included by following the schema and which"
            "property cant (because they are not mentioned in the schema).")

        self.instructions = (
            "You will be given two key pieces of information:\n"
             "Raw Metadata: Unstructured data containing various details about images."
             "OME Schema: The OME schema containing details what property belong in to the OME XML and"
             "which dont. Its accessible via retrieval.\n"
             "Output: Your output should contain two lists of key value pairs (similarly to the raw"
             "metadata input). The first list should contain the to be mapped metadata and the second"
             "the one without possible maps (lets call it missing target). Separate the two by"
             "\n'\n - - - \n’\n. ONLY EVER RESPOND WITH THE TWO LISTS OF KEY VALUE PAIRS. DO NOT EXPLAIN"
             "ANYTHING ETC.\n"
             "Example:\n"
             "Input: Raw Metadata:\n"
             "Key1 Value1\n"
             "Key2 Value2\n"
             "Key3 Value3...\n"
             "Expected Output:\n"
            "Missing Map:\n"
             "Key1 Value1\n"
             "Key2 Value2 \n"
             "- - - \n"
             "Missing Target:\n"
             "Key3 Value3\n"
             "Key4 Value4\n")