from metagpt.predictors.predictor_template import PredictorTemplate
from marvin.beta.applications import Application
from ome_types import OME


class PredictorMarvin(PredictorTemplate):
    """
    TODO: Add docstring
    """
    def __init__(self, raw_meta:str) -> None:
        super().__init__()
        self.raw_metadata = raw_meta
        self.prompt = f"""
        Here is the raw metadata \n
        {raw_meta}
        """
        self.instructions = """
        You are a tool designed to predict metadata for the OME
        model.
        Your task will be to take raw metadata in the form of a dictionary of key-value pairs
        and put them into well formed OME XML.
        The goal here is to understand the meaning of the key value pairs and infer the correct ome properties.
        Importantly, always structure the metadata to follow the ome datamodel.
        Try to understnad exactly how the metadata properties are
        related to each other and make sense of them.
        Try to figure out a good structure by looking at the raw metadata in a
        holistic manner.
        Furthermore a file search function is provided with which you can 
        look at the ome schema to make sure you produce valid metadata.
        Always use the tool to get reliable results.
        Since this is a hard problem, I will need you to think step by step and use chain of thought.
        Before you produce actual metadata, carefully analyze the probmlem, come up with a plan and structure.
        Here is the structure of how to approach the problem step by step:
        1. Look at the ome xml schema.
        2. Look at the raw metadata.
        3. Which proerties from the key value pairs map to which ome properties?
        4. Discard any properties that are not part of the ome schema.
        5. Generate the well formed ome xml
        Remember to solve this problem step by step and use chain of thought to solve it.
        You are not interacting with a human but are part of a larger pipeline, therefore you dont need to "chat".
        Under no circumstances can you ask questions.
        To return your response use the OMEXMLResponse function, which helps you to create a consistent output.
        The OMEXMLResponse function has a ome_xml field, which you should fill with the well formed OME XML.
        ALWAYS USE THE FUNCTION TO PRODUCE RELIABLE OUTPUTS.
        You will have to decide on your own. ONLY EVER RESPOND WITH THE WELL FORMED OME XML.
        Use the provided functions to solve the task. YOU NEED TO CALL THE FUNCTIONs TO SOLVE THE TASK. The chat response will be ignored.
        If you understood all of this, and will follow the instructions, answer with "." and wait for the metadata to be provided.
        """

        self.state = None
    def predict(self):
        """
        Predict the image annotations using the Marvin model.
        :param image_path: The path to the image.
        :return: The predicted image annotations.
        """
        self.init_state()
        app = Application(
            state=self.state,
            name=self.__class__.__name__,
            instructions=self.instructions,
            )
        
        app.say(self.prompt)

        return app.state
    
    def init_state(self):
        """
        Initialize the state
        """
        self.state = OME()