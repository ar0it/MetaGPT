from pydantic import BaseModel, Field

from typing import Optional
from ome_types._autogenerated.ome_2016_06 import OME, StructuredAnnotations, XMLAnnotation
import instructor
from openai import OpenAI
import time
import sys
import importlib
import ast
from predictors.predictor_template import PredictorTemplate
import utils

class PredictorXMLAnnotation(PredictorTemplate):
    """
    TODO: Add docstring
    """
    def __init__(self, raw_meta: str) -> None:
        super().__init__()
        self.raw_metadata = raw_meta
        self.client = OpenAI()
        self.full_message = "The raw data is: \n" + str(self.raw_metadata)
        self.prompt = """
        You are part of a toolchain designed to predict metadata for the OME model, specifically the structured annotations part.
        You will be interacting with other toolchain components, therefore asking questions or providing any human-readable output is not necessary.
        You should concisely and structured document your thought process for logging purposes. mark those as "thoughts".
        The toolchain is a Tree structure of models, designed with the OME schema in mind.
        This means each node has a predictor such as yourself, and will be predicting the metadata for a submodel of the OME schema.
        One such submodel could be the  XMLAnnotation node.
        The tree structure is designed to predict the metadata in a bottom-up manner, starting from the lower level node.
        This way there are no missing dependencies when predicting the metadata.
        Incoming metadata will be provided in raw format, which means a list of key-value pairs.
        Your task will be to translate these key-value pairs to the appropriate OME schema property.
        Try to figure out which property is which by looking at the schema and the raw metadata in a holistic manner.
        The function call arguments you were given are the corresponding schema snippet you are supposed to fill out, remember only the highest level; the lower levels provided are only for validation and context.
        Since this is a hard problem, I will need you to think step by step and use chain of thought.
        Here is the structure of how to approach the problem step by step:
        1. Look at the raw metadata, which properties can not be served by the ome data model and therefore should be added as structured annotations?
        2. Figure out if you can groupt these properties in a hierarchical manner, do they cluster together?
        3. Come to a conclusion how you want to structure the metadata.
        4. Call the XMLAnnotationFunction function with the structured annotations as xml Json.
        It is very well possible some fields remain empty.
        Remember to solve this problem step by step and use chain of thought to solve it.
        Again, you are not interacting with a human but are part of a chain of tools that are supposed to solve this problem.
        Under no circumstances can you ask questions.
        You will have to decide on your own. ONLY EVER RESPOND WITH THE JSON FUNCTION CALL.
        If you understood all of this, and will follow the instructions, answer with "." and wait for the first metadata to be provided.
        Use the provided function to solve the task. YOU NEED TO CALL THE FUNCTION TO SOLVE THE TASK. The chat response will be ignored.
        """
        self.pred_prompt = """
        You are part of a toolchain designed to predict metadata for the OME model, specifically the structured annotations part.
        You will be interacting with other toolchain components, therefore asking questions or providing any human-readable output is not necessary.
        Your task will be to take raw metadata in the form of a dictionary of key-value pairs and put them into the XMLAnnotation section of the OME model.
        Importantly, try to structure the metadata in a hierarchical manner, grouping related properties together.
        Try to understnad exactly how the metadata properties are related to each other and make sense of them.
        Try to figure out a good structure by looking at the raw metadata in a holistic manner.
        Furthermore the function StructuredAnnotations is provided to help you structure the metadata.
        Always use the tool to get reliable results.
        Fill out the StructuredAnnotations object with the metadata you think is appropriate, in the appopriate structure.
        Since this is a hard problem, I will need you to think step by step and use chain of thought.
        Here is the structure of how to approach the problem step by step:
        1. Look at the raw metadata, which properties can be grouped together or are related?
        2. Figure out if you can groupt these properties in a hierarchical manner, do they cluster together?
        3. Come to a conclusion how you want to structure the metadata.
        4. Call the StructuredAnnotations function with the annotations as JSON string.
        Remember to solve this problem step by step and use chain of thought to solve it.
        Again, you are not interacting with a human but are part of a chain of tools that are supposed to solve this problem.
        Under no circumstances can you ask questions.
        You will have to decide on your own. ONLY EVER RESPOND WITH THE JSON FUNCTION CALL.
        Use the provided function to solve the task. YOU NEED TO CALL THE FUNCTION TO SOLVE THE TASK. The chat response will be ignored.
        If you understood all of this, and will follow the instructions, answer with "." and wait for the first metadata to be provided.
        """

    
    def predict(self) -> dict:
        """
        TODO: Add docstring
        """
        self.init_thread()
        self.init_assistant()   
        self.init_run()
        response = self.run.required_action.submit_tool_outputs.tool_calls[0]
        response = ast.literal_eval(response.function.arguments) # converts string to dict
        cost = self.get_cost(run=self.run)

        self.clean_assistants()
        return response, cost
    
    def init_run(self):
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            tool_choice={"type": "function", "function": {"name": "XMLAnnotationFunction"}},
            temperature=0.0,
            )
        
        end_status = ["complete", "requires_action", "failed"]
        while self.run.status not in end_status:
            print(self.run.status)
            time.sleep(5)
            self.run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id
                )
            
        print(self.run.status)
        
    def init_assistant(self):
        self.assistant = self.client.beta.assistants.create(
            name="OME XML Annotator",
            description="An assistant to predict OME XML annotations from raw metadata",
            instructions=self.pred_prompt,
            model="gpt-4o",
            tools=[utils.openai_schema(self.XMLAnnotationFunction)],

        )
        self.assistants.append(self.assistant)

    def init_thread(self):
        self.thread = self.client.beta.threads.create(messages=[{"role": "user", "content": self.pred_prompt},
                                                                {"role": "assistant", "content": "."},
                                                                {"role": "user", "content": self.full_message}])
        self.threads.append(self.thread)

    class XMLAnnotationFunction(BaseModel):
        """
        The function call to hand in the structured annotations to the OME XML.
        """
        annotations: Optional[dict] = Field(None, description="The structured annotations dictionary, with the metadata which is to be added to the OME XML. The dictionary shouldnt contain meta keys such as MetaData which are not explicitly in the raw metadat.")
