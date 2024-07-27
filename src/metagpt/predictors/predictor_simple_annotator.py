"""
This module contains the PredictorSimpleAnnotation class, which is responsible for
predicting structured annotations for the OME model from raw metadata.
"""

import ast
import time
from typing import Dict, Optional, Tuple, Any

from pydantic import BaseModel, Field
from openai import OpenAI

from metagpt.predictors.predictor_template import PredictorTemplate
import metagpt.utils.utils as utils

class PredictorSimpleAnnotation(PredictorTemplate):
    """
    A predictor class that generates structured annotations for the OME model
    from raw metadata using OpenAI's language model.
    """

    def __init__(self, raw_meta: str) -> None:
        """
        Initialize the PredictorSimpleAnnotation.

        Args:
            raw_meta (str): The raw metadata to be processed.
        """
        super().__init__()
        self.raw_metadata = raw_meta
        self.message = f"The raw data is: \n{self.raw_metadata}"
        self.prompt = """
        You are part of a toolchain designed to predict metadata for the OME model, specifically the structured annotations part.
        You will be interacting with other toolchain components, therefore asking questions or providing any human-readable output is not necessary.
        Your task will be to take raw metadata in the form of a dictionary of key-value pairs and put them into the XMLAnnotation section of the OME model.
        Importantly, try to structure the metadata in a hierarchical manner, grouping related properties together.
        Try to understand exactly how the metadata properties are related to each other and make sense of them.
        Try to figure out a good structure by looking at the raw metadata in a holistic manner.
        Furthermore the function XMLAnnotationFunction is provided to help you structure the metadata.
        Always use the tool to get reliable results.
        Fill out the XMLAnnotationFunction object with the metadata you think is appropriate, in the appropriate structure.
        Since this is a hard problem, I will need you to think step by step and use chain of thought.
        Here is the structure of how to approach the problem step by step:
        1. Look at the raw metadata, which properties can be grouped together or are related?
        2. Figure out if you can group these properties in a hierarchical manner, do they cluster together?
        3. Come to a conclusion how you want to structure the metadata.
        4. Call the XMLAnnotationFunction with the annotations as JSON string.
        Remember to solve this problem step by step and use chain of thought to solve it.
        Again, you are not interacting with a human but are part of a chain of tools that are supposed to solve this problem.
        Under no circumstances can you ask questions.
        You will have to decide on your own. ONLY EVER RESPOND WITH THE JSON FUNCTION CALL.
        Use the provided function to solve the task. YOU NEED TO CALL THE FUNCTION TO SOLVE THE TASK. The chat response will be ignored.
        If you understood all of this, and will follow the instructions, answer with "." and wait for the first metadata to be provided.
        """

    def predict(self) -> Tuple[Optional[Dict[str, Any]], float, int]:
        """
        Predict structured annotations based on the raw metadata.

        Returns:
            Tuple[Optional[Dict[str, Any]], float, int]:
                - The predicted annotations as a dictionary, or None if prediction fails
                - The cost of the prediction
                - The number of attempts made
        """
        print(f"Predicting for {self.name}, attempt: {self.attempts}")
        if self.last_error is not None:
            self.message += self.last_error_msg
        self.init_thread()
        self.init_assistant()   
        self.init_run()
        response = None
        try:
            self.add_attempts(1)
            function_call = self.run.required_action.submit_tool_outputs.tool_calls[0]
            self.out_tokens += utils.num_tokens_from_string(function_call.function.arguments)
            response = ast.literal_eval(function_call.function.arguments)  # converts string to dict
            utils.merge_xml_annotation(annot=response)
        except Exception as e:
            print(f"There was an exception in the {self.name}: {e}")
            if self.attempts < self.max_attempts:
                print(f"Retrying {self.name}...")
                self.clean_assistants()
                return self.predict()
            else:
                print(f"Failed {self.name} after {self.attempts} attempts.")
            
        self.clean_assistants()
        return response, self.get_cost(), self.attempts
    
    def init_run(self) -> None:
        """Initialize and monitor the run of the assistant."""
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            tool_choice={"type": "function", "function": {"name": "XMLAnnotationFunction"}},
            temperature=self.temperature,
        )
        
        end_status = ["completed", "requires_action", "failed"]
        while self.run.status not in end_status and self.run_iter < self.max_iter:
            self.run_iter += 1
            print(self.run.status)
            time.sleep(5)
            self.run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id
            )
            
        print(self.run.status)
        
    def init_assistant(self) -> None:
        """Initialize the OpenAI assistant."""
        self.assistant = self.client.beta.assistants.create(
            name="OME XML Annotator",
            description="An assistant to predict OME XML annotations from raw metadata",
            instructions=self.prompt,
            model=self.model,
            tools=[utils.openai_schema(self.XMLAnnotationFunction)],
        )
        self.assistants.append(self.assistant)

    class XMLAnnotationFunction(BaseModel):
        """
        The function call to hand in the structured annotations to the OME XML.
        """
        annotations: Optional[Dict[str, Any]] = Field(
            None, 
            description="The structured annotations dictionary, with the metadata which is to be added to the OME XML. The dictionary shouldn't contain meta keys such as MetaData which are not explicitly in the raw metadata."
        )