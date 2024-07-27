"""
This module contains the PredictorSeperator class, which is responsible for
separating raw metadata into structured annotations and OME properties.
"""

import ast
import time
from typing import Optional, Tuple, Dict

from pydantic import BaseModel, Field
from openai import OpenAI

from metagpt.predictors.predictor_template import PredictorTemplate
import metagpt.utils.utils as utils

class PredictorSeperator(PredictorTemplate):
    """
    A predictor class that separates raw metadata into structured annotations
    and OME properties using OpenAI's language model and vector embeddings.
    """
    
    def __init__(self, raw_meta: str) -> None:
        """
        Initialize the PredictorSeperator.

        Args:
            raw_meta (str): The raw metadata to be processed.
        """
        super().__init__()
        self.raw_metadata = raw_meta
        self.message = f"The raw data is: \n{self.raw_metadata}"
        self.sep_response = None
        self.file_paths = ["/home/aaron/Documents/Projects/MetaGPT/in/schema/ome_xsd.txt"]
        self.prompt = """
        You are part of a toolchain designed to predict metadata for the OME model, specifically the structured annotations part.
        You will be interacting with other toolchain components, therefore asking questions or providing any human-readable output is not necessary.
        Your task will be to take raw metadata in the form of key-value pairs and sort out the ones that do not have an appropriate place in the OME datamodel,
        but instead need to be added as structured annotations. For that purpose you have access to the OME schema via vectorized embeddings.
        Furthermore to improve the consistency of the output, you have access to the SepOutputTool which will structure the output key value pairs appropriately.
        ALWAYS USE THE TOOL TO PRODUCE RELIABLE OUTPUTS.
        The tool has two fields, annotation_properties and ome_properties. The annotation_properties are the properties that should be added as structured annotations.
        The ome_properties are the key value pairs that are represented in the OME Datamodel model.
        If you understood all of this, and will follow the instructions, answer with "." and wait for the first metadata to be provided.
        """

    def predict(self) -> Tuple[Optional[Tuple[Dict[str, str], Dict[str, str]]], float, int]:
        """
        Predict the separation of raw metadata into structured annotations and OME properties.

        Returns:
            Tuple[Optional[Tuple[Dict[str, str], Dict[str, str]]], float, int]:
                - A tuple containing two dictionaries (annotation_properties, ome_properties),
                  or None if prediction fails
                - The cost of the prediction
                - The number of attempts made
        """
        print(f"Predicting for {self.name}, attempt: {self.attempts}")
        if self.last_error is not None:
            self.message += self.last_error_msg
        self.init_thread()
        self.init_vector_store()
        self.init_assistant()   
        self.init_run()
        response = None

        try:
            self.add_attempts()
            self.sep_response = self.sep_run.required_action.submit_tool_outputs.tool_calls[0].function.arguments
            self.out_tokens += utils.num_tokens_from_string(str(self.sep_response))
            self.sep_response = ast.literal_eval(self.sep_response)
            if self.sep_response:
                sep_response_annot = self.sep_response["annotation_properties"]
                sep_response_ome = self.sep_response["ome_properties"]
                response = (sep_response_annot, sep_response_ome)
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
        if self.run_iter >= self.max_iter:
            return
        
        self.sep_run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.sep_assistant.id,
            tool_choice={"type": "function", "function": {"name": "SepOutputTool"}},
            temperature=self.temperature,
        )
        
        end_status = ["completed", "requires_action", "failed"]
        while self.sep_run.status not in end_status and self.run_iter < self.max_iter:
            self.run_iter += 1
            print(self.sep_run.status)
            time.sleep(5)
            self.sep_run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.sep_run.id
            )
        
        print(self.sep_run.status)

    def init_assistant(self) -> None:
        """Initialize the OpenAI assistant."""
        self.sep_assistant = self.client.beta.assistants.create(
            name="OME XML Seperator",
            description="An assistant to separate raw metadata into already contained and new metadata. Use the knowledge base of the OME XML schema to make the best decision.",
            instructions=self.prompt,
            model=self.model,
            tools=[{"type": "file_search"}, utils.openai_schema(self.SepOutputTool)],
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store.id]}}
        )
        self.assistants.append(self.sep_assistant)
    
    class SepOutputTool(BaseModel):
        """
        This tool automatically formats and structures the metadata in the appropriate way.
        """
        annotation_properties: Dict[str, str] = Field(
            default_factory=dict,
            description="A dictionary of properties which are to be put into the structured annotations."
        )
        ome_properties: Dict[str, str] = Field(
            default_factory=dict,
            description="A dictionary of properties which are already contained in the OME model."
        )