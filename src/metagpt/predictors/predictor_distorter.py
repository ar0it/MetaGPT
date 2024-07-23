from pydantic import BaseModel, Field
from typing import Optional, Any
from ome_types._autogenerated.ome_2016_06 import OME, StructuredAnnotations, XMLAnnotation
import instructor
from openai import OpenAI
import time
import sys
import importlib
import ast
from ome_types import from_xml, to_xml
from metagpt.predictors.predictor_template import PredictorTemplate
import metagpt.utils.utils as utils

class PredictorDistorter(PredictorTemplate):
    """
    TODO: Add docstring
    """
    def __init__(self, raw_meta: str) -> None:
        super().__init__()
        self.raw_metadata = raw_meta
        self.full_message = "The paths to translate into another syntax are:\n" + str(self.raw_metadata)
        self.prompt = """
        You are a tool designed to invent new metadata syntax for microscopy images.
        Explictily your task will look as follows:
        Input: A list or dictionary of key value pairs, which correspond to the original metadata.
        Take this metadata and translate ONLY THE KEYS into a new syntax.
        NEVER CHANGE THE VALUES.
        Use the same translation if the same key comes up multiple times, i.e. be consistent.
        Use the same seperator operator "/" as in the original metadata.
        The operator indicates the hierarchy in the metadata.
        The goal is to create a new metadata syntax that is not the same as the original.
        The output should be a dictionary of key value pairs,with the mapping from old keys to new keys.
        Here are some example to make things clear:
        Input: {"key1": "value1", "key2": "value2"}
        Output: {"key1": "new_key1", "key2": "new_key2"}
        This is a hard task therefore I need you to work step by step and use chain of thought.
        I have provided you with the function out_new_meta that will help you with outputting the new metadata.
        You are required to use this function.
        If you understood all this and will follow the instructions, answer with "." and wait for the first metadata to be provided.
        """

    
    def predict(self) -> dict:
        """
        TODO: Add docstring
        """
        print(f"Predicting for {self.name}, attempt: {self.attempts}")
        self.init_thread()
        self.init_assistant()   
        self.init_run()
        response, cost = None, None
        try:
            self.add_attempts()
            response = self.run.required_action.submit_tool_outputs.tool_calls[0]
            response = ast.literal_eval(response.function.arguments)

        except Exception as e:
            response = None
            print(f"There was an exception in the {self.name}" ,e)
            if self.attempts < self.max_attempts:
                print(f"Retrying {self.name}...")
                self.clean_assistants()   
                return self.predict()
            else:
                print(f"Failed {self.name} after {self.attempts} attempts.")
            

        self.clean_assistants()        
        return response
    
    def init_run(self):
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            tool_choice={"type": "function", "function": {"name": "out_new_meta"}},
            temperature=0.0,
            )
        
        end_status = ["completed", "requires_action", "failed"]
        while self.run.status not in end_status and self.run_iter<self.max_iter:
            self.run_iter += 1
            print(self.run.status)
            time.sleep(5)
            self.run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id
                )
            
        print(self.run.status)
        
    def init_assistant(self):
        self.assistant = self.client.beta.assistants.create(
            name="MetadataInventor",
            description="An assistant to translate metadata in to a made up syntax",
            instructions=self.prompt,
            model="gpt-4o-mini",
            tools=[{"type": "file_search"}, utils.openai_schema(self.out_new_meta)]
        )
        self.assistants.append(self.assistant)

    def init_thread(self):
        self.thread = self.client.beta.threads.create(messages=[{"role": "user", "content": self.prompt},
                                                                {"role": "assistant", "content": "."},
                                                                {"role": "user", "content": self.full_message}])
        self.threads.append(self.thread)


    class out_new_meta(BaseModel):
        """
        Helper class to define the output of the assistant.
        """
        new_meta: dict[str, Any] = Field(
            default_factory=dict,
            description="A dictionary of key value pairs, with the keys corresponding to the old keys and the values to the new keys.")
