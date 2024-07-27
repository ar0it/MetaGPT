"""
This module contains the PredictorSimple class, which is responsible for
predicting well-formed OME XML from raw metadata using OpenAI's language model.
"""

import ast
import time
from typing import Optional, Tuple, Dict, Any

from pydantic import BaseModel, Field
from openai import OpenAI
from ome_types import from_xml

from metagpt.predictors.predictor_template import PredictorTemplate
import metagpt.utils.utils as utils

class PredictorSimple(PredictorTemplate):
    """
    A predictor class that generates well-formed OME XML from raw metadata
    using OpenAI's language model and vector embeddings.
    """

    def __init__(self, raw_meta: str) -> None:
        """
        Initialize the PredictorSimple.

        Args:
            raw_meta (str): The raw metadata to be processed.
        """
        super().__init__()
        self.raw_metadata = raw_meta
        self.message = f"The raw data is: \n{str(self.raw_metadata)}"
        self.file_paths = ["/home/aaron/Documents/Projects/MetaGPT/in/schema/ome_xsd.txt"]
        self.prompt = """
        You are a tool designed to predict metadata for the OME model.
        Your task will be to take raw metadata in the form of a dictionary of key-value pairs
        and put them into well-formed OME XML.
        The goal here is to understand the meaning of the key-value pairs and infer the correct OME properties.
        Importantly, always structure the metadata to follow the OME datamodel.
        Try to understand exactly how the metadata properties are related to each other and make sense of them.
        Try to figure out a good structure by looking at the raw metadata in a holistic manner.
        Furthermore, a file search function is provided with which you can look at the OME schema to make sure you produce valid metadata.
        Always use the tool to get reliable results.
        Since this is a hard problem, I will need you to think step by step and use chain of thought.
        Before you produce actual metadata, carefully analyze the problem, come up with a plan and structure.
        Here is the structure of how to approach the problem step by step:
        1. Look at the OME XML schema.
        2. Look at the raw metadata.
        3. Which properties from the key-value pairs map to which OME properties?
        4. Discard any properties that are not part of the OME schema.
        5. Generate the well-formed OME XML.
        Remember to solve this problem step by step and use chain of thought to solve it.
        You are not interacting with a human but are part of a larger pipeline, therefore you don't need to "chat".
        Under no circumstances can you ask questions.
        To return your response use the OMEXMLResponse function, which helps you to create a consistent output.
        The OMEXMLResponse function has an ome_xml field, which you should fill with the well-formed OME XML.
        ALWAYS USE THE FUNCTION TO PRODUCE RELIABLE OUTPUTS.
        You will have to decide on your own. ONLY EVER RESPOND WITH THE WELL-FORMED OME XML.
        Use the provided functions to solve the task. YOU NEED TO CALL THE FUNCTIONS TO SOLVE THE TASK. The chat response will be ignored.
        If you understood all of this, and will follow the instructions, answer with "." and wait for the metadata to be provided.
        """

    def predict(self) -> Tuple[Optional[str], float, int]:
        """
        Predict well-formed OME XML based on the raw metadata.

        Returns:
            Tuple[Optional[str], float, int]:
                - The predicted OME XML as a string, or None if prediction fails
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
            if self.run.status == 'completed':
                response = self.client.beta.threads.messages.list(thread_id=self.thread.id)
                self.out_tokens += utils.num_tokens_from_string(response.data[0].content[0].text.value)    
                response = response.data[0].content[0].text.value[7:][:-4]
            elif self.run.status == 'requires_action':
                function_call = self.run.required_action.submit_tool_outputs.tool_calls[0]
                self.out_tokens += utils.num_tokens_from_string(function_call.function.arguments)
                response_dict = ast.literal_eval(function_call.function.arguments)
                response = response_dict['ome_xml']

            test_ome_pred = from_xml(response)
        except Exception as e:
            self.last_response = response
            self.last_error = e
            print(f"There was an exception in the {self.name}: {e}")
            if self.attempts < self.max_attempts:
                print(f"Retrying {self.name}...")
                self.clean_assistants()   
                return self.predict()
            else:
                response = None
                print(f"Failed {self.name} after {self.attempts} attempts.")

        self.clean_assistants()        
        return response, self.get_cost(), self.attempts
    
    def init_run(self) -> None:
        """Initialize and monitor the run of the assistant."""
        print("Initiating run...")
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            tool_choice={"type": "function", "function": {"name": "OMEXMLResponse"}},
            temperature=self.temperature,
        )
        print("Waiting for run to complete...")
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
            name="OME XML Predictor",
            description="An assistant to predict well-formed OME XML from raw metadata",
            instructions=self.prompt,
            model=self.model,
            tools=[{"type": "file_search"}, utils.openai_schema(self.OMEXMLResponse)],
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store.id]}}
        )
        self.assistants.append(self.assistant)

    class OMEXMLResponse(BaseModel):
        """
        The response containing the well-formed OME XML.
        """
        ome_xml: Optional[str] = Field(None, description="The well-formed OME XML starting with <OME> and ending with </OME>")