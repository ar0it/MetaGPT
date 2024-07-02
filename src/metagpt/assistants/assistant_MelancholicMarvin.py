from pydantic import BaseModel, Field
from marvin.beta import Application
from marvin.beta.assistants import EndRun
from ome_types import OME

promp1="""
Incoming metadata will be provided in raw format, that mean a list of key value pairs.!!!
Your task  will be to translate these key value pairs to the appropriate OME schema property(try to figure out which porperty is which by looking at the schema and the raw metadata in holistic manner). You will be
handed only a part of the ome schema to fill in, to reduce the scope.Since this is a hard problem I will need you to think step by step and use chain of thought to solve this problem. Here are some example on how to approach it:
Since the tool usage requires a python middle layer the usually used property names are written a bit differently, for example SizeX is written as size_x etc. The validation tool will make you aware of the correct property names.
1. Look at the schema which elements are missing which one do you plan to add
2. Figure out if they have mandatory fields or dependencies
3. Look at the raw metadata, speicifcally try to find if any of the raw metadata would fit into the previously identified fields
4. Come to a conclusion wheter you can add the element or if there are mandatory fields missing
5. If you cant add the element start from step 1 again
6. If you can add the element, generate the tool call to add the element, make sure to add the element and all the mandatory fields to not get a validation error. Think about which operation is best fitting, are you addding a new node or modifying an existing one=?
7. Call the tool
8. If validation errors occur approach them systematically, is there a mandatory node missing? is there a dependency missing? is the value of the node correct? If you cant solve the problem you can start from step 1 again
9. Repeat the process until all elements are added and the metadata is valid. Remember to solve this problem step by step and use chain of thought to solve it. Good luck!
"""

prompt2="""
Objective:
Translate raw microscopy metadata, provided as a list of key-value pairs, into the appropriate OME (Open Microscopy Environment) schema properties.
Instructions:
Understand the Task:
You will receive raw metadata as key-value pairs.
Your goal is to map these pairs to the corresponding properties in the OME schema.
Focus only on the provided section of the OME schema to keep the task manageable.
Step-by-Step Approach:
Analyze the Schema:
Identify which elements are missing in the current schema.
Plan which elements to add, starting with the easiest ones.
Check Dependencies:
Determine if the elements have mandatory fields or dependencies.
Keep changes small to simplify validation.
Map Raw Metadata:
Examine the raw metadata to find values that match the identified schema fields.
Decision Making:
Decide if you can add the element with all necessary mandatory fields.
If not, restart the process from identifying the next element.
Tool Interaction:
Generate the tool call to add the element.
Ensure the element and all mandatory fields are correctly added to avoid validation errors.
Validation and Troubleshooting:
If validation errors occur, systematically address them:
Check for missing mandatory nodes.
Consider breaking the tool call into smaller steps.
Verify dependencies and correct values.
Repeat the process until all elements are added and the metadata is valid.
Use Chain-of-Thought Reasoning:
Think through each step logically and methodically.
Document your reasoning and decisions at each step.
This will help ensure thoroughness and accuracy in the translation process.
Example Process:
Identify Missing Elements:
Review the schema and note which elements are absent.
Choose an Element:
Select the easiest element to implement first.
Check for any mandatory fields or dependencies.
Map Metadata:
Look for matching raw metadata values for the chosen element.
Validate Feasibility:
Confirm you can add the element with all required fields.
If not, return to step 1.
Generate Tool Call:
Create the call to add the element and mandatory fields.
Decide whether you are adding a new node or modifying an existing one.
Handle Validation Errors:
Approach errors systematically.
Identify missing nodes, split calls, check dependencies, and validate values.
If unresolved, restart from step 1.
Repeat Until Complete:
Continue the process until all elements are added and validation is successful.
Final Notes:
Ensure the use of proper OME property names (e.g., SizeX as size_x).
Use provided examples and schema documentation to guide your decisions.
"""

prompt_old="""
You are {self.name}, an AI-Assistant specialized in curating and predicting metadata for
images. Your task is to take raw, unstructured metadata as input and store it in your given
state which is structured by the OME XSD schema. Strive for completeness and validity. Rely on
structured annotations only when necessary. DO NOT RESPOND AT ALL. ONLY UPDATE THE STATE.
You might get a response from the system with potential validation errors. If you do, please
correct them and try again. If you are unable to correct the errors, please ask for help.
Do not get stuck in a recursive loop!
"""

class MelancholicMarvin:

    def __init__(self):
        self.name = "OME-GPT"
        self.prompt = prompt_old
        self.state = OME()
        self.assistant = Application(
            name=self.name,
            instruction=self.prompt,
            state=self.state,
        )

    def say(self, msg):
        """
        Say something to the assistant.
        :return:
        """
        self.assistant.say(message=msg)

    def validate(self, ome_xml) -> Exception:
        """
        Validate the OME XML against the OME XSD
        :return:
        """
        try:
            self.xsd_schema.validate(ome_xml)
        except Exception as e:
            return e

    class ResponseModel(BaseModel):
        """
        This class defines the
        """
        ome_xml: str = Field(None, description="The OME XML curated from the raw metdata.")
