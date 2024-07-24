from pydantic import BaseModel, Field
from typing import Optional
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
from pydantic import BaseModel, Field, RootModel
from typing import List, Literal, Any, Union
import json


class PredictorState(PredictorTemplate):
    """
    TODO: Add docstring
    """
    def __init__(self, raw_meta: str, state:BaseModel=None) -> None:
        super().__init__()
        if state is None:
            state = OME()
        self.state = state
        self.json_patches = "No Patch has been generated yet."
        self.last_error = "No error was thrown yet."
        self.raw_metadata = raw_meta
        self.query = f"""
        The raw data is: \n" {str(self.raw_metadata)} \n please predict the metaddata for the {type(self.state)} object.
        In a previous attempt the following json patches were generated: {self.json_patches} and the following error was
        returned: {self.last_error}. Try to correct the mistake and generate a valid json patch.
        """
        self.current_state_documentation = f"""
        Here is the schema to the respective state object, this is what you are supposed to predict:
        {utils.browse_schema(self.state.__class__, max_depth=1)}
        """
        self.current_state_message = "The current state is: \n" + self.state.model_dump_json() + "You can think of this as your starting point from which you are supposed to expand."

        self.json_patch_examples = """

        # Example State, created for you with the ome_types python library
        pixel = Pixels(
                    size_x=1024,
                    size_y=768,
                    size_t=1,
                    size_c=1,
                    size_z=1, 
                    dimension_order="XYZCT",
                    type="uint8",
                    channels=[{"id": "Channel:0", "name": "Red"},
                                {"id": "Channel:1", "name": "Green"},
                                {"id": "Channel:3", "name": "Blue"}])

        image = Image(name="Test Image", pixels=pixel)
        initial_state = OME(images=[image])

        # Example 1: Add operation
        patch1 = [
            {"op": "add", "path": "/images/0/description", "value": "A new description"}
        ]

        # Example 2: Remove operation
        patch2 = [
            {"op": "remove", "path": "/images/0/pixels/channels/2"}
        ]

        # Example 3: Replace operation
        patch3 = [
            {"op": "replace", "path": "/images/0/name", "value": "Updated Image Name"}
        ]
        
        # Example 4: Multiple operations in one patch
        patch4 = [
            {"op": "replace", "path": "/images/0/pixels/size_x", "value": 2048},
            {"op": "replace", "path": "/images/0/pixels/size_y", "value": 1536},
            {"op": "add", "path": "/images/0/pixels/channels/-", "value": {"id": "Channel:3", "name": "Alpha"}}
        ]

        # Example 5: Required and Nested actions
        patch5 = [
        {
            "op": "add",
            "path": "/images/-",
            "value": {
            "id": "Image:0",
            "name": "New Image",
            "acquisition_date": "2024-07-21T12:00:00",
            "pixels": {
                "id": "Pixels:0",
                "type": "uint16",
                "dimension_order": "XYZCT",
                "size_x": 1024,
                "size_y": 1024,
                "size_z": 1,
                "size_c": 3,
                "size_t": 1,
                "physical_size_x": 0.1,
                "physical_size_x_unit": "µm",
                "physical_size_y": 0.1,
                "physical_size_y_unit": "µm",
                "physical_size_z": 1.0,
                "physical_size_z_unit": "µm",
                "channels": [
                {
                    "id": "Channel:0",
                    "name": "Red",
                    "color": -16776961
                },
                {
                    "id": "Channel:1",
                    "name": "Green",
                    "color": 16711935
                },
                {
                    "id": "Channel:2",
                    "name": "Blue",
                    "color": 65535
                }
                ]
            }
            }
        }
        ]
        """
        self.ome_types_doc = """
        # Usage

        `ome_types` is useful for parsing the [OME-XML
        format](https://docs.openmicroscopy.org/ome-model/latest/ome-xml/) into
        Python objects for interactive or programmatic access in python. It can
        also take these Python objects and turn them back into OME-XML.

        For example, you can parse an ome.xml, and then explore it with pythonic
        `camel_case` syntax and readable object representations:

        ## Reading

        ``` python
        In [1]: from ome_types import from_xml

        In [2]: ome = from_xml('tests/data/hcs.ome.xml')  # or ome_types.OME.from_xml()
        ```

        [ome_types.from_xml][] returns an instance of [ome_types.OME][].
        This object is a container for all information objects accessible by OME.

        ``` python
        In [3]: ome
        Out[3]: 
        OME(
            images=[<1 Images>],
            plates=[<1 Plates>],
        )

        In [4]: ome.plates[0]
        Out[4]: 
        Plate(
            id='Plate:1',
            name='Control Plate',
            column_naming_convention='letter',
            columns=12,
            row_naming_convention='number',
            rows=8,
            wells=[<1 Wells>],
        )

        In [5]: ome.plates[0].wells[0]
        Out[5]: 
        Well(
            id='Well:1',
            column=0,
            row=0,
            well_samples=[<1 Well_Samples>],
        )

        In [6]: ome.images[0]
        Out[6]: 
        Image(
            id='Image:0',
            name='Series 1',
            pixels=Pixels(
                id='Pixels:0',
                dimension_order='XYCZT',
                size_c=3,
                size_t=16,
                size_x=1024,
                size_y=1024,
                size_z=1,
                type='uint16',
                bin_data=[<1 Bin_Data>],
                channels=[<3 Channels>],
                physical_size_x=0.207,
                physical_size_y=0.207,
                time_increment=120.1302,
            ),
            acquisition_date=datetime.fromisoformat('2008-02-06T13:43:19'),
            description='An example OME compliant file, based on Olympus.oib',
        )

        In [7]: ome.images[0].pixels.channels[0]
        Out[7]: 
        Channel(
            id='Channel:0:0',
            name='CH1',
            acquisition_mode='LaserScanningConfocalMicroscopy',
            emission_wavelength=523.0,
            excitation_wavelength=488.0,
            illumination_type='Epifluorescence',
            pinhole_size=103.5,
            samples_per_pixel=1,
        )

        In [8]: ome.images[0].pixels.channels[0].emission_wavelength                                                                               
        Out[8]: 523.0
        ```

        ## Modifying or Creating

        The `OME` object is mutable, and you may make changes:

        ``` python
        In [9]: from ome_types.model import UnitsLength

        In [10]: from ome_types.model.channel import AcquisitionMode

        In [11]: ome.images[0].description = "This is the new description."

        In [12]: ome.images[0].pixels.physical_size_x = 350.0

        In [13]: ome.images[0].pixels.physical_size_x_unit = UnitsLength.NANOMETER

        In [14]: for c in ome.images[0].pixels.channels:
                    c.acquisition_mode = AcquisitionMode.SPINNING_DISK_CONFOCAL
        ```

        And add elements by constructing new `OME` model objects:

        ``` python
        In [15]: from ome_types.model import Instrument, Microscope, Objective, InstrumentRef

        In [16]: microscope_mk4 = Microscope(
                    manufacturer='OME Instruments',
                    model='Lab Mk4',
                    serial_number='L4-5678',
                )

        In [17]: objective_40x = Objective(
                    manufacturer='OME Objectives',
                    model='40xAir',
                    nominal_magnification=40.0,
                )

        In [18]: instrument = Instrument(
                    microscope=microscope_mk4,
                    objectives=[objective_40x],
                )

        In [19]: ome.instruments.append(instrument)

        In [20]: ome.images[0].instrument_ref = InstrumentRef(instrument.id)

        In [21]: ome.instruments
        Out[21]:
        [Instrument(
            id='Instrument:1',
            microscope=Microscope(
            manufacturer='OME Instruments',
            model='Lab Mk4',
            serial_number='L4-5678',
            ),
            objectives=[<1 Objectives>],
        )]
        ```
        """
        self.prompt = f"""
        To solve the task of genrating json patches from raw unstructured metadata,
        you are given several tools etc.
        First of you are updating a persistent state object, this helps you to
        update the state step by step. To do so you are provided the update_json_state
        function, which takes a list of json patches as input and updates the state object.
        Importantly you are not always given the OME state object, but only a part of it.
        This means you need to generate the json patches for the provided part of the schema.
        The paths in the patches are therefore relative to the provided state object.
        Make sure you dont add the full path to the state object, but only the path to the provided part of the schema.
        The json patches are structured as follows:
        {self.json_patch_examples}
        The ome_types OME object is structured slightly different from the default xml schema.
        Namely the properties are written lower case with underscores instead of camel case.
        Further Elements nodes such as image, pixels, channel etc. are collected as list of objects.
        Such as images, pixels, channels etc. in practise the element nodes therefore look as follows:
        images=[image1, image2, ...] furthermore properties such as the id and the top lvl attributes of the OME object such as Namespace, SchemaLocation etc.
        are genetated automatically by the ome_types library. DO NOT ADD OR CHANGE PROPERTIES SUCH AS THE ID. THESE ARE GENERATED AUTOMATICALLY.
        You need to remember this when generating the json patches.
        You can access specific elements in the respective lists by calling its index in the path.
        For example:
        /images/0/pixels/size_x
        would access the size_x property of the pixels object of the first image in the images list.
        To give you some additional information on how the ome_types library works, here is a short documentation:
        {self.ome_types_doc}
        Since the task is quite difficult I need you to work step by step and use chain of thought.
        Furthermore the task might be provided to you in multiple steps, which means you are only provided part of the schema (such as the subnode pixels).
        In that case you only need to generate the json patches for the provided part of the schema.
        Because of this pay extra attention to the state provided, and only generate the json patches for the provided part of the schema.
        For example if you are only provided the pixels state a patch to create the pixel could look like this:
        {'{}'}
        Here is a structure of how to approach the problem step by step:
        1. look at the schema, what propeties are there, which properties are required to generate the property in question?
        2. Look at the raw metadata, which properties could be related to the property in question?
        3. If the needed properties are not in the raw metadata, skip the property and exit without generating a patch.
        4. If the needed properties are the in the raw metadata, generate a minimal patch, that creates the property in question.
        Dont try to geneate a patch that inclued all properties at once, but work step by step.
        This way the automatic validation tool can give you feedback on each step.
        5. After each successful patch, the state will be updated and the result whill be provided to you for reference.
        6. If the patch was not successful, the state will not be updated and you will be asked to retry.
        You will also be provided with the error mesaage that was generated.
        Carefully study the error message and try to understand what went wrong, before attempting to retry.
        If you understood all of this, and will follow the instructions, answer with "." and wait for the first metadata to be provided.
        """
        self.description = """
        You are OMEGPT, a toold designed to predict metadata for the OME model.
        More specifically you are suppose to generate json patches that will be
        used to update the ome_types OME data object.
        To solve the task you are given a set of key value pairs that represent
        raw unstructured metadat, which does not yet follow the ome schema.
        Your task is to figure out what the metadata could mean in the context
        of the OME schema.
        """
        ome_types_schema_path = "/home/aaron/Documents/Projects/MetaGPT/in/schema/ome_types_schema.json"
        with open(ome_types_schema_path, "w") as f:
            json.dump(OME.model_json_schema(), f)

        self.file_paths = [ome_types_schema_path]

    def predict(self, indent:Optional[int]=0) -> str:
        """
        TODO: Add docstring
        """
        print(indent*"  "+f"Predicting for {self.name}, attempt: {self.attempts}")
        print(self.state.model_dump_json())
        print(type(self.state))
        print(self.current_state_documentation)
        self.message = self.current_state_message + self.current_state_documentation + self.query
        self.init_thread()
        self.init_vector_store()
        self.init_assistant()   
        self.init_run()
        response = None

        try:
            self.add_attempts()
            response = self.run.required_action.submit_tool_outputs.tool_calls[0]
            self.out_tokens += utils.num_tokens_from_string(response.function.arguments)
            response = ast.literal_eval(response.function.arguments)
            response = response['json_patches']
            self.json_patches = response
            for patch in response:
                print("patch:", patch)
                self.state = utils.update_state(self.state, [patch])
            if self.state.__class__.__name__.startswith("Maybe"):
                self.state = getattr(self.state, self.model.__name__)
            response = to_xml(self.state)  

        except Exception as e:
            print(f"There was an exception in the {self.name}" ,e)
            print(response)
            self.last_error = e
            if self.attempts < self.max_attempts:
                print(f"Retrying {self.name}...")
                self.clean_assistants()
                   
                return self.predict()
            else:
                response = None
                print(f"Failed {self.name} after {self.attempts} attempts.")
        
        self.clean_assistants()
        return response, self.get_cost(), self.attempts

    def init_run(self):
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            tool_choice={"type": "file_search", "type": "function", "function": {"name": "update_json_state"}},
            temperature=self.temperature,
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
            name="OMEGPT",
            description=self.description,
            instructions=self.promtp,
            model=self.model,
            tools=[{"type": "file_search"},
                   utils.openai_schema(update_json_state)],
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store.id]}}
        )
        self.assistants.append(self.assistant)


class AddReplaceTestOperation(BaseModel):
    op: Literal["add", "replace", "test"]
    path: str = Field(..., description="A JSON Pointer path.")
    value: Any = Field(..., description="The value to add, replace or test.")

class RemoveOperation(BaseModel):
    op: Literal["remove"]
    path: str = Field(..., description="A JSON Pointer path.")

class MoveCopyOperation(BaseModel):
    op: Literal["move", "copy"]
    path: str = Field(..., description="A JSON Pointer path.")
    from_: str = Field(..., alias="from", description="A JSON Pointer path pointing to the location to move/copy from.")

class JsonPatch(BaseModel):
    root: List[Union[AddReplaceTestOperation, RemoveOperation, MoveCopyOperation]]

    class Config:
        title = "JSON schema for JSONPatch files"
        json_schema_extra = {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "id": "https://json.schemastore.org/json-patch.json",
        }

class update_json_state(BaseModel):
    """
    Update the state of the predictor from a list of json patches.
    """
    json_patches: Optional[list[JsonPatch]] = Field(default_factory=list[JsonPatch], description="")
    #no_properties: Optional[bool] = Field(default=False,
                                          #description="Only fill out if no fitting properties were found in the raw metadata, in this case set this to True. Remember to write upercase True or False as this is a boolean.")
