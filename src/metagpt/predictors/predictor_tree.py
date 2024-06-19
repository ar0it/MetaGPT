from typing import List, Dict, Set, Type, Optional
from pydantic import BaseModel, Field
from marvin.beta import Application
from collections.abc import Iterator
from ome_types import OME
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("metagpt", "/home/aaron/Documents/Projects/MetaGPT/src/metagpt/BioformatsReader.py")
BioformatsReader = importlib.util.module_from_spec(spec)
sys.modules["metagpt"] = BioformatsReader
spec.loader.exec_module(BioformatsReader)

import javabridge
import bioformats
import ome_types
spec = importlib.util.spec_from_file_location("metagpt", "/home/aaron/Documents/Projects/MetaGPT/src/metagpt/utils.py")
utils = importlib.util.module_from_spec(spec)
sys.modules["metagpt"] = utils
spec.loader.exec_module(utils)


javabridge.start_vm(class_path=bioformats.JARS)
utils._init_logger()
image_path = "/home/aaron/Documents/Projects/MetaGPT/in/images/testetst_Image8_edited_.ome.tif"
ome_xml = BioformatsReader.get_omexml_metadata(image_path)
ome_raw = BioformatsReader.get_raw_metadata(image_path)
ome_tree = BioformatsReader.raw_to_tree(ome_raw)
ome_dict = ome_types.to_dict(ome_xml)

javabridge.kill_vm()

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
I have split the ome model into several submodels, so if you feel like therer is not metadata for the data model you are working on, no worries just skip that model. Its very well possible some fields remain empty.
"""

# Define the tree node class
class TreeNode:
    def __init__(self, model: Type[BaseModel]):
        self.model = model
        self.object = None
        self.children = []

    def add_child(self, child: 'TreeNode'):
        self.children.append(child)

    def __repr__(self):
        return f"TreeNode(model={self.model.__name__}, children={self.children})"
    
    
    def required_fields(self, model: type[BaseModel], recursive: bool = False) -> Iterator[str]:
        """
        https://stackoverflow.com/questions/75146792/get-all-required-fields-of-a-nested-pydantic-model
        """
        for name, field in model.model_fields.items():
            if not field.is_required():
                continue
            t = field.annotation
            if recursive and isinstance(t, type) and issubclass(t, BaseModel):
                yield from self.required_fields(t, recursive=True)
            else:
                yield name

    
    def predict_meta(self, raw_meta) -> BaseModel:
        # add the metdata from the child nodes first
        child_objects = {}
        for child in self.children:
            child_objects[child.model.__name__] = child.predict_meta(raw_meta)
        
        self.object = self.instantiate_model(child_objects)
        # TODO: loop here in case the field allows for the same type multiple times (i.e OME(images=[Image, Image, Image])) Maybe loop until AI doesnt predict any new
        # In that case I need to remove metadata from the raw_meta that has already been used
        print(f"Predicting metadata for {self.model.__name__}, self.object={type(self.object)}, required={list(self.required_fields(self.model))}")

        return self.object
        app = Application(
            name='OME Metadata Store',
            instructions=(promp1),
            state=self.object,
        )
        app.say("here is the raw metadata: {}".format(raw_meta))
        
        return self.object
    
    def instantiate_model(self, child_objects) -> BaseModel:
        if obj:= create_instance(self.model, child_objects):
            return obj

        # Create a MaybeModel class with a dynamic name
        maybe_model_name = f"Maybe{self.model.__name__}"

        annotations = {self.model.__name__: Optional[self.model]}
        MaybeModel = type(maybe_model_name,
                            (BaseModel,),
                            {'__annotations__': annotations,
                            self.model.__name__: Field(default=None,
                                                        description="The actual object to be filled with metadata.")})

        return MaybeModel()  # we will need to fill in child objects later
        
def create_instance(instance, obj_dict:dict):
    new_obj_dict = {}
    for obj_key, obj in obj_dict.items():
        # Get the type of the object
        obj_type = type(obj)
        
        # Define a set of ignored types
        ignored_types = {str, int, float, bool} # could be updated to != ome_types
        # Skip ignored types
        if obj_type in ignored_types:
            continue
        # Iterate over the attributes and their types in the instance's class
        for attr_name, attr_type in instance.__annotations__.items():
            if attr_type == Optional[obj_type] or attr_type == obj_type:
                new_obj_dict[attr_name] = obj
                #setattr(instance, attr_name, obj)
                continue
            elif hasattr(attr_type, '__origin__') and attr_type.__origin__ == list and attr_type.__args__[0] == obj_type:
                new_obj_dict[attr_name] = [obj] # TODO: If the same attribute is already set, append to the list
                #getattr(instance, attr_name).append(obj)
                continue

    try:
        return instance(**new_obj_dict)
    except:
        return None # TODO: Think about how to handle this case properly
        

# Function to identify dependencies and collect all models
def collect_dependencies(model: Type[BaseModel], known_models: Dict[str, Type[BaseModel]], collected: Dict[str, Type[BaseModel]]):
    if model.__name__ in collected:
        return
    collected[model.__name__] = model
    for field in model.model_fields.values():
        field_type = field.annotation
        if hasattr(field_type, '__fields__'):  # If the field is a Pydantic model
            collect_dependencies(field_type, known_models, collected)
        elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            item_type = field_type.__args__[0]
            if hasattr(item_type, '__fields__'):
                collect_dependencies(item_type, known_models, collected)

# Function to create the tree
def create_dependency_tree(model: Type[BaseModel], known_models: Dict[str, Type[BaseModel]], visited: Set[str]) -> TreeNode:
    node = TreeNode(model)
    visited.add(model.__name__)
    
    for field in model.model_fields.values():
        field_type = field.annotation
        if hasattr(field_type, '__fields__') and field_type.__name__ not in visited:  # If the field is a Pydantic model
            child_node = create_dependency_tree(field_type, known_models, visited)
            node.add_child(child_node)
        elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            item_type = field_type.__args__[0]
            if hasattr(item_type, '__fields__') and item_type.__name__ not in visited:
                child_node = create_dependency_tree(item_type, known_models, visited)
                node.add_child(child_node)
    
    return node

# Function to build the tree starting from the root model
def build_tree(root_model: Type[BaseModel]) -> TreeNode:
    known_models = {model.__name__: model for model in globals().values() if isinstance(model, type) and issubclass(model, BaseModel)}
    collected_models = {}
    collect_dependencies(root_model, known_models, collected_models)
    
    return create_dependency_tree(root_model, known_models, set())

# Build the tree starting from the OME model
dependency_tree = build_tree(OME)

# Print the tree structure
def print_tree(node: TreeNode, indent: str = ""):
    print(f"{indent}{node.model.__name__}")
    for child in node.children:
        print_tree(child, indent + "  ")

dependency_tree.predict_meta(ome_tree)

print(dependency_tree.object.model_dump_json())