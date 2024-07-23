from typing import List, Dict, Set, Type, Optional
from pydantic import BaseModel, Field
#from marvin.beta import Application
from collections.abc import Iterator
import importlib.util
import sys
from ome_types import from_xml, to_xml
from ome_types.model import OME
import metagpt.utils.BioformatsReader as BioformatsReader
from metagpt.predictors.predictor_template import PredictorTemplate
from metagpt.predictors.predictor_state import PredictorState
import metagpt.utils.utils as utils

prompt="""
You are part of a toolchain which is supposed to predict metadata for the OME model.
You will be interacting with other toolchain, therefore asking questions or providing any human readable output is not necessary.
You should concisely and structured document your thought process anyway for logging purposes.
THe toolchain is a Tree structure of models, designed with the ome schema in mind.
This mean each node has a predictor such as yourself, and will be predicting the metadata for a submodel of the OME schema.
One such submodel could be the Image node or the Pixels node, you therefore only need to predict the highes level node in your given state (lower levels were already attempted by other agents).
The tree structure is designed to predict the metadata in a bottom up manner, starting from the lower level node.
This way ther are no missing dependencies when predicting the metadata.
Incoming metadata will be provided in raw format, that mean a list of key value pairs.
Your task  will be to translate these key value pairs to the appropriate OME schema property.
Try to figure out which porperty is which by looking at the schema and the raw metadata in holistic manner.
The state you were given is the corresponing schema snipped you are supposed to fill out, remember only the highes level, the lower levels provided are only for validation and context.
Since this is a hard problem I will need you to think step by step and use chain of thought.
Here is the structure of how to approach the problem step by step:
1. Look at the schema/state, which elements are missing, which one do you plan to add
2. Figure out if they have mandatory fields or dependencies, by looking at the state
3. Look at the raw metadata, speicifcally try to find if any of the raw metadata would fit into the previously identified fields
4. Come to a conclusion wheter you can add the element or if there are mandatory fields missing
5. If you cant add the element thats okay, just skip it and dont update the state not every state needs to be updated
6. If you can add the element, generate the tool call to add the element, make sure to add the element and all the MANDATORY fields at once to not get a validation error. Think about which operation is best fitting, are you addding a new node or modifying an existing one=?
7. Call the tool with the minimum amount of metadata required for validity, then iteratively add more metadata. Dont add everything at once.
8. If validation errors occur approach them systematically, is there a mandatory node missing? is there a dependency missing? is the value of the node correct? If you cant solve the problem skip the node and dont update the state.
9. Iteratively add as much metadata as possible, until ther are no fitting key value pairs left.
Since the tool usage requires a python middle layer the commonly used property names (ome schema) are written a bit differently, for example SizeX is written as size_x etc.
Check the state for the correct property names.
The validation tool will make you aware of the correct property names if you failed.
I have split the ome model into several submodels/ states, not all are actually required.
If you feel like there is not metadata for the current data model you are working on, no worries just skip that model and dont update the state.
Sometimes you might be provided with a "MaybeModel", the reason for this is simple, those models couldnt be instantiated yet because they have mandatory fields missing.
If you find the necessary metadata to fill in the missing fields, fill out the MaybeModel accordingly.
Subsequent tools will take care of the merging of the states.
Its very well possible some fields remain empty.
Remember to solve this problem step by step and use chain of thought to solve it.
Again, you are not interacting with a human but are part of a chain of tools that are supposed to solve this problem.
Under no circumstances can you ask questions.
You will have to decide on your own, if in doubt, skip the node.
"""

# Define the tree node class
class TreeNode:
    def __init__(self, model: Type[BaseModel]):
        self.model = model
        self.state:BaseModel = None
        self.children:list[TreeNode] = []


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

    
    def predict_meta(self, raw_meta:str, indent:int=0) -> BaseModel:

        # add the metdata from the child nodes first
        child_objects = {}
        for child in self.children:
            if child_obj := child.predict_meta(raw_meta, indent=indent+1):
                child_objects[child.model.__name__] = child_obj

        self.state = self.instantiate_model(child_objects)
        # TODO: loop here in case the field allows for the same type multiple times (i.e OME(images=[Image, Image, Image])) Maybe loop until AI doesnt predict any new
        # In that case I need to remove metadata from the raw_meta that has already been used
        #print(f"Predicting metadata for {self.model.__name__}, self.object={self.object}, required={list(self.required_fields(self.model))}")
        response, cost, attemtps = PredictorState(state=self.state, raw_meta=raw_meta).predict()

        if response!=None:
            self.state = from_xml(response)
            # MaybeModel to Model
            if self.state.__class__.__name__.startswith("Maybe"):
                self.state = getattr(self.state, self.model.__name__)
            # return None if the object is empty other than the ID
            attributes = {k:v for k, v in self.state.__dict__.items() if v and k != "kind" and k != "id"}
            if not attributes:
                self.state = None
        else:
            self.state = None
        return self.state
    
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
        

class PredictorStateTree(PredictorTemplate):
    def __init__(self, raw_meta:str, model:BaseModel=None):
        if model is None:
            model = OME
        self.model = model
        self.raw_meta = raw_meta
        #self.state = state
        self.dependency_tree = self.build_tree(model)

    def predict(self) -> BaseModel:
        return self.dependency_tree.predict_meta(self.raw_meta), None, None
    
    def collect_dependencies(self, model: Type[BaseModel], known_models: Dict[str, Type[BaseModel]], collected: Dict[str, Type[BaseModel]]):
        if model.__name__ in collected:
            return
        collected[model.__name__] = model
        for field in model.model_fields.values():
            field_type = field.annotation
            if hasattr(field_type, '__fields__'):  # If the field is a Pydantic model
                self.collect_dependencies(field_type, known_models, collected)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                item_type = field_type.__args__[0]
                if hasattr(item_type, '__fields__'):
                    self.collect_dependencies(item_type, known_models, collected)

    # Function to create the tree
    def create_dependency_tree(self, model: Type[BaseModel], known_models: Dict[str, Type[BaseModel]], visited: Set[str]) -> TreeNode:
        node = TreeNode(model)
        visited.add(model.__name__)
        
        for field in model.model_fields.values():
            field_type = field.annotation
            if hasattr(field_type, '__fields__') and field_type.__name__ not in visited:  # If the field is a Pydantic model
                child_node = self.create_dependency_tree(field_type, known_models, visited)
                node.add_child(child_node)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                item_type = field_type.__args__[0]
                if hasattr(item_type, '__fields__') and item_type.__name__ not in visited:
                    child_node = self.create_dependency_tree(item_type, known_models, visited)
                    node.add_child(child_node)
        
        return node

    # Function to build the tree starting from the root model
    def build_tree(self, root_model: Type[BaseModel]) -> TreeNode:
        known_models = {model.__name__: model for model in globals().values() if isinstance(model, type) and issubclass(model, BaseModel)}
        collected_models = {}
        self.collect_dependencies(root_model, known_models, collected_models)
        
        return self.create_dependency_tree(root_model, known_models, set())

    # Build the tree starting from the OME model
    #dependency_tree = build_tree(Image)

    # Print the tree structure
    def print_tree(self, node: TreeNode = None, indent: str = ""):
        if node is None:
            node = self.dependency_tree
        print(f"{indent}{node.model.__name__}")
        for child in node.children:
            self.print_tree(child, indent + "  ")