"""
This module contains the PredictorStateTree class and related components for
predicting OME metadata using a tree-based approach with OpenAI's language model.
"""

from typing import List, Dict, Set, Type, Optional, Iterator, Any, Tuple
from pydantic import BaseModel, Field
from collections.abc import Iterator

from ome_types import from_xml, to_xml
from ome_types.model import OME

from metagpt.predictors.predictor_template import PredictorTemplate
from metagpt.predictors.predictor_state import PredictorState
import metagpt.utils.utils as utils

PROMPT = """
You are part of a toolchain which is supposed to predict metadata for the OME model.
You will be interacting with other toolchain components, therefore asking questions or providing any human-readable output is not necessary.
You should concisely and structurally document your thought process for logging purposes.
The toolchain is a tree structure of models, designed with the OME schema in mind.
This means each node has a predictor such as yourself, and will be predicting the metadata for a submodel of the OME schema.
One such submodel could be the Image node or the Pixels node, you therefore only need to predict the highest level node in your given state (lower levels were already attempted by other agents).
The tree structure is designed to predict the metadata in a bottom-up manner, starting from the lower-level nodes.
This way there are no missing dependencies when predicting the metadata.
Incoming metadata will be provided in raw format, that means a list of key-value pairs.
Your task will be to translate these key-value pairs to the appropriate OME schema property.
Try to figure out which property is which by looking at the schema and the raw metadata in a holistic manner.
The state you were given is the corresponding schema snippet you are supposed to fill out, remember only the highest level, the lower levels provided are only for validation and context.
Since this is a hard problem, I will need you to think step by step and use chain of thought.
Here is the structure of how to approach the problem step by step:
1. Look at the schema/state, which elements are missing, which ones do you plan to add
2. Figure out if they have mandatory fields or dependencies, by looking at the state
3. Look at the raw metadata, specifically try to find if any of the raw metadata would fit into the previously identified fields
4. Come to a conclusion whether you can add the element or if there are mandatory fields missing
5. If you can't add the element that's okay, just skip it and don't update the state. Not every state needs to be updated
6. If you can add the element, generate the tool call to add the element, make sure to add the element and all the MANDATORY fields at once to not get a validation error. Think about which operation is best fitting, are you adding a new node or modifying an existing one?
7. Call the tool with the minimum amount of metadata required for validity, then iteratively add more metadata. Don't add everything at once.
8. If validation errors occur, approach them systematically: is there a mandatory node missing? Is there a dependency missing? Is the value of the node correct? If you can't solve the problem, skip the node and don't update the state.
9. Iteratively add as much metadata as possible, until there are no fitting key-value pairs left.
Since the tool usage requires a Python middle layer, the commonly used property names (OME schema) are written a bit differently, for example, SizeX is written as size_x etc.
Check the state for the correct property names.
The validation tool will make you aware of the correct property names if you failed.
I have split the OME model into several submodels/states, not all are actually required.
If you feel like there is no metadata for the current data model you are working on, no worries, just skip that model and don't update the state.
Sometimes you might be provided with a "MaybeModel", the reason for this is simple, those models couldn't be instantiated yet because they have mandatory fields missing.
If you find the necessary metadata to fill in the missing fields, fill out the MaybeModel accordingly.
Subsequent tools will take care of the merging of the states.
It's very well possible some fields remain empty.
Remember to solve this problem step by step and use chain of thought to solve it.
Again, you are not interacting with a human but are part of a chain of tools that are supposed to solve this problem.
Under no circumstances can you ask questions.
You will have to decide on your own, if in doubt, skip the node.
"""

class TreeNode:
    """Represents a node in the dependency tree for OME metadata prediction."""

    def __init__(self, model: Type[BaseModel]):
        self.model = model
        self.state: Optional[BaseModel] = None
        self.children: List['TreeNode'] = []

    def add_child(self, child: 'TreeNode') -> None:
        """Add a child node to this node."""
        self.children.append(child)

    def __repr__(self) -> str:
        return f"TreeNode(model={self.model.__name__}, children={self.children})"
    
    def required_fields(self, model: Type[BaseModel], recursive: bool = False) -> Iterator[str]:
        """
        Get all required fields of a Pydantic model, optionally including nested models.
        
        Args:
            model (Type[BaseModel]): The Pydantic model to inspect.
            recursive (bool): Whether to include fields from nested models.

        Yields:
            str: Names of required fields.
        """
        for name, field in model.model_fields.items():
            if not field.is_required():
                continue
            t = field.annotation
            if recursive and isinstance(t, type) and issubclass(t, BaseModel):
                yield from self.required_fields(t, recursive=True)
            else:
                yield name

    def predict_meta(self, raw_meta: str, indent: int = 0) -> Tuple[Optional[BaseModel], float, int]:
        """
        Predict metadata for this node and its children.

        Args:
            raw_meta (str): The raw metadata to process.
            indent (int): The indentation level for printing (used for debugging).

        Returns:
            Tuple[Optional[BaseModel], float, int]: The predicted state, total cost, and total attempts.
        """
        cost, attempts = 0, 0
        child_objects = {}
        for child in self.children:
            child_obj, cost_child, attempts_child = child.predict_meta(raw_meta, indent=indent+1)
            cost += cost_child
            attempts += attempts_child
            if child_obj:
                child_objects[child.model.__name__] = child_obj

        self.state = self.instantiate_model(child_objects)
        response, cost_pred, attempts_pred = PredictorState(state=self.state, raw_meta=raw_meta).predict()
        cost += cost_pred
        attempts += attempts_pred
        if response is not None:
            self.state = from_xml(response)
            if self.state.__class__.__name__.startswith("Maybe"):
                self.state = getattr(self.state, self.model.__name__)
            attributes = {k: v for k, v in self.state.__dict__.items() if v and k != "kind" and k != "id"}
            if not attributes:
                self.state = None
        else:
            self.state = None
        return self.state, cost, attempts
    
    def instantiate_model(self, child_objects: Dict[str, Any]) -> BaseModel:
        """
        Instantiate the model for this node, including child objects.

        Args:
            child_objects (Dict[str, Any]): Dictionary of child objects to include.

        Returns:
            BaseModel: The instantiated model, or a MaybeModel if instantiation fails.
        """
        if obj := create_instance(self.model, child_objects):
            return obj

        maybe_model_name = f"Maybe{self.model.__name__}"
        annotations = {self.model.__name__: Optional[self.model]}
        MaybeModel = type(maybe_model_name,
                          (BaseModel,),
                          {'__annotations__': annotations,
                           self.model.__name__: Field(default=None,
                                                      description="The actual object to be filled with metadata.")})
        return MaybeModel()

def create_instance(instance: Type[BaseModel], obj_dict: Dict[str, Any]) -> Optional[BaseModel]:
    """
    Create an instance of a Pydantic model, filling it with child objects.

    Args:
        instance (Type[BaseModel]): The Pydantic model class to instantiate.
        obj_dict (Dict[str, Any]): Dictionary of child objects to include.

    Returns:
        Optional[BaseModel]: The instantiated model, or None if instantiation fails.
    """
    new_obj_dict = {}
    for obj_key, obj in obj_dict.items():
        obj_type = type(obj)
        ignored_types = {str, int, float, bool}
        if obj_type in ignored_types:
            continue
        for attr_name, attr_type in instance.__annotations__.items():
            if attr_type == Optional[obj_type] or attr_type == obj_type:
                new_obj_dict[attr_name] = obj
            elif hasattr(attr_type, '__origin__') and attr_type.__origin__ == list and attr_type.__args__[0] == obj_type:
                new_obj_dict[attr_name] = [obj]

    try:
        return instance(**new_obj_dict)
    except:
        return None

class PredictorStateTree(PredictorTemplate):
    """
    A predictor class that uses a tree-based approach to predict OME metadata.
    """

    def __init__(self, raw_meta: str, model: Type[BaseModel] = None):
        """
        Initialize the PredictorStateTree.

        Args:
            raw_meta (str): The raw metadata to process.
            model (Type[BaseModel], optional): The root model to use. Defaults to OME.
        """
        super().__init__()
        self.model = model or OME
        self.raw_meta = raw_meta
        self.dependency_tree = self.build_tree(self.model)

    def predict(self) -> Tuple[Optional[BaseModel], Optional[float], Optional[int]]:
        """
        Predict the OME metadata using the dependency tree.

        Returns:
            Tuple[Optional[BaseModel], Optional[float], Optional[int]]:
                The predicted metadata, cost (None for this implementation), and attempts (None for this implementation).
        """
        return self.dependency_tree.predict_meta(self.raw_meta)
    
    def collect_dependencies(self, model: Type[BaseModel], known_models: Dict[str, Type[BaseModel]], collected: Dict[str, Type[BaseModel]]) -> None:
        """
        Collect all dependent models for a given model.

        Args:
            model (Type[BaseModel]): The model to collect dependencies for.
            known_models (Dict[str, Type[BaseModel]]): Dictionary of known models.
            collected (Dict[str, Type[BaseModel]]): Dictionary to store collected models.
        """
        if model.__name__ in collected:
            return
        collected[model.__name__] = model
        for field in model.model_fields.values():
            field_type = field.annotation
            if hasattr(field_type, '__fields__'):
                self.collect_dependencies(field_type, known_models, collected)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                item_type = field_type.__args__[0]
                if hasattr(item_type, '__fields__'):
                    self.collect_dependencies(item_type, known_models, collected)

    def create_dependency_tree(self, model: Type[BaseModel], known_models: Dict[str, Type[BaseModel]], visited: Set[str]) -> TreeNode:
        """
        Create a dependency tree for a given model.

        Args:
            model (Type[BaseModel]): The model to create a tree for.
            known_models (Dict[str, Type[BaseModel]]): Dictionary of known models.
            visited (Set[str]): Set of visited model names.

        Returns:
            TreeNode: The root node of the created tree.
        """
        node = TreeNode(model)
        visited.add(model.__name__)
        
        for field in model.model_fields.values():
            field_type = field.annotation
            if hasattr(field_type, '__fields__') and field_type.__name__ not in visited:
                child_node = self.create_dependency_tree(field_type, known_models, visited)
                node.add_child(child_node)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                item_type = field_type.__args__[0]
                if hasattr(item_type, '__fields__') and item_type.__name__ not in visited:
                    child_node = self.create_dependency_tree(item_type, known_models, visited)
                    node.add_child(child_node)
        
        return node

    def build_tree(self, root_model: Type[BaseModel]) -> TreeNode:
        """
        Build the complete dependency tree starting from the root model.

        Args:
            root_model (Type[BaseModel]): The root model to start building the tree from.

        Returns:
            TreeNode: The root node of the built tree.
        """
        known_models = {model.__name__: model for model in globals().values() if isinstance(model, type) and issubclass(model, BaseModel)}
        collected_models = {}
        self.collect_dependencies(root_model, known_models, collected_models)
        
        return self.create_dependency_tree(root_model, known_models, set())

    def print_tree(self, node: Optional[TreeNode] = None, indent: str = "") -> None:
        """
        Print the structure of the dependency tree.

        Args:
            node (Optional[TreeNode]): The node to start printing from. If None, starts from the root.
            indent (str): The current indentation string.
        """
        if node is None:
            node = self.dependency_tree
        print(f"{indent}{node.model.__name__}")
        for child in node.children:
            self.print_tree(child, indent + "  ")