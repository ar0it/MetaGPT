from ome_types import OME
from typing import List, Dict, Set, Type, Any
from pydantic import BaseModel
from deprecated import deprecated
from contextlib import contextmanager
import json
from datetime import datetime
import sys
from io import StringIO
from docstring_parser import parse

def render_cell_output(output_path):
    """
    Load the captured output from a file and render it.

    Parameters:
    output_path (str): Path to the output file where the cell output is saved.
    """
    try:
        # Read the output file
        with open(output_path, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        
        # Print the captured stdout and stderr
        if 'stdout' in output_data:
            print(output_data['stdout'], end='')
        if 'stderr' in output_data:
            print(output_data['stderr'], end='', file=sys.stderr)
        
        print(f"\nCell output loaded from {output_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
    
    def flush(self):
        for stream in self.streams:
            stream.flush()

@contextmanager
def save_and_stream_output(output_path=f"out/jupyter_cell_outputs/cell_output_{datetime.now().isoformat()}_.json"):
    """
    Context manager to capture the output of a code block, save it to a file,
    and print it to the console in real-time.

    Parameters:
    output_path (str): Path to the output file where the cell output will be saved.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    sys.stdout = Tee(sys.stdout, stdout_buffer)
    sys.stderr = Tee(sys.stderr, stderr_buffer)
    
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        output_data = {
            'stdout': stdout_buffer.getvalue(),
            'stderr': stderr_buffer.getvalue()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"\nCell output saved to {output_path}")


def from_dict(ome_dict):
    """
    Convert a dictionary to an OME object.
    """
    def set_attributes(obj, data):
        for key, value in data.items():
            if isinstance(value, dict):
                attr = getattr(obj, key, None)
                if attr is not None:
                    set_attributes(attr, value)
                else:
                    setattr(obj, key, value)
            elif isinstance(value, list):
                # Assume the list items are dictionaries that need similar treatment
                existing_list = getattr(obj, key, [])
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        if i < len(existing_list):
                            set_attributes(existing_list[i], item)
                        else:
                            # Handle the case where the list item does not already exist
                            existing_list.append(item)
                    else:
                        existing_list.append(item)
                setattr(obj, key, existing_list)
            else:
                setattr(obj, key, value)
    
    ome = OME()
    set_attributes(ome, ome_dict)
    return ome

def _init_logger():
    """This is so that Javabridge doesn't spill out a lot of DEBUG messages
    during runtime.
    From CellProfiler/python-bioformats.
    """
    import javabridge as jb
    rootLoggerName = jb.get_static_field("org/slf4j/Logger",
                                         "ROOT_LOGGER_NAME",
                                         "Ljava/lang/String;")

    rootLogger = jb.static_call("org/slf4j/LoggerFactory",
                                "getLogger",
                                "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                rootLoggerName)

    logLevel = jb.get_static_field("ch/qos/logback/classic/Level",
                                   "ERROR",
                                   "Lch/qos/logback/classic/Level;")

    jb.call(rootLogger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            logLevel)
    
@deprecated()
# Function to identify dependencies and collect all models
def collect_dependencies(model: Type[BaseModel], known_models: Dict[str, Type[BaseModel]], collected: Dict[str, Type[BaseModel]]):
    if model.__name__ in collected:
        return
    collected[model.__name__] = model
    for field in model.__fields__.values():
        field_type = field.annotation
        if hasattr(field_type, '__fields__'):  # If the field is a Pydantic model
            collect_dependencies(field_type, known_models, collected)
        elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            item_type = field_type.__args__[0]
            if hasattr(item_type, '__fields__'):
                collect_dependencies(item_type, known_models, collected)

@deprecated()
# Function to identify dependencies for sorting
def get_dependencies(model: Type[BaseModel], known_models: Dict[str, Type[BaseModel]]) -> Set[str]:
    dependencies = set()
    for field in model.model_fields.values():
        field_type = field.annotation
        if hasattr(field_type, '__fields__'):  # If the field is a Pydantic model
            dependencies.add(field_type.__name__)
            dependencies.update(get_dependencies(field_type, known_models))
        elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            item_type = field_type.__args__[0]
            if hasattr(item_type, '__fields__'):
                dependencies.add(item_type.__name__)
                dependencies.update(get_dependencies(item_type, known_models))
    return dependencies

@deprecated()
# Function to sort models by dependencies
def sort_models_by_dependencies(root_model: Type[BaseModel]) -> List[Type[BaseModel]]:
    known_models = {model.__name__: model for model in globals().values() if isinstance(model, type) and issubclass(model, BaseModel)}
    collected_models = {}
    collect_dependencies(root_model, known_models, collected_models)
    
    dependency_graph: Dict[str, Set[str]] = {
        model.__name__: get_dependencies(model, known_models) for model in collected_models.values()
    }
    
    sorted_models = []
    while dependency_graph:
        # Find models with no dependencies or dependencies that are already resolved
        resolvable_models = {name for name, deps in dependency_graph.items() if not deps - set(m.__name__ for m in sorted_models)}
        if not resolvable_models:
            raise ValueError("Circular dependency detected or unresolved dependency found")
        
        # Add resolvable models to the sorted list
        for model_name in resolvable_models:
            sorted_models.append(collected_models[model_name])
            dependency_graph.pop(model_name)
    
    return sorted_models


import re
from docstring_parser import parse
from typing import Any, Dict, Union

def openai_schema(cls) -> Dict[str, Any]:
    """
    Return the schema in the format of OpenAI's schema as jsonschema

    Note:
        It's important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

    Returns:
        dict: A dictionary in the format of OpenAI's schema as jsonschema
    """
    schema = cls.model_json_schema()
    docstring = parse(cls.__doc__ or "")
    
    def clean_description(desc: Union[str, Any]) -> str:
        if not isinstance(desc, str):
            return str(desc)
        cleaned = re.sub(r'\s+', ' ', desc)
        return cleaned.strip()

    def resolve_ref(ref: str, definitions: Dict[str, Any]) -> Dict[str, Any]:
        if not ref.startswith('#/$defs/'):
            return {'type': 'object', 'description': f'Unresolved reference: {ref}'}
        def_name = ref.split('/')[-1]
        return definitions.get(def_name, {'type': 'object', 'description': f'Missing definition: {def_name}'})

    def flatten_schema(prop: Any, definitions: Dict[str, Any]) -> Any:
        if not isinstance(prop, dict):
            return prop

        flattened = {}
        unnecessary_keys = ['title', 'name', 'namespace', "type"]  # Add any other keys you want to exclude
        for key, value in prop.items():
            if key not in unnecessary_keys:
                if key == '$ref':
                    flattened.update(flatten_schema(resolve_ref(value, definitions), definitions))
                elif key == "required" and (value == True or value == False):
                    continue
                elif key == 'description' and type(value) == str:
                    flattened[key] = clean_description(value)
                elif isinstance(value, dict):
                    flattened[key] = flatten_schema(value, definitions)
                elif isinstance(value, list):
                    flattened[key] = [flatten_schema(item, definitions) for item in value]
                else:
                    flattened[key] = value

        return flattened

    # Extract $defs if present
    definitions = schema.pop('$defs', {})

    # Flatten the main schema
    flattened_schema = flatten_schema(schema, definitions)

    # Add flattened definitions to properties
    if 'properties' not in flattened_schema:
        flattened_schema['properties'] = {}
    flattened_schema['properties']['definitions'] = {
        'type': 'object',
        'properties': {k: flatten_schema(v, definitions) for k, v in definitions.items()}
    }

    # Add descriptions from docstring
    for param in docstring.params:
        if param.arg_name in flattened_schema.get('properties', {}) and param.description:
            flattened_schema['properties'][param.arg_name]['description'] = clean_description(param.description)

    # Combine short_description and long_description for a more complete description
    full_description = docstring.short_description or ""
    if docstring.long_description:
        full_description += " " + docstring.long_description if full_description else docstring.long_description
    
    description = clean_description(full_description) if full_description else f"Correctly extracted `{cls.__name__}` with all the required parameters with correct types"

    return {
        "type": "function",
        "function":{
               "name": schema.get('title', cls.__name__),
               "description": description,
               "parameters": {
                   "type": "object",
                   "required": [],
                   "properties": flattened_schema["properties"]
                }
            }
    }
    
