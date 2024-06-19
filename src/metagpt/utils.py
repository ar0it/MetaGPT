from ome_types import OME
from typing import List, Dict, Set, Type
from pydantic import BaseModel
from deprecated import deprecated

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