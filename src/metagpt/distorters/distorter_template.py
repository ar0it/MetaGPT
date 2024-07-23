from metagpt.utils import utils
import random
from openai import OpenAI
from metagpt.predictors.predictor_distorter import PredictorDistorter
import xml.etree.ElementTree as ET
from ome_types import from_xml, to_xml, to_dict
import json
import os

class DistorterTemplate:
    """
    The distorter takes well formed ome xml as input and returns a "distorted" key value version of it.
    Distortion can include:
    - ome xml to key value
    - shuffling of the order of entried
    - keys get renamed to similair words
    """
    def xml_to_key_value(self, ome_xml: str) -> dict:
        """
        Convert the ome xml to key value pairs
        """
        try:
            out2 = utils.get_json(ET.fromstring(ome_xml))
        except Exception as e:
            print("The ET parsing failed, trying the ome_types parser")
        return out2

    def shuffle_order(self, dict_meta: dict) -> dict:
        """
        Shuffle the order of the keys in the ome xml
        """
        l_meta = list(dict_meta.items())
        random.shuffle(l_meta)
        dict_meta = dict(l_meta)
        return dict_meta

    def gen_mapping(self, dict_meta: dict) -> dict:
        """
        Rename the keys in the ome xml to similar words using a GPT model.
        """
        pred = PredictorDistorter(str(dict_meta)).predict()["definitions"]
        return pred
    
    def extract_unique_keys(self, metadata):
        """
        Extract all unique key names from a dictionary, including nested structures,
        without full paths or indices.
        
        Args:
        metadata (dict): The dictionary containing metadata.
        
        Returns:
        list: A list of unique key names.
        """
        def extract_keys(data):
            keys = set()
            if isinstance(data, dict):
                for key, value in data.items():
                    keys.add(key)
                    keys.update(extract_keys(value))
            elif isinstance(data, list):
                for item in data:
                    keys.update(extract_keys(item))
            return keys

        return list(extract_keys(metadata))
    
    def rename_metadata_keys(self, metadata, key_mapping):
        """
        Rename keys in a metadata dictionary based on a provided mapping.
        
        Args:
        metadata (dict): The original metadata dictionary.
        key_mapping (dict): A dictionary mapping original key names to new key names.
        
        Returns:
        dict: A new dictionary with renamed keys.
        """
        def rename_keys(data):
            if isinstance(data, dict):
                return {key_mapping.get(k, k): rename_keys(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [rename_keys(item) for item in data]
            else:
                return data

        return rename_keys(metadata)
    
    def save_fake_data(self, fake_data: dict, path: str):
        """
        Save the fake data to a file
        """
        with open(path, 'w', encoding='utf-8') as f: 
            json.dump(fake_data, f, ensure_ascii=False, indent=4)

    def load_fake_data(self, path: str) -> dict:
        """
        Load the fake data from a file
        """
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def isolate_keys(self, dict_meta: dict) -> dict:
        """
        Isolate the keys in the ome xml
        """
        dict_keys = {}
        for key in dict_meta.keys():
            dict_keys[key] = None
        return dict
    
    def pred(self, ome_xml: str, out_path:str) -> dict:
        """
        Predict the distorted data
        """
        dict_meta = self.xml_to_key_value(ome_xml)
        dict_keys = self.extract_unique_keys(dict_meta)
        dict_mapping = self.gen_mapping(dict_keys)
        dict_new_meta = self.rename_metadata_keys(dict_meta, dict_mapping)
        dict_meta = self.shuffle_order(dict_new_meta)
        self.save_fake_data(dict_new_meta, out_path)
        return dict_meta
                            
    def modify_metadata_structure(self, metadata, operations=None, probability=0.3):
        """
        Modify the structure of a metadata dictionary systematically and randomly.
        
        Args:
        metadata (dict): The original metadata dictionary.
        operations (list): List of operations to perform. If None, all operations are used.
        probability (float): Probability of applying an operation to each element (0.0 to 1.0).
        
        Returns:
        dict: A new dictionary with modified structure.
        """
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        def nest_dict(d, sep='.'):
            result = {}
            for key, value in d.items():
                parts = key.split(sep)
                d = result
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
            return result
        
        all_operations = [
            lambda x: {k: v for k, v in x.items() if random.random() > probability},  # Remove random keys
            lambda x: {**x, f"new_key_{random.randint(1,100)}": random.choice(["new_value", 42, True])},  # Add random key-value
            lambda x: {k: [v] if not isinstance(v, list) else v for k, v in x.items()},  # Wrap values in lists
            lambda x: flatten_dict(x),  # Flatten nested structure
            lambda x: nest_dict(flatten_dict(x)),  # Re-nest flattened structure differently
            lambda x: {k: {k: v} for k, v in x.items()},  # Nest each key-value pair
        ]
        
        operations = operations or all_operations
        
        def apply_operations(data):
            if isinstance(data, dict):
                data = {k: apply_operations(v) for k, v in data.items()}
                if random.random() < probability:
                    data = random.choice(operations)(data)
            elif isinstance(data, list):
                data = [apply_operations(item) for item in data]
            return data
        
        return apply_operations(metadata)

    def distort(self, ome_xml: str, out_path:str, should_pred:str="maybe") -> dict:
        """
        Distort the ome xml
        """
        if should_pred == "no":
            out = self.load_fake_data(out_path) or None
        elif should_pred == "yes":
            out = self.pred(ome_xml, out_path=out_path) or None
        elif should_pred == "maybe":
            out = self.load_fake_data(out_path) or self.pred(ome_xml, out_path=out_path) or None
        return out
