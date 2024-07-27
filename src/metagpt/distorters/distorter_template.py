"""
This module contains the DistorterTemplate class, which is responsible for distorting
well-formed OME XML into a modified key-value representation. The distortion process
can include converting XML to key-value pairs, shuffling the order of entries, and
renaming keys to similar words.
"""

import os
import json
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional

from metagpt.utils import utils
from metagpt.predictors.predictor_distorter import PredictorDistorter

class DistorterTemplate:
    """
    A class for distorting OME XML into modified key-value representations.

    The distorter takes well-formed OME XML as input and returns a "distorted"
    key-value version of it. Distortion can include:
    - OME XML to key-value conversion
    - Shuffling of the order of entries
    - Renaming keys to similar words
    """

    def xml_to_key_value(self, ome_xml: str) -> Dict[str, Any]:
        """
        Convert the OME XML to key-value pairs.

        Args:
            ome_xml (str): The input OME XML string.

        Returns:
            Dict[str, Any]: A dictionary representation of the XML.

        Raises:
            Exception: If parsing fails.
        """
        try:
            return utils.get_json(ET.fromstring(ome_xml))
        except Exception as e:
            print(f"ET parsing failed: {e}. Trying the ome_types parser.")
            raise

    def shuffle_order(self, dict_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shuffle the order of the keys in the OME XML.

        Args:
            dict_meta (Dict[str, Any]): The input dictionary.

        Returns:
            Dict[str, Any]: A new dictionary with shuffled keys.
        """
        items = list(dict_meta.items())
        random.shuffle(items)
        return dict(items)

    def gen_mapping(self, dict_meta: Dict[str, Any]) -> Dict[str, str]:
        """
        Rename the keys in the OME XML to similar words using a GPT model.

        Args:
            dict_meta (Dict[str, Any]): The input dictionary.

        Returns:
            Dict[str, str]: A dictionary mapping original keys to new keys.
        """
        predictor = PredictorDistorter(str(dict_meta))
        return predictor.predict()["definitions"]

    def extract_unique_keys(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Extract all unique key names from a dictionary, including nested structures,
        without full paths or indices.

        Args:
            metadata (Dict[str, Any]): The dictionary containing metadata.

        Returns:
            List[str]: A list of unique key names.
        """
        def extract_keys(data: Any) -> set:
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

    def rename_metadata_keys(self, metadata: Dict[str, Any], key_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Rename keys in a metadata dictionary based on a provided mapping.

        Args:
            metadata (Dict[str, Any]): The original metadata dictionary.
            key_mapping (Dict[str, str]): A dictionary mapping original key names to new key names.

        Returns:
            Dict[str, Any]: A new dictionary with renamed keys.
        """
        def rename_keys(data: Any) -> Any:
            if isinstance(data, dict):
                return {key_mapping.get(k, k): rename_keys(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [rename_keys(item) for item in data]
            else:
                return data

        return rename_keys(metadata)

    def save_fake_data(self, fake_data: Dict[str, Any], path: str) -> None:
        """
        Save the fake data to a file.

        Args:
            fake_data (Dict[str, Any]): The data to be saved.
            path (str): The file path where the data will be saved.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(fake_data, f, ensure_ascii=False, indent=4)

    def load_fake_data(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Load the fake data from a file.

        Args:
            path (str): The file path from which to load the data.

        Returns:
            Optional[Dict[str, Any]]: The loaded data, or None if the file doesn't exist.
        """
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def isolate_keys(self, dict_meta: Dict[str, Any]) -> Dict[str, None]:
        """
        Isolate the keys in the OME XML.

        Args:
            dict_meta (Dict[str, Any]): The input dictionary.

        Returns:
            Dict[str, None]: A dictionary with the same keys as the input, but all values set to None.
        """
        return {key: None for key in dict_meta.keys()}

    def pred(self, ome_xml: str, out_path: str) -> Dict[str, Any]:
        """
        Predict the distorted data.

        Args:
            ome_xml (str): The input OME XML string.
            out_path (str): The path where the distorted data will be saved.

        Returns:
            Dict[str, Any]: The distorted metadata.
        """
        dict_meta = self.xml_to_key_value(ome_xml)
        dict_keys = self.extract_unique_keys(dict_meta)
        dict_mapping = self.gen_mapping(dict_keys)
        dict_new_meta = self.rename_metadata_keys(dict_meta, dict_mapping)
        dict_meta = self.shuffle_order(dict_new_meta)
        self.save_fake_data(dict_new_meta, out_path)
        return dict_meta

    def modify_metadata_structure(self, metadata: Dict[str, Any], operations: Optional[List[callable]] = None, probability: float = 0.3) -> Dict[str, Any]:
        """
        Modify the structure of a metadata dictionary systematically and randomly.

        Args:
            metadata (Dict[str, Any]): The original metadata dictionary.
            operations (Optional[List[callable]]): List of operations to perform. If None, all operations are used.
            probability (float): Probability of applying an operation to each element (0.0 to 1.0).

        Returns:
            Dict[str, Any]: A new dictionary with modified structure.
        """
        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        def nest_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
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

        def apply_operations(data: Any) -> Any:
            if isinstance(data, dict):
                data = {k: apply_operations(v) for k, v in data.items()}
                if random.random() < probability:
                    data = random.choice(operations)(data)
            elif isinstance(data, list):
                data = [apply_operations(item) for item in data]
            return data

        return apply_operations(metadata)

    def distort(self, ome_xml: str, out_path: str, should_pred: str = "maybe") -> Optional[Dict[str, Any]]:
        """
        Distort the OME XML.

        Args:
            ome_xml (str): The input OME XML string.
            out_path (str): The path where the distorted data will be saved.
            should_pred (str): Whether to predict new data or use existing data. Options are "yes", "no", or "maybe".

        Returns:
            Optional[Dict[str, Any]]: The distorted metadata, or None if no data is available.
        """
        if should_pred == "no":
            return self.load_fake_data(out_path)
        elif should_pred == "yes":
            return self.pred(ome_xml, out_path=out_path)
        elif should_pred == "maybe":
            return self.load_fake_data(out_path) or self.pred(ome_xml, out_path=out_path)
        return None