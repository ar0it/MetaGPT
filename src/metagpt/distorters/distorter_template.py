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
        out2 = utils.get_json(ET.fromstring(ome_xml))
        return out2

    def shuffle_order(self, dict_meta: dict) -> dict:
        """
        Shuffle the order of the keys in the ome xml
        """
        l_meta = list(dict_meta.items())
        random.shuffle(l_meta)
        dict_meta = dict(l_meta)
        return dict_meta

    def rename_keys(self, dict_meta: dict) -> dict:
        """
        Rename the keys in the ome xml to similar words using a GPT model.
        """
        pred = PredictorDistorter(str(dict_meta)).predict()["definitions"]
        return pred
    
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


    def pred(self, ome_xml: str, out_path:str) -> dict:
        """
        Predict the distorted data
        """
        dict_meta = self.xml_to_key_value(ome_xml)
        dict_meta = self.rename_keys(dict_meta)
        dict_meta = self.shuffle_order(dict_meta)
        self.save_fake_data(dict_meta, out_path)
        return dict_meta
                            

    def distort(self, ome_xml: str, out_path:str, should_pred:str="maybe") -> dict:
        """
        Distort the ome xml
        """
        if should_pred == "no":
            out = self.load_fake_data(out_path) or None
        elif should_pred == "yes":
            out = self.pred(ome_xml) or None
        elif should_pred == "maybe":
            out = self.load_fake_data(out_path) or self.pred(ome_xml, out_path=out_path) or None
        return out
