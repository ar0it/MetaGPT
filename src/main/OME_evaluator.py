import os
import openai
import autogen
import argparse
from OME_predictor import XMLPredictor
import xml.etree.ElementTree as ET
from ome_types import from_xml
from ome_types import from_tiff


class OMEEvaluator:
    """
    This class evaluates the performance of a OME XML generation model by calculating the similarity between the generated
    and the ground truth OME XML. The similarity is defined as the number of identical nodes divided by the total number of
    nodes in the ground truth OME XML.
    """

    def __init__(self, path_to_raw_metadata=None,
                 gt_path="/home/aaron/PycharmProjects/MetaGPT/raw_data/testetst_Image8_edited_.ome.tif",
                 pred_path="/home/aaron/PycharmProjects/MetaGPT/raw_data/testetst_Image8_edited_.ome.tif"):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """
        if pred_path is None:
            pass
            # self.prediction = XMLPredictor(path_to_raw_metadata).getOMEXML()
        else:
            self.prediction = self.read_ome_xml(pred_path)

        self.path_ground_truth = gt_path
        self.ground_truth = self.read_ome_xml(gt_path)
        self.score = None
        self.evaluate()
        self.report()

    def edit_distance(self, xml_a, xml_b):
        """
        Calculate the edit distance between two xml trees.
        """
        pass

    def path_difference(self, xml_a, xml_b):
        """
        Calculates the length of the difference between the path sets in two xml trees.
        """
        pass
    def read_ome_xml(self, path):
        """
        This method calls a xml reader to read the ome xml file.
        """
        return from_tiff(path)

    def evaluate(self):
        """
        compare the two ome xml trees and return the similarity score by calculating the edit distance between the two xml.
        """
        print(self.prediction)
        print("Evaluation in progress...")
        print(self.prediction.experimenter_groups)
        print(self.prediction.instruments)
        print(self.prediction._calculate_keys)
        self.score = 0

    def report(self):
        """
        Write evaulation report to file.
        """
        with open(f"out/report_{self.path_ground_truth.split('/')[-1]}.txt", "w") as f:
            f.write("Evaluation Report\n")
            f.write(f"File: {self.path_ground_truth.split('/')[-1]}\n")
            f.write(f"Score: {self.score}\n")
