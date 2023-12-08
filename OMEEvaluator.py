import os
import openai
import autogen
import argparse
from XMLPredictor import XMLPredictor
import xml.etree.ElementTree as ET


class OMEEvaluator:
    """
    This class evaluates the performance of a OME XML generation model by calculating the similarity between the generated
    and the ground truth OME XML. The similarity is defined as the number of identical nodes divided by the total number of
    nodes in the ground truth OME XML.
    """

    def __init__(self, path_to_raw_metadata, gt_path=None, pred_path=None):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """
        if pred_path is None:
            self.prediction = XMLPredictor(path_to_raw_metadata).getOMEXML()
        else:
            self.prediction = self.read_ome_xml(pred_path)

        self.path_ground_truth = gt_path
        self.ground_truth = self.read_ome_xml(gt_path)

    def read_ome_xml(self, path):
        """
        This method calls an xml reader to read the ome xml file.
        """
        return ET.parse(path).getroot()

    def evaluate(self):
        """
        compare the two ome xml trees and return the similarity score by calculating the edit distance between the two xml.
        """
        for child in self.prediction:
            print(child.tag, child.attrib)
            for child in self.ground_truth:
                print(child.tag, child.attrib)






