import os
import openai
import autogen
import argparse
from OME_predictor import XMLPredictor
import xml.etree.ElementTree as ET
from ome_types import from_xml
from ome_types import from_tiff
import javabridge
import bioformats

javabridge.start_vm(class_path=bioformats.JARS)


class OMEEvaluator:
    """
    This class evaluates the performance of a OME XML generation model by calculating the similarity between the generated
    and the ground truth OME XML. The similarity is defined as the number of identical nodes divided by the total number of
    nodes in the ground truth OME XML.
    """

    def __init__(self, path_to_raw_metadata=None,
                 gt_path="/home/aaron/PycharmProjects/MetaGPT/out/ome_xml.ome.xml",
                 pred_path="/home/aaron/PycharmProjects/MetaGPT/out/ome_xml.ome.xml"):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """

        self.prediction = self.read_ome_xml(pred_path)
        self.ground_truth = self.read_ome_xml(gt_path)
        self.score = None
        self.evaluate()

    def edit_distance(self, xml_a, xml_b):
        """
        Calculate the edit distance between two xml trees.
        """
        pass

    def get_paths(self, xml_root, path=''):
        """
        Helper function to get all paths in an XML tree.
        """
        paths = set()
        for child in xml_root:
            new_path = path + '/' + child.tag.split('}')[1]
            if child.attrib:
                for key in child.attrib.keys():
                    paths.add(new_path + '/' + key + '=' + child.attrib[key])
                    paths.update(self.get_paths(child, new_path))
            else:
                paths.add(new_path)
                paths.update(self.get_paths(child, new_path))
        return paths

    def path_difference(self, xml_a, xml_b):
        """
        Calculates the length of the difference between the path sets in two xml trees.
        """
        paths_a = self.get_paths(xml_a)
        paths_b = self.get_paths(xml_b)
        print("path difference: ", paths_a)
        return len(paths_a.symmetric_difference(paths_b))

    def read_ome_xml(self, path):
        """
        This method reads the ome xml file and returns the root element.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        return root

    def read_ome_tiff(self, path):
        """
        This method reads the ome tiff file.
        """
        ome = bioformats.OMEXML()
        print(ome)
        return ome

    def evaluate(self):
        """
        compare the two ome xml trees and return the similarity score by calculating the edit distance between the two xml.
        """
        print("Evaluation in progress...")
        self.score = self.path_difference(self.prediction, self.ground_truth)

    def report(self):
        """
        Write evaluation report to file.
        """
        with open(f"out/report_{self.path_ground_truth.split('/')[-1]}.txt", "w") as f:
            f.write("Evaluation Report\n")
            f.write(f"File: {self.path_ground_truth.split('/')[-1]}\n")
            f.write(f"Score: {self.score}\n")


javabridge.kill_vm()
