from OME_evaluator import OMEEvaluator
from OME_predictor import XMLPredictor
import argparse
import os


#Predictor = XMLPredictor(path_to_raw_metadata="/home/aaron/PycharmProjects/MetaGPT/raw_Metadata_Image8.txt")
Evaluator = OMEEvaluator(path_to_raw_metadata="/ground_truth/raw_Metadata_Image8.txt")