"""
Main file for the MetaGPT project. This file runs the entire experiment.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------

from predictors.predictor_template import PredictorTemplate
from predictors.predictor_xml_annotation import PredictorXMLAnnotation
from predictors.predictor_annotation_net import PredictorXMLAnnotationNet
from predictors.predictor_network import PredictorNetwork
from predictors.predictor_simple import PredictorSimple

import importlib
import sys
import utils
import bioformats.logback
from OME_evaluator import OMEEvaluator
from DataClasses import Sample
from DataClasses import Experiment
from BioformatsReader import get_omexml_metadata
from BioformatsReader import get_raw_metadata
from BioformatsReader import raw_to_tree
from utils import _init_logger
import os
import bioformats
import javabridge
from contextlib import contextmanager
import ome_types

# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------
wd = os.getcwd()
ome_schema_path = f'{wd}/in/schema/ome_xsd.txt'
raw_meta_path = f"{wd}/in/metadata/raw_Metadata_Image8.txt"
network_paths = [
    f"{wd}/out/assistant_outputs/veronika_example_response.txt"
    ]

gt_paths = [
    f"{wd}/in/images/small_images/Rio9 0528.tif",
    #f"{wd}/in/images/Image_8.czi",
    #f"{wd}/in/images/testetst_Image8_edited_.ome.tif",
    #f"{wd}/in/images/11_10_21_48h_H2BmCherryB16F10OVA_CD4cytcells_bsAb_endpoint.lif",
    #f"{wd}/in/images/rFUNC.nii",
    ]
# read the image input folder for all image paths
all_paths = [f"{wd}/in/images/{f}" for f in os.listdir(f"{wd}/in/images")]
out = f"{wd}/out/"
print(out)

# ----------------------------------------------------------------------------------------------------------------------
# Read the Data
# ----------------------------------------------------------------------------------------------------------------------
with open(ome_schema_path, "r") as f:
    ome_xsd = f.read()
#
# with open(raw_meta_path, "r") as f:
#     raw_meta = f.read()

experiment = Experiment(name="Experiment1", samples={})

# ----------------------------------------------------------------------------------------------------------------------
# Prediction Pipeline
# ----------------------------------------------------------------------------------------------------------------------

javabridge.start_vm(class_path=bioformats.JARS)
_init_logger()
should_predict = "maybe"

for path in gt_paths:
    print("-"*60)
    print("Processing image:")
    print(path)
    print("-"*60)
    # ------------------------------------------------------------------------------------------------------------------
    # Bioformats
    # ------------------------------------------------------------------------------------------------------------------
    print("-"*10+"Bioformats"+"-"*10)

    out_bioformats = get_omexml_metadata(path=path) # the raw metadata as ome xml str
    raw_meta = get_raw_metadata(path=path) # the raw metadata as dictionary of key value pairs
    tree_meta = raw_to_tree(raw_meta) # the raw metadata as nested dictionary

    name = path.split("/")[-1].split(".")[0]
    format = path.split("/")[-1].split(".")[1]

    bio_sample = Sample(name=name,
                        metadata_str=out_bioformats,
                        method="Bioformats",
                        format=format)
    
    experiment.add_sample(bio_sample)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Simple Predictor
    # ------------------------------------------------------------------------------------------------------------------
    utils.make_prediction(
        predictor=PredictorSimple,
        in_data=tree_meta,
        experiment=experiment,
        name=name,
        should_predict=should_predict,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Marvin
    # ------------------------------------------------------------------------------------------------------------------
    
    """
    utils.make_prediction(
        predictor=None,
        in_data=tree_meta,
        experiment=experiment,
        name=name,
        should_predict=should_predict,
    )
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Network Predictor
    # ------------------------------------------------------------------------------------------------------------------
    utils.make_prediction(
        predictor=PredictorNetwork,
        in_data=tree_meta,
        experiment=experiment,
        name=name,
        should_predict=should_predict,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Agent Graph
    # ------------------------------------------------------------------------------------------------------------------
    """
    utils.make_prediction(
        predictor=Pred,
        in_data=tree_meta,
        experiment=experiment,
        name=name,
        should_predict=should_predict,
    )
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Simple Annotation Predictor
    # ------------------------------------------------------------------------------------------------------------------
    utils.make_prediction(
        predictor=PredictorXMLAnnotation,
        in_data=tree_meta,
        experiment=experiment,
        name=name,
        should_predict=should_predict,
        start_point=out_bioformats
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Network Annotation Predictor
    # ------------------------------------------------------------------------------------------------------------------
    utils.make_prediction(
        predictor=PredictorXMLAnnotationNet,
        in_data=tree_meta,
        experiment=experiment,
        name=name,
        should_predict=should_predict,
        start_point=out_bioformats
    )

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation Pipeline
# ----------------------------------------------------------------------------------------------------------------------
print("-"*60)
print("Evaluation")
print("-"*60)
ome_eval = OMEEvaluator(schema=ome_xsd,
                        experiment=experiment,
                        out_path=out)
javabridge.kill_vm()
