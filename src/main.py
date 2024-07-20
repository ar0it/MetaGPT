"""
Main file for the MetaGPT project. This file runs the entire experiment.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------
from metagpt.predictors.predictor_template import PredictorTemplate
from metagpt.predictors.predictor_simple_annotator import PredictorSimpleAnnotation
from metagpt.predictors.predictor_network_annotator import PredictorNetworkAnnotation
from metagpt.predictors.predictor_network import PredictorNetwork
from metagpt.predictors.predictor_marvin import PredictorMarvin
from metagpt.predictors.predictor_simple import PredictorSimple
from metagpt.experiments.experiment_template import ExperimentTemplate
import importlib
import sys
import metagpt.utils.utils as utils
import bioformats.logback
from metagpt.evaluators.OME_evaluator import OMEEvaluator
from metagpt.utils.DataClasses import Sample
from metagpt.utils.DataClasses import Dataset
from metagpt.utils.BioformatsReader import get_omexml_metadata
from metagpt.utils.BioformatsReader import get_raw_metadata
from metagpt.utils.BioformatsReader import raw_to_tree
from metagpt.utils.utils import _init_logger
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
# read the image input folder for all image paths
all_paths = [f"{wd}/in/images/{f}" for f in os.listdir(f"{wd}/in/images")]
out = f"{wd}/out/"

# ----------------------------------------------------------------------------------------------------------------------
# Read the Data
# ----------------------------------------------------------------------------------------------------------------------
with open(ome_schema_path, "r") as f:
    ome_xsd = f.read()
#
# with open(raw_meta_path, "r") as f:
#     raw_meta = f.read()

# ----------------------------------------------------------------------------------------------------------------------
# Prediction Pipeline
# ----------------------------------------------------------------------------------------------------------------------

javabridge.start_vm(class_path=bioformats.JARS)
_init_logger()

experiment = ExperimentTemplate()
experiment.data_paths = all_paths
experiment.predictors = [
    PredictorSimple,
    PredictorNetwork,
    PredictorSimpleAnnotation,
    PredictorNetworkAnnotation,
    PredictorMarvin
    ]
experiment.data_paths = [
    #f"{wd}/in/images/dataset/2021_10_27_FRET_T001_Fret_Turquoise.tif"
    f"{wd}/in/images/working/testetst_Image8_edited_.ome.tif",
    #f"{wd}/in/images/Image_8.czi",
    #f"{wd}/in/images/11_10_21_48h_H2BmCherryB16F10OVA_CD4cytcells_bsAb_endpoint.lif",
    #f"{wd}/in/images/rFUNC.nii",
    ]
experiment.out_path = out
experiment.dataset = Dataset(name="Dataset1", samples={})
experiment.schema = ome_xsd
experiment.should_predict = "maybe"
experiment.reps = 1
experiment.evaluators = [OMEEvaluator]
experiment.run()

javabridge.kill_vm()
