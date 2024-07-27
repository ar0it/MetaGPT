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
from metagpt.predictors.predictor_state import PredictorState
from metagpt.predictors.predictor_simple import PredictorSimple
from metagpt.predictors.predictor_state_tree import PredictorStateTree
from metagpt.experiments.experiment_template import ExperimentTemplate
import importlib
import sys
import metagpt.utils.utils as utils
import bioformats.logback
from metagpt.evaluators.evaluator_template import EvaluatorTemplate
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
import datetime
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
    PredictorState,
    ]
datafolder_path ="/home/aaron/Documents/Projects/MetaGPT/in/images/working/"
# list of all the FILES from that folder
experiment.data_paths = [datafolder_path + f for f in os.listdir(datafolder_path) if os.path.isfile(os.path.join(datafolder_path, f))]
experiment.out_path = out
experiment.dataset = Dataset(name="Dataset1", samples={})
experiment.schema = ome_xsd
experiment.should_predict = "maybe"
experiment.reps = 1
experiment.time = datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")
experiment.evaluators = [EvaluatorTemplate]
experiment.out_path_experiment = "/home/aaron/Documents/Projects/MetaGPT/out/experiment_2024-07-27T11-44-01-418605_0/"
experiment.run()

javabridge.kill_vm()
