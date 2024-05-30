"""
Main file for the MetaGPT project. This file will run the entire experiment.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------
from src.main.predictors.predictor_curation_swarm import CurationSwarm
from src.main.predictors.predictor_simple import SimplePredictor
from src.main.assistants.assistant_MelancholicMarvin import MelancholicMarvin
from OME_evaluator import OMEEvaluator
from src.main.DataClasses import Sample
from src.main.DataClasses import Experiment
# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------
ome_schema_path = '/home/aaron/PycharmProjects/MetaGPT/raw_data/ome_xsd.txt'
ome_starting_point_path = "/out/image8_start_point.ome.xml"
raw_meta_path = "/out/raw_Metadata_Image8.txt"
out = "/home/aaron/PycharmProjects/MetaGPT/out/"

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
# Simple Predictor
# ----------------------------------------------------------------------------------------------------------------------
#melancholic_marvin = MelancholicMarvin()
#out_marvin = melancholic_marvin.assistant.say(f"Here is the raw metadata {raw_meta} for you to curate.")
# marvin_path = ""
# with open(marvin_path, "r") as f:
#     out_marvin = f.read()
# ----------------------------------------------------------------------------------------------------------------------
# Structured Agent Network
# ----------------------------------------------------------------------------------------------------------------------
network_path = "/home/aaron/PycharmProjects/MetaGPT/out/assistant_outputs/veronika_example_response.txt"
with open(network_path, "r") as f:
    out_network = f.read()

network_sample = Sample(name="Image8", metadata_str=out_network, method="Network")
experiment.add_sample(network_sample)
# ----------------------------------------------------------------------------------------------------------------------
# Agent Swarm
# ----------------------------------------------------------------------------------------------------------------------
#out_swarm = None

# ----------------------------------------------------------------------------------------------------------------------
# Bioformats
# ----------------------------------------------------------------------------------------------------------------------
gt_path = "/home/aaron/PycharmProjects/MetaGPT/out/testetst_Image8_edited_.ome.xml"
with open(gt_path, "r") as f:
    out_bioformats = f.read()

bio_sample = Sample(name="Image8", metadata_str=out_bioformats, method="Bioformats")
experiment.add_sample(bio_sample)
# ----------------------------------------------------------------------------------------------------------------------
# Evaluation Pipeline
# ----------------------------------------------------------------------------------------------------------------------
# TODO: make evaluator work with entire datasets?
ome_eval = OMEEvaluator(schema=ome_xsd,
                        experiment=experiment,
                        out_path=out)




