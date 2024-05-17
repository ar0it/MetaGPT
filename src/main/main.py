"""
Main file for the MetaGPT project. This file will run the entire experiment.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------
from src.main.predictors.predictor_curation_swarm import CurationSwarm
from src.main.predictors.predictor_simple import SimplePredictor
from src.main.assistants.assistant_MelancholicMarvin import MelancholicMarvin
import OME_evaluator

# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------
ome_schema_path = '/home/aaron/PycharmProjects/MetaGPT/raw_data/ome_xsd.txt'
ome_starting_point_path = "/home/aaron/PycharmProjects/MetaGPT/raw_data/image8_start_point.ome.xml"
raw_meta_path = "/home/aaron/PycharmProjects/MetaGPT/raw_data/raw_Metadata_Image8.txt"
out = "/home/aaron/PycharmProjects/MetaGPT/out/"

# ----------------------------------------------------------------------------------------------------------------------
# Read the Data
# ----------------------------------------------------------------------------------------------------------------------
with open(ome_schema_path, "r") as f:
    ome_xsd = f.read()

with open(raw_meta_path, "r") as f:
    raw_meta = f.read()

# ----------------------------------------------------------------------------------------------------------------------
# Simple Predictor
# ----------------------------------------------------------------------------------------------------------------------
melancholic_marvin = MelancholicMarvin()
out_marvin = melancholic_marvin.assistant.say(f"Here is the raw metadata {raw_meta} for you to curate.")

# ----------------------------------------------------------------------------------------------------------------------
# Structured Agent Network
# ----------------------------------------------------------------------------------------------------------------------
out_network = None

# ----------------------------------------------------------------------------------------------------------------------
# Agent Swarm
# ----------------------------------------------------------------------------------------------------------------------
out_swarm = None

# ----------------------------------------------------------------------------------------------------------------------
# Bioformats
# ----------------------------------------------------------------------------------------------------------------------
out_bioformats = None

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation Pipeline
# ----------------------------------------------------------------------------------------------------------------------
ome_eval = OME_evaluator(schema=ome_xsd,
                         ground_truth=out_bioformats,
                         predicted=[out_marvin, out_network, out_swarm],
                         out_path=out)
