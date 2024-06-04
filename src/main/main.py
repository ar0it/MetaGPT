"""
Main file for the MetaGPT project. This file runs the entire experiment.
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
from src.main.BioformatsReader import get_omexml_metadata
from src.main.BioformatsReader import get_raw_metadata
import os
import bioformats
import javabridge
# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------
wd = os.getcwd() + "/../.."
ome_schema_path = f'{wd}/in/schema/ome_xsd.txt'
raw_meta_path = f"{wd}/in/metadata/raw_Metadata_Image8.txt"
network_paths = [f"{wd}/out/assistant_outputs/veronika_example_response.txt"]
gt_paths = [f"{wd}/in/images/Image_12.czi",
            f"{wd}/in/images/Image_8.czi",
            f"{wd}/in/images/testetst_Image8_edited_.ome.tif"
            ]
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
for path in gt_paths:
    # ------------------------------------------------------------------------------------------------------------------
    # Bioformats
    # ------------------------------------------------------------------------------------------------------------------
    out_bioformats = get_omexml_metadata(path=path)
    raw_meta = get_raw_metadata(path=path)
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
    #melancholic_marvin = MelancholicMarvin()
    #out_marvin = melancholic_marvin.assistant.say(f"Here is the raw metadata {raw_meta} for you to curate.")
    # marvin_path = ""
    # with open(marvin_path, "r") as f:
    #     out_marvin = f.read()
    # ------------------------------------------------------------------------------------------------------------------
    # Structured Agent Network
    # ------------------------------------------------------------------------------------------------------------------
    #with open(path, "r") as f:
    #    out_network = f.read()
    network_sample = Sample(name=name,
                            metadata_str=out_bioformats,
                            method="Network",
                            format=format)
    experiment.add_sample(network_sample)
    # ------------------------------------------------------------------------------------------------------------------
    # Agent Swarm
    # ------------------------------------------------------------------------------------------------------------------
    #out_swarm = None

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation Pipeline
# ----------------------------------------------------------------------------------------------------------------------
ome_eval = OMEEvaluator(schema=ome_xsd,
                        experiment=experiment,
                        out_path=out)
javabridge.kill_vm()


