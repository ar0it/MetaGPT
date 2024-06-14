"""
Main file for the MetaGPT project. This file runs the entire experiment.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------
#from predictors.predictor_curation_swarm import CurationSwarm
#from predictors.predictor_simple import SimplePredictor
#from assistants.assistant_MelancholicMarvin import MelancholicMarvin
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
import sys
from contextlib import contextmanager

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
    #f"{wd}/in/images/Image_12.czi",
    #f"{wd}/in/images/Image_8.czi",
    #f"{wd}/in/images/testetst_Image8_edited_.ome.tif",
    f"{wd}/in/images/11_10_21_48h_H2BmCherryB16F10OVA_CD4cytcells_bsAb_endpoint.lif",
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
for path in gt_paths:
    print("Processing image:")
    print(path)
    # ------------------------------------------------------------------------------------------------------------------
    # Bioformats
    # ------------------------------------------------------------------------------------------------------------------
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
    # Marvin
    # ------------------------------------------------------------------------------------------------------------------
    #melancholic_marvin = MelancholicMarvin()
    #out_marvin = melancholic_marvin.assistant.say(f"Here is the raw metadata {raw_meta} for you to curate.")
    # save the output to file
    #with open(f"{out}/assistant_outputs/{name}_marvin.txt", "w") as f:
    #    f.write(out_marvin)
    marvin_sample = Sample(name=name,
                           metadata_str=out_bioformats,
                           method="Marvin",
                           format=format)
    experiment.add_sample(marvin_sample)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Structured Agent Network
    # ------------------------------------------------------------------------------------------------------------------
    # curation_network = CurationNetwork()
    # out_network = curation_network.predict(raw_meta)
    # save the output to file
    # with open(f"{out}/assistant_outputs/{name}_network.txt", "w") as f:
    #     f.write(out_network)
    network_sample = Sample(name=name,
                            metadata_str=out_bioformats,
                            method="Network",
                            format=format)
    experiment.add_sample(network_sample)

    # ------------------------------------------------------------------------------------------------------------------
    # Agent Swarm
    # ------------------------------------------------------------------------------------------------------------------
    #curation_swarm = CurationSwarm()
    #out_swarm = curation_swarm.predict(raw_meta)
    # save the output to file
    #with open(f"{out}/assistant_outputs/{name}_swarm.txt", "w") as f:
    #    f.write(out_swarm)
    swarm_sample = Sample(name=name,
                          metadata_str=out_bioformats,
                          method="Swarm",
                          format=format)
    experiment.add_sample(swarm_sample)

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation Pipeline
# ----------------------------------------------------------------------------------------------------------------------
ome_eval = OMEEvaluator(schema=ome_xsd,
                        experiment=experiment,
                        out_path=out)
javabridge.kill_vm()
