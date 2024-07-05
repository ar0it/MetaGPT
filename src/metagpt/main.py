"""
Main file for the MetaGPT project. This file runs the entire experiment.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------
#from predictors.predictor_curation_swarm import CurationSwarm
#from predictors.predictor_simple import SimplePredictor
#from assistants.assistant_MelancholicMarvin import MelancholicMarvin
import importlib
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("metagpt", "/home/aaron/Documents/Projects/MetaGPT/src/metagpt/utils.py")
utils = importlib.util.module_from_spec(spec)
sys.modules["metagpt"] = utils
spec.loader.exec_module(utils)

spec = importlib.util.spec_from_file_location("metagpt", "/home/aaron/Documents/Projects/MetaGPT/src/metagpt/BioformatsReader.py")
BioformatsReader = importlib.util.module_from_spec(spec)
sys.modules["metagpt"] = BioformatsReader
spec.loader.exec_module(BioformatsReader)

spec = importlib.util.spec_from_file_location("predictors", "/home/aaron/Documents/Projects/MetaGPT/src/metagpt/predictors/predictor_xml_annotation.py")
predictors = importlib.util.module_from_spec(spec)
sys.modules["predictors"] = predictors
spec.loader.exec_module(predictors)

spec = importlib.util.spec_from_file_location("predictors2", "/home/aaron/Documents/Projects/MetaGPT/src/metagpt/predictors/predictor_annotation_net.py")
predictors2 = importlib.util.module_from_spec(spec)
sys.modules["predictors2"] = predictors2
spec.loader.exec_module(predictors2)

spec = importlib.util.spec_from_file_location("predictors3", "/home/aaron/Documents/Projects/MetaGPT/src/metagpt/predictors/predictor_simple.py")
predictors3 = importlib.util.module_from_spec(spec)
sys.modules["predictors3"] = predictors3
spec.loader.exec_module(predictors3)

spec = importlib.util.spec_from_file_location("predictors4", "/home/aaron/Documents/Projects/MetaGPT/src/metagpt/predictors/predictor_network.py")
predictors4 = importlib.util.module_from_spec(spec)
sys.modules["predictors4"] = predictors4
spec.loader.exec_module(predictors4)

#from src.metagpt.predictors.predictor_xml_annotation import PredictorXMLAnnotation
# from src.metagpt.utils import dict_to_xml_annotation
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
    f"{wd}/in/images/testetst_Image8_edited_.ome.tif",
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
for path in gt_paths:
    print("Processing image:")
    print(path)
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
    print("-"*10+"Simple Predictor"+"-"*10)
    #base_predictor = predictors3.Pred(str(tree_meta))
    #out_simple = base_predictor.predict()
    out_simple = out_bioformats
    # save the output to file
    with open(f"{out}/assistant_outputs/{name}_simple.txt", "w") as f:
        f.write(out_simple)
    simple_sample = Sample(name=name,
                           metadata_str=out_simple,
                           method="SimplePredictor",
                           format=format)
    experiment.add_sample(simple_sample)

    # ------------------------------------------------------------------------------------------------------------------
    # Marvin
    # ------------------------------------------------------------------------------------------------------------------
    print("-"*10+"Marvin"+"-"*10)
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
    # Network Predictor
    # ------------------------------------------------------------------------------------------------------------------
    print("-"*10+"Structured Agent Network"+"-"*10)
    network_predictor = predictors4.PredictorNetwork(str(tree_meta))
    out_network = network_predictor.predict()
    # save the output to file
    with open(f"{out}/assistant_outputs/{name}_network.txt", "w") as f:
         f.write(out_network)
    network_sample = Sample(name=name,
                            metadata_str=out_network,
                            method="Network",
                            format=format)
    experiment.add_sample(network_sample)

    # ------------------------------------------------------------------------------------------------------------------
    # Agent Graph
    # ------------------------------------------------------------------------------------------------------------------
    print("-"*10+"Agent Graph"+"-"*10)
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

    # ------------------------------------------------------------------------------------------------------------------
    # Simple Annotation Predictor
    # ------------------------------------------------------------------------------------------------------------------
    print("-"*10+"Simple Annotation Predictor"+"-"*10)
    #base_annotation_predictor = predictors.PredictorXMLAnnotation(str(tree_meta))
    out_annotation = out_bioformats #base_annotation_predictor.predict()
    # merge the annotation section with the ome xml
    #out_xml_annotation = utils.dict_to_xml_annotation(out_annotation)
    #ome_start_obj = ome_types.from_xml(out_bioformats)
    #ome_start_obj.structured_annotations.append(out_xml_annotation)
    #out_annotated = ome_types.to_xml(ome_start_obj)
    annotation_sample = Sample(name=name,
                            metadata_str=out_annotation,
                            method="Simple_Annotation",
                            format=format)
    experiment.add_sample(annotation_sample)

    # ------------------------------------------------------------------------------------------------------------------
    # Network Annotation Predictor
    # ------------------------------------------------------------------------------------------------------------------
    print("-"*10+"Network Annotation Predictor"+"-"*10)
    # net_annotation_predictor = predictors2.PredictorXMLAnnotationNet(str(tree_meta))
    out_annotation_net = out_bioformats #net_annotation_predictor.predict()
    # merge the annotation section with the ome xml
    #out_xml_annotation_net = utils.dict_to_xml_annotation(out_annotation_net)
    #ome_start_obj = ome_types.from_xml(out_bioformats)
    #ome_start_obj.structured_annotations.append(out_xml_annotation_net)
    #out_annotated_net = ome_types.to_xml(ome_start_obj)
    annotation_net_sample = Sample(name=name,
                            metadata_str=out_annotation_net,
                            method="Network_Annotation",
                            format=format)
    experiment.add_sample(annotation_net_sample)

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation Pipeline
# ----------------------------------------------------------------------------------------------------------------------
ome_eval = OMEEvaluator(schema=ome_xsd,
                        experiment=experiment,
                        out_path=out)
javabridge.kill_vm()
