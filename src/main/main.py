from OME_evaluator import OMEEvaluator
from OME_predictor import XMLPredictor
import argparse
import os
import xmlschema
import xml.etree.ElementTree as ET


my_schema = xmlschema.XMLSchema('/home/aaron/PycharmProjects/MetaGPT/raw_data/ome.xsd')
ome_xml_path = "/home/aaron/PycharmProjects/MetaGPT/out/ome_xml.ome.xml"
with open(ome_xml_path, "r") as f:
    ome_xml = f.read()
    ome_xml = ome_xml.split("</OME>")[0].split("<OME")[1]
    ome_xml = "<OME" + ome_xml + "</OME>"
    print(ome_xml)

try:
    my_schema.validate(ome_xml)

except Exception as e:
    print(e)
    print("Validation failed")


# Predictor = XMLPredictor(path_to_raw_metadata="/home/aaron/PycharmProjects/MetaGPT/raw_Metadata_Image8.txt")
# Evaluator = OMEEvaluator(path_to_raw_metadata="/ground_truth/raw_Metadata_Image8.txt")
