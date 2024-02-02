import xmlschema
from predictors.predictor_simple import SimplePredictor


ome_schema_path = '/home/aaron/PycharmProjects/MetaGPT/raw_data/ome_xsd.txt'
ome_starting_point_path = "/home/aaron/PycharmProjects/MetaGPT/raw_data/image8_start_point.ome.xml"
raw_meta_path = "/home/aaron/PycharmProjects/MetaGPT/raw_data/raw_Metadata_Image8.txt"
out = "/home/aaron/PycharmProjects/MetaGPT/out/ome_xml.ome.xml"

myPredictor = SimplePredictor(path_to_raw_metadata=raw_meta_path,
                              path_to_ome_starting_point=ome_starting_point_path,
                              ome_xsd_path=ome_schema_path,
                              out_path=out)

myPredictor.predict()