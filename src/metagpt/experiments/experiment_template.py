from metagpt.utils.DataClasses import Dataset, Sample
from metagpt.utils import utils, BioformatsReader
from metagpt.distorters.distorter_template import DistorterTemplate
import json
from ome_types import from_xml

class ExperimentTemplate:
    """
    The experiment template class defines an experiment object that can be used to run experiments.
    The experiment defines the dataset, the predictors, the evaluator and more.
    """
    def __init__(self) -> None:
        self.data_paths:list = []
        self.predictors:list = []
        self.reps:int = 1
        self.create_report:bool = True
        self.dataset = Dataset()
        self.should_predict = "maybe"
        self.evaluators:list = []
        self.out_path:str = None
        self.schema:str = None
        self.model:str = "gpt-4o-mini"

    def run(self):
        """
        Run the experiment
        """
        for i in range (self.reps):
            for path in self.data_paths:
                print("-"*60)
                print("Processing image:")
                print(path)
                print("-"*60)
                print("-"*10+"Bioformats"+"-"*10)
                out_bioformats = BioformatsReader.get_omexml_metadata(path=path) # the raw metadata as ome xml str
                #raw_meta = BioformatsReader.get_raw_metadata(path=path) # the raw metadata as dictionary of key value pairs
                #tree_meta = BioformatsReader.raw_to_tree(raw_meta) # the raw metadata as nested dictionary --> more compressed

                file_name = path.split("/")[-1].split(".")[0]
                data_format = path.split("/")[-1].split(".")[1]

                
                dt = DistorterTemplate()
                fake_meta = dt.distort(
                    out_bioformats,
                    out_path=self.out_path + "distorted_data/" + file_name + "_distorted.json",
                    should_pred="maybe") # the distorted metadata as dictionary of key value pairs

                bio_sample = Sample(name=file_name,
                                    metadata_str=out_bioformats,
                                    method="Bioformats",
                                    format=data_format)
                
                self.dataset.add_sample(bio_sample)

                for j, predictor in enumerate(self.predictors):
                    if isinstance(self.should_predict, list):
                        should_predict = self.should_predict[j]
                    else:
                        should_predict = self.should_predict
                    utils.make_prediction(
                        predictor=predictor,
                        in_data=fake_meta,
                        dataset=self.dataset,
                        name=file_name,
                        should_predict=should_predict,
                        data_format=data_format,
                        start_point=out_bioformats,
                        iter=i,
                        model=self.model
                    )
        
        for eval in self.evaluators:
            eval(
                schema=self.schema,
                dataset=self.dataset,
                out_path=self.out_path).report()