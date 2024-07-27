"""
This module contains the ExperimentTemplate class, which defines an experiment object
that can be used to run experiments. The experiment defines the dataset, predictors,
evaluators, and other parameters necessary for running experiments on OME XML metadata.
"""

import os
import time
from typing import List, Union, Optional

from ome_types import from_xml

from metagpt.utils.DataClasses import Dataset, Sample
from metagpt.utils import utils, BioformatsReader
from metagpt.distorters.distorter_template import DistorterTemplate

class ExperimentTemplate:
    """
    The ExperimentTemplate class defines an experiment object that can be used to run experiments.
    It encapsulates the dataset, predictors, evaluators, and other experiment parameters.
    """

    def __init__(self) -> None:
        """Initialize the ExperimentTemplate with default values."""
        self.data_paths: List[str] = []
        self.predictors: List[callable] = []
        self.evaluators: List[callable] = []
        self.reps: int = 1
        self.create_report: bool = True
        self.dataset = Dataset()
        self.should_predict: Union[str, List[str]] = "maybe"
        self.out_path: Optional[str] = None
        self.out_path_experiment: Optional[str] = None
        self.schema: Optional[str] = None
        self.model: str = "gpt-4o-mini"
        self.time: str = time.strftime("%Y%m%d-%H%M%S")

    def run(self) -> None:
        """
        Run the experiment.
        This method processes each image in the data_paths, generates metadata,
        and runs predictors and evaluators.
        """
        for i in range(self.reps):
            print(f"---File {i + 1}/{self.reps}---")
            self._setup_experiment_output_path(i)
            
            for path in self.data_paths:
                self._process_image(path, i)

        self._run_evaluators()

    def _setup_experiment_output_path(self, rep_index: int) -> None:
        """Set up the output path for the current experiment repetition."""
        if self.out_path_experiment is None:
            self.out_path_experiment = os.path.join(self.out_path, f"experiment_{self.time}_{rep_index}")
        os.makedirs(self.out_path_experiment, exist_ok=True)

    def _process_image(self, path: str, rep_index: int) -> None:
        """Process a single image file."""
        print("-" * 60)
        print(f"Processing image: {path}")
        print("-" * 60)
        print("-" * 10 + "Bioformats" + "-" * 10)

        file_name, data_format = self._get_file_info(path)
        out_bioformats, processing_time = self._get_bioformats_metadata(path)
        fake_meta = self._generate_distorted_metadata(out_bioformats, file_name)

        bio_sample = self._create_bioformats_sample(file_name, out_bioformats, data_format, processing_time, rep_index)
        self.dataset.add_sample(bio_sample)

        for predictor_index, predictor in enumerate(self.predictors):
            self._run_predictor(predictor, predictor_index, fake_meta, file_name, data_format, out_bioformats, rep_index)

    def _get_file_info(self, path: str) -> tuple:
        """Extract file name and format from the given path."""
        file_name = os.path.splitext(os.path.basename(path))[0]
        data_format = os.path.splitext(path)[1][1:]  # Remove the leading dot
        return file_name, data_format

    def _get_bioformats_metadata(self, path: str) -> tuple:
        """Get OME XML metadata using BioformatsReader and measure processing time."""
        start_time = time.time()
        out_bioformats = BioformatsReader.get_omexml_metadata(path=path)
        end_time = time.time()
        return out_bioformats, end_time - start_time

    def _generate_distorted_metadata(self, out_bioformats: str, file_name: str) -> dict:
        """Generate distorted metadata using DistorterTemplate."""
        dt = DistorterTemplate()
        dt.model = self.model
        distorted_path = os.path.join(self.out_path_experiment, "distorted_data", f"{file_name}_distorted.json")
        return dt.distort(out_bioformats, out_path=distorted_path, should_pred="maybe")

    def _create_bioformats_sample(self, file_name: str, metadata: str, data_format: str, processing_time: float, rep_index: int) -> Sample:
        """Create a Sample object for Bioformats metadata."""
        return Sample(
            file_name=file_name,
            metadata_str=metadata,
            method="Bioformats",
            format=data_format,
            time=processing_time,
            name=f"{file_name}_Bioformats_{rep_index}",
            index=rep_index,
            attempts=1
        )

    def _run_predictor(self, predictor: callable, predictor_index: int, fake_meta: dict, file_name: str, data_format: str, start_point: str, rep_index: int) -> None:
        """Run a single predictor on the distorted metadata."""
        should_predict = self.should_predict[predictor_index] if isinstance(self.should_predict, list) else self.should_predict
        utils.make_prediction(
            predictor=predictor,
            in_data=fake_meta,
            dataset=self.dataset,
            file_name=file_name,
            should_predict=should_predict,
            data_format=data_format,
            start_point=start_point,
            index=rep_index,
            model=self.model,
            out_path=self.out_path_experiment,
        )

    def _run_evaluators(self) -> None:
        """Run all evaluators on the dataset."""
        for evaluator in self.evaluators:
            evaluator(
                schema=self.schema,
                dataset=self.dataset,
                out_path=self.out_path_experiment
            ).report()