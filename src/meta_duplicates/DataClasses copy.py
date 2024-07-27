"""
Data classes for the metagpt package.

This module defines the core data structures used in the metagpt package,
including Sample and Dataset classes.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pydantic import Field
from ome_types import OME


@dataclass
class Sample:
    """
    Represents a single sample in the dataset.

    Attributes:
        format (str): The format of the sample.
        attempts (float): The number of attempts made for this sample.
        index (int): The index of the sample.
        file_name (str): The name of the file associated with this sample.
        name (Optional[str]): The name of the sample. Defaults to None.
        metadata_str (Optional[str]): The metadata as an XML string. Defaults to None.
        method (Optional[str]): The method used for this sample. Defaults to None.
        metadata_xml (OME): The metadata as an OME object. Defaults to an empty OME object.
        cost (Optional[float]): The cost in dollars. Defaults to None.
        paths (Optional[List[str]]): List of paths associated with this sample. Defaults to None.
        time (Optional[float]): Time taken for processing this sample. Defaults to None.
        gpt_model (Optional[str]): The GPT model used, if applicable. Defaults to None.
    """

    format: str
    attempts: float
    index: int
    file_name: str
    name: Optional[str] = None
    metadata_str: Optional[str] = None
    method: Optional[str] = None
    metadata_xml: OME = Field(default_factory=OME, description="The metadata as an OME object")
    cost: Optional[float] = None
    paths: Optional[List[str]] = None
    time: Optional[float] = None
    gpt_model: Optional[str] = None


@dataclass
class Dataset:
    """
    Represents a collection of samples.

    Attributes:
        name (Optional[str]): The name of the dataset. Defaults to None.
        samples (Dict[str, Sample]): A dictionary of samples, keyed by their names.
        cost (float): The total cost of all samples in the dataset. Defaults to 0.
        time (float): The total time taken for all samples in the dataset. Defaults to 0.
    """

    name: Optional[str] = None
    samples: Dict[str, Sample] = field(default_factory=dict)
    cost: float = 0
    time: float = 0

    def add_sample(self, sample: Sample) -> None:
        """
        Add a sample to the dataset.

        Args:
            sample (Sample): The sample to add to the dataset.

        This method also updates the total cost and time of the dataset.
        """
        self.samples[sample.name] = sample
        if sample.cost is not None:
            self.cost += sample.cost
        if sample.time is not None:
            self.time += sample.time