"""
Data classes for the metagpt package.
"""
from dataclasses import dataclass
from xml.etree import ElementTree as ET
from typing import Optional
from pydantic import Field
from ome_types import OME


@dataclass
class Sample:
    name: str = None
    metadata_str: str = None  # the metadata as xml string
    method: str = None
    metadata_xml: OME = Field(default_factory=OME, description="The metadata as an OME object")
    cost: Optional[float] = None # the cost in $
    paths: Optional[list[str]] = None
    time: Optional[float] = None
    format: Optional[str] = None
    attempts: Optional[float] = None
    iter: Optional[int] = None
    gpt_model: Optional[str] = None

@dataclass
class Dataset:
    name: str = None
    samples: dict[str:Sample] = Field(default_factory=dict)
    cost: Optional[float] = 0
    time: Optional[float] = 0

    def add_sample(self, sample: Sample):
        self.samples[f"{sample.name}_{sample.method}"] = sample
        if sample.cost != None:
            self.cost += sample.cost
        if sample.time != None:
            self.time += sample.time


