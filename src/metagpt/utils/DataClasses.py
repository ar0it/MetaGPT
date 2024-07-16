"""
Data classes for the metagpt package.
"""
from dataclasses import dataclass
from xml.etree import ElementTree as ET
from typing import Optional
from pydantic import Field

@dataclass
class Sample:
    name: str = None
    metadata_str: str = None# the metadata as xml string
    method: str = None
    metadata_xml: Optional[ET.Element] = None  # the metadata as xml object
    cost: Optional[float] = None # the cost in $
    paths: Optional[list[str]] = None
    time: Optional[float] = None
    format: Optional[str] = None
    attempts: Optional[float] = None
    iter: Optional[int] = None


@dataclass
class Dataset:
    name: str = None
    samples: dict[str:Sample] = Field(default_factory=dict)
    cost: Optional[float] = None
    time: Optional[float] = None

    def add_sample(self, sample: Sample):
        self.samples[f"{sample.name}_{sample.method}"] = sample
        if sample.cost:
            self.cost += sample.cost
        if sample.time:
            self.time += sample.time


