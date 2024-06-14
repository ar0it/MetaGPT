"""
Data classes for the metagpt package.
"""
from dataclasses import dataclass
from xml.etree import ElementTree as ET
from typing import Optional


@dataclass
class Sample:
    name: str
    metadata_str: str  # the metadata as xml string
    method: str
    metadata_xml: Optional[ET.Element] = None  # the metadata as xml object
    cost: Optional[float] = None
    paths: Optional[list[str]] = None
    time: Optional[float] = None
    format: Optional[str] = None


@dataclass
class Experiment:
    name: str
    samples: dict[str:Sample]
    cost: Optional[float] = None
    time: Optional[float] = None

    def add_sample(self, sample: Sample):
        self.samples[f"{sample.name}_{sample.method}"] = sample
        if sample.cost:
            self.cost += sample.cost
        if sample.time:
            self.time += sample.time


@dataclass
class Dataset:
    raw_meta_paths: list[str]
