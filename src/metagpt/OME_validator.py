import os
import openai
import autogen
import argparse
from openai import OpenAI
import xml.etree.ElementTree as ET
import time


class OME_Validator:
    """
    Class to validate the OME XML prediction.
    """

    def __init__(self):
        self.ome_xml = None
        self.ome_xsd = None
        self.errors = []
        self.new_message = None

    def validate(self):
        """
        Validate the OME XML prediction
        """
        pass

    def write_new_message(self):
        """
        Write the new message to the assistant
        """
        pass
