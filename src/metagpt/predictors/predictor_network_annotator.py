"""
This module contains the PredictorNetworkAnnotation class, which uses a network
of predictors to process and annotate metadata for microscopy images.
"""

from typing import Optional, Tuple, Any

from metagpt.predictors.predictor_template import PredictorTemplate
from metagpt.predictors.predictor_simple_annotator import PredictorSimpleAnnotation
from metagpt.predictors.predictor_seperator import PredictorSeperator

class PredictorNetworkAnnotation(PredictorTemplate):
    """
    A predictor class that uses two assistants to process and annotate metadata.

    This predictor approach uses two assistants:
    1. A separator to split the raw metadata into already contained and new metadata.
    2. An annotator to predict structured annotations from the new metadata.
    """
    
    def __init__(self, raw_meta: str) -> None:
        """
        Initialize the PredictorNetworkAnnotation.

        Args:
            raw_meta (str): The raw metadata to be processed and annotated.
        """
        super().__init__()
        self.raw_metadata = raw_meta
        self.assistants = [
            "predictor_seperator",
            "predictor_simple_annotator"
        ]

    def predict(self) -> Tuple[Optional[Any], float, float]:
        """
        Predict structured annotations based on the raw metadata.

        This method uses two predictors in sequence:
        1. PredictorSeperator to split the metadata.
        2. PredictorSimpleAnnotation to generate annotations.

        Returns:
            Tuple[Optional[Any], float, float]: 
                - The predicted annotations (or None if prediction fails)
                - The total cost of the prediction
                - The total number of attempts made
        """
        print(f"Predicting for {self.name}, attempt: {self.attempts}")
        
        # Step 1: Separate the metadata
        sep_response, sep_cost, sep_attempts = PredictorSeperator(
            f"Here is the raw metadata \n{self.raw_metadata}").predict()
        
        if sep_response is None:
            print("Separation failed. Aborting prediction.")
            return None, sep_cost, sep_attempts
        
        response_annot, response_ome = sep_response

        # Step 2: Generate annotations
        response, pred_cost, pred_attempts = PredictorSimpleAnnotation(
            f"Here is the preselected raw metadata \n{response_annot}").predict()
        
        total_attempts = sep_attempts + pred_attempts
        self.add_attempts(total_attempts)

        total_cost = sep_cost + pred_cost
        
        return response, total_cost, self.attempts