from src.main.predictors.predictor_template import PredictorTemplate
from src.main.assistants.assistant_AllAloneAugust import AllAloneAugust


class SimplePredictor(PredictorTemplate):
    """
    The most simple predictor that uses one assistant to predict the OME XML from the raw metadata in one step.
    """

    def __init__(self,
                 path_to_raw_metadata=None,
                 path_to_ome_starting_point=None,
                 ome_xsd_path=None,
                 out_path=None):
        super().__init__(path_to_raw_metadata=path_to_raw_metadata,
                         path_to_ome_starting_point=path_to_ome_starting_point,
                         ome_xsd_path=ome_xsd_path,
                         out_path=out_path)

        self.assistant = AllAloneAugust(ome_xsd_path, self.client)
