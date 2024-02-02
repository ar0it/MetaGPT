from predictor_template import PredictorTemplate


class SimplePredictor(PredictorTemplate):
    """
    The most simple predictor that uses one assistant to predict the OME XML from the raw metadata in one step.
    """

    def __init__(self, config, path_to_raw_metadata):
        super().__init__(path_to_raw_metadata)
        self.config = config

    def predict(self, data):
        return data
