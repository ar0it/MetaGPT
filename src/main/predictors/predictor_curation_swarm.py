from ..predictors.predictor_template import PredictorTemplate


class CurationSwarm(PredictorTemplate):
    """
    This class implements a swarm of AI assistants that work together to curate the metadata.
    """

    def __init__(self, path_to_raw_metadata=None, path_to_ome_starting_point=None, ome_xsd_path=None, out_path=None):
        super().__init__(path_to_raw_metadata=path_to_raw_metadata,
                         path_to_ome_starting_point=path_to_ome_starting_point,
                         ome_xsd_path=ome_xsd_path,
                         out_path=out_path)

    def run_message(self):
        """
        Run the message
        """
        self.run_descriminator_dave()
        self.run_step_by_step_sally()
