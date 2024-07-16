from metagpt.predictors.predictor_template import PredictorTemplate
from marvin.beta.applications import Application
class PredictorMarvin(PredictorTemplate):
    """
    
    """
    def __init__(self, raw_meta:str) -> None:
        super().__init__()
        self.attempt = 0
        self.raw_metadata = raw_meta
        self.prompt = """
        blah blah blah
        """
        self.instructions = """

        """
        self.state = None
    def predict(self):
        """
        Predict the image annotations using the Marvin model.
        :param image_path: The path to the image.
        :return: The predicted image annotations.
        """
        app = Application(
            state=self.state,
            name=self.__class__.__name__,
            instructions=self.instructions)
        
        app.say(self.prompt)

        return app.state