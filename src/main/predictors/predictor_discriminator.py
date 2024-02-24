from src.main.predictors.predictor_template import PredictorTemplate
from src.main.assistants.assistant_DiscriminatorDave import DiscriminatorDave
import time


class DiscriminatorPredictor(PredictorTemplate):
    """
    This class implements only the discriminator assistant to predict the overhead metadata.
    """

    def __init__(self, path_to_raw_metadata=None, path_to_ome_starting_point=None, ome_xsd_path=None, out_path=None):
        super().__init__(path_to_raw_metadata=path_to_raw_metadata,
                         path_to_ome_starting_point=path_to_ome_starting_point,
                         ome_xsd_path=ome_xsd_path,
                         out_path=out_path)
        discriminator_dave = DiscriminatorDave(ome_xsd_path, self.client)
        self.assistant_id_path = "src/main/assistant_ids/" + discriminator_dave.name + "_assistant_id.txt"
        self.assistant = discriminator_dave.create_assistant(assistant_id_path=None)

    def predict(self):
        """
        Predict the OME XML from the raw metadata
        """

        print("- - - Generating Thread - - -")
        self.init_thread()

        print("- - - Generating Prompt - - -")
        full_message = "The starting points is:\n" + self.ome_starting_point + "\n\n" + "The raw data is: \n" + self.raw_metadata
        self.generate_message(msg=full_message)

        print("- - - Predicting OME XML - - -")
        self.run_message()

        print("- - - Exporting OME XML - - -")
        self.export_ome_xml()

    def run_message(self):  # TODO: Change name
        """
        Predict the OME XML from the raw metadata
        """

        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )

        while self.run.status == "in_progress" or self.run.status == "queued":
            print("Polling for run completion...")
            print(self.run.status)
            self.run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id
            )

            time.sleep(5)

        print(self.run.status)
        if self.run.status == "failed":
            print("Run failed")
            print(self.run)
            return None

        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        self.response = messages.data[0].content[0].text.value
        self.export_ome_xml()
