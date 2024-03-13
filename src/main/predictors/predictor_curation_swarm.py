from ..predictors.predictor_template import PredictorTemplate
import time
from src.main.assistants.assistant_MissingMyrte import MissingMyrte
from src.main.assistants.assistant_MappingMargarete import MappingMargarete
from src.main.assistants.assistant_DiscriminatorDave import DiscriminatorDave
from src.main.assistants.assistant_TrashTimothy import TrashTimothy
from src.main.assistants.assistant_TargetTorben import TargetTorben



class CurationSwarm(PredictorTemplate):
    """
    This class implements a swarm of AI assistants that work together to curate the metadata.
    """

    def __init__(self, path_to_raw_metadata=None, path_to_ome_starting_point=None, ome_xsd_path=None, out_path=None):
        super().__init__(path_to_raw_metadata=path_to_raw_metadata,
                         path_to_ome_starting_point=path_to_ome_starting_point,
                         ome_xsd_path=ome_xsd_path,
                         out_path=out_path)

        self.ome_xml = None
        self.raw_metadata = self.read_raw_metadata()
        dave = DiscriminatorDave(ome_xsd_path, self.client)
        self.assistant_dave = dave.create_assistant()
        myrte = MissingMyrte(ome_xsd_path, self.client)
        self.assistant_myrte = myrte.create_assistant()
        margarete = MappingMargarete(ome_xsd_path, self.client)
        self.assistant_margarete = margarete.create_assistant()
        torben = TargetTorben(ome_xsd_path, self.client)
        self.assistant_torben = torben.create_assistant()
        timothy = TrashTimothy(ome_xsd_path, self.client)
        self.assistant_timothy = timothy.create_assistant()
        self.conversation = []

    def predict(self):
        """
        Predict the OME XML
        :return:
        """
        self.hierachical_planning()
        self.export_ome_xml()
        self.export_convo()

    def hierachical_planning(self):
        """
        Run the message
        """
        # 1. Run the discriminator assistant to split the metadata into the contained and missing parts
        print("- - - Prompting Dave - - -")
        dave_prompt = "The starting points is:\n" + self.ome_starting_point + "\n\n" + "The raw data is: \n" + self.raw_metadata
        out_1 = self.run_assistant(self.assistant_dave, dave_prompt)

        # 2. Run the missing assistant to split the missing metadata into mapping-issues and the target-issues parts
        print("- - - Prompting Myrte - - -")
        myrte_prompt = out_1
        out_2 = self.run_assistant(self.assistant_myrte, myrte_prompt)
        mapping_issues = out_2[0]
        target_issues = out_2[1]

        # 3.(a) Run the mapping assistant to map the mapping issue metadata to the ome xml
        print("- - - Prompting Margarete - - -")
        out_3a = self.run_assistant(self.assistant_margarete, mapping_issues)

        # 3.(b) Run the target assistant to map the target issue to some kind of ontology and then to the ome xml
        # out_3b = self.run_target_torben(target_issues)

        # 4. Run the trash assistant to take the metadata that has not been added so far and add it as unstructured
        # metadata
        # out_4 = self.run_trash_timothy()

    def run_assistant(self, assistant, msg):
        thread = self.client.beta.threads.create()

        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=msg
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        while run.status == "in_progress" or run.status == "queued":
            print("Polling for run completion...")
            print(run.status)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            time.sleep(5)

        print(run.status)
        if run.status == "failed":
            print("Run failed")
            print(run)
            return None

        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        self.conversation.append(messages)
        return messages.data[0].content[0].text.value

    def export_convo(self):
        with open(self.out_path + "convo.txt", "w") as f:
            for message in self.conversation:
                f.write(message.data[0].content[0].text.value + "\n" + "\n")
