import time
from src.main.assistants.assistant_MissingMyrte import MissingMyrte
from src.main.assistants.assistant_MappingMargarete import MappingMargarete
from src.main.assistants.assistant_DiscriminatorDave import DiscriminatorDave
from src.main.assistants.assistant_TrashTimothy import TrashTimothy
from src.main.assistants.assistant_TargetTorben import TargetTorben
from src.main.assistants.assistant_ValidationVeronika import ValidationVeronika
from src.main.predictors.predictor_template import PredictorTemplate


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
        # timothy = TrashTimothy(ome_xsd_path, self.client)
        # self.assistant_timothy = timothy.create_assistant()
        veronika = ValidationVeronika(ome_xsd_path, self.client)
        self.assistant_veronika = veronika.create_assistant()
        self.conversation = {}
        self.thread = self.client.beta.threads.create()
        self.functions = {"validate_ome_xml": [self.validate]}

    def predict(self):
        """
        Predict the OME XML
        :return:
        """
        self.hierachical_planning()
        # self.export_ome_xml()
        self.export_convo()

    def hierachical_planning(self):
        """
        Run the message
        """
        # --------------------------------------------------------------------------------------------------------------
        # 1. Run the discriminator assistant to split the metadata into the contained and missing parts
        # --------------------------------------------------------------------------------------------------------------
        print("- - - Prompting Dave - - -")
        dave_prompt = ("The starting points is:\n" + self.ome_starting_point + "\n\n" + "The raw data is: \n" +
                       self.raw_metadata)
        self.conversation["Dave Prompt"] = self.assistant_dave.instructions + "\n" + dave_prompt

        # dave_out = self.run_assistant(self.assistant_dave, dave_prompt)
        with open("/in/assistant_outputs/dave_example_response.txt", "r") as f:
            dave_out = f.read()
            self.conversation["Dave Response"] = dave_out

        # --------------------------------------------------------------------------------------------------------------
        # 2. Run the missing assistant to split the missing metadata into mapping-issues and the target-issues parts
        # --------------------------------------------------------------------------------------------------------------
        print("- - - Prompting Myrte - - -")
        myrte_prompt = self.assistant_dave.instructions + "\n" "Here is the raw metadata: \n" + dave_out
        self.conversation["Myrte Prompt"] = self.assistant_myrte.instructions + "\n" + myrte_prompt
        # myrte_out = self.run_assistant(self.assistant_myrte, myrte_prompt)
        with open("/in/assistant_outputs/myrte_example_response.txt", "r") as f:
            myrte_out = f.read()
            self.conversation["Myrte Response"] = myrte_out
        mapping_issues, target_issues = myrte_out.split("- - -")

        # --------------------------------------------------------------------------------------------------------------
        # 3.(a) Run the mapping assistant to map the mapping issue metadata to the ome xml
        # --------------------------------------------------------------------------------------------------------------
        print("- - - Prompting Margarete - - -")
        margarete_prompt = ("The raw data is: \n" + mapping_issues + "\n\n" + "The OME XML is:\n" +
                            self.ome_starting_point)
        self.conversation["Margarete Prompt"] = self.assistant_margarete.instructions + "\n" + margarete_prompt
        # margarete_out = self.run_assistant(self.assistant_margarete, self.conversation["Margarete Prompt"])
        with open("/in/assistant_outputs/margarete_example_response.txt", "r") as f:
            margarete_out = f.read()
        self.conversation["Margarete Response"] = margarete_out

        # --------------------------------------------------------------------------------------------------------------
        # 3.(b) Run the target assistant to map the target issue to some kind of ontology and then to the ome xml
        # --------------------------------------------------------------------------------------------------------------
        print("- - - Prompting Torben - - -")
        torben_prompt = "The raw data is: \n" + target_issues + "\n\n" + "The OME XML is:\n" + margarete_out
        self.conversation["Torben Prompt"] = self.assistant_torben.instructions + "\n" + torben_prompt
        # torben_out = self.run_assistant(self.assistant_torben, torben_prompt)
        with open("/in/assistant_outputs/torben_example_response.txt", "r") as f:
            torben_out = f.read()
        self.conversation["Torben Response"] = torben_out

        # --------------------------------------------------------------------------------------------------------------
        # 4. Run the trash assistant to take the metadata that has not been added so far and add it as unstructured
        # --------------------------------------------------------------------------------------------------------------
        # out_4 = self.run_trash_timothy()

        # --------------------------------------------------------------------------------------------------------------
        # 5. Run the validation assistant to validate the OME XML
        # --------------------------------------------------------------------------------------------------------------
        print("- - - Prompting Veronika - - -")
        error = self.validate(torben_out)
        if error:
            veronika_prompt = "The OME XML is:\n" + torben_out + "\n" + "Validator Error message:\n" + str(error) + "\n"
            self.conversation["Veronika Prompt"] = self.assistant_veronika.instructions + "\n" + veronika_prompt
            self.conversation["Veronika Response"] = ""
            self.export_convo()
            # veronika_out = self.run_assistant(self.assistant_veronika, veronika_prompt)
            with open("/in/assistant_outputs/veronika_example_response.txt", "r") as f:
                veronika_out = f.read()
            self.conversation["Veronika Response"] += veronika_out

    def run_assistant(self, assistant, msg, thread=None):
        """
        Run the assistant
        :param thread:
        :param assistant:
        :param msg:
        :return:
        """
        if thread is None:
            thread = self.thread

        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=msg
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        while run.status in ['queued', 'in_progress', 'cancelling']:
            print("Polling for run completion...")
            print(run.status)
            time.sleep(2)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            if run.status == "requires_action":
                print(run.status)
                tool_outputs = []
                for call in run.required_action.submit_tool_outputs.tool_calls:
                    try:
                        out = self.functions[call.name][0](call)
                    except Exception as e:
                        out = "Error: " + str(e)

                    tool_outputs += {"tool_call_id": call.id, "output": out}

                run = self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )

        print(run.status)

        if run.status == "failed":
            print("Run failed")
            print(run)
            return None

        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        messages = messages.data[0].content[0].text.value

        return messages

        if assistant == self.assistant_veronika:
            error = self.validate(messages)
            print(error)
            if error:
                self.conversation["Veronika Response"] += messages + "\n" + "Validator Error message:\n" + error + "\n"

                messages = self.run_assistant(assistant, self.conversation["Veronika Response"])

        return messages

    def export_convo(self):
        with open(self.out_path + "convo.txt", "w") as f:
            for k in self.conversation:
                f.write(str(k) + "\n" + "\n" + self.conversation[k] + "\n" + "\n")
