from metagpt.assistants.assistant_template import AssistantTemplate
from pydantic import BaseModel, Field
import time


class AllAloneAugust(AssistantTemplate):
    """
    This assistant is the most basic assistant approach and attempts to translate the raw meta-data all alone :(.
    """

    def __init__(self, ome_xsd_path, client):
        super().__init__(ome_xsd_path, client)
        self.name = "AllAloneAugust"
        self.pre_prompt = (
            f"You are {self.name}, an AI-Assistant specialized in curating and predicting metadata for images. Your "
            f"task is"
            "to transform raw, unstructured metadata into well-formed XML, adhering to the OME XML"
            "standard. You have access to the OME XSD for reference via retrieval. Your responses"
            "should be exclusively in OME-XML format, aligning closely with the standard. Use the provided"
            "function for better reliability. Strive for"
            "completeness and validity. Rely on structured annotations only when necessary.")
        self.response_model = self.AugustResponseModel
        self.assistant = self.create_assistant()

    def run_assistant(self, msg, thread=None):
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
            assistant_id=self.assistant.id,
        )

        while run.status in ['queued', 'in_progress', 'cancelling']:
            print("Polling for run completion...")
            print(run.status)
            time.sleep(2)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
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

                    print(out)
                    tool_outputs += {"tool_call_id": call.id, "output": out}

        print(run.status)

    class AugustResponseModel(BaseModel):
        ome_xml: str = Field(description="The valid OME XML generated from the raw metadata.")
