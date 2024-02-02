from src.main.assistants.assistant_template import AssistantTemplate


class AllAloneAugust(AssistantTemplate):
    """
    This assistant is the most basic assistant approach and attempts to translate the raw meta-data all alone :(.
    """

    def __init__(self, ome_xsd_path, client):
        super().__init__(ome_xsd_path, client)
        self.name = __name__.split(".")[-1].replace("_", " ")
