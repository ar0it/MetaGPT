from assistant_template import AssistantTemplate


class DiscriminatorDave(AssistantTemplate):
    """
    This assistants goal is to decide which part of the raw metadata is already contained in the start point OME XML.
    """
    def __init__(self):
        super().__init__()
