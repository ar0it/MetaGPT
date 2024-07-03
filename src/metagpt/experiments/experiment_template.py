class ExperimentTemplate:
    """
    The experiment template class defines an experiment object that can be used to run experiments.
    The experiment defines the dataset, the predictors, the evaluator and more.
    """
    def __init__(self) -> None:
        self.dataset = None
        self.predictors:list = []
        self.reps:int = 1
        self.create_report:bool = True