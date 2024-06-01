class BaseDataset:

    def __init__(self, task_name: str):

        self.valid_tasks = ['hello']
        self.validate_task(task_name)

    def validate_task(self, task_name):

        formatted_name = task_name.replace('_', '-').lower()
        if formatted_name not in self.valid_tasks:
            raise ValueError('Parameter `task_name` not recognised for the given dataset' \
                ' ', f'(got task `{task_name}` for dataset {self.__class__.__name__}).')
        
    def train(self, model):

        raise NotImplementedError
    
    def eval(self, model):

        raise NotImplementedError