import random
import torch

class GridSearch():
    GRID_LIST_SUFFIX = '_grid'
    GRID_PARAM_SEPARATOR = ' '
    GRID_VALUE_SEPARATOR = '-'

    def __init__(self, solution, randomSearch = True):
        self.solution = solution
        self.solution.__grid_search__ = self
        self.randomSearch = randomSearch
        self.writer = None
        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = enabled
        return self

    def get_grid_attributes(self, solution):
        s = solution
        return {self.get_grid_attribute(s, a): self.get_grid_attribute_list(s, a) for a in dir(s) if self.filter_grid_attribute(s, a)}

    def get_grid_attribute(self, obj, attr):
        return attr[:-len(GridSearch.GRID_LIST_SUFFIX)]

    def get_grid_attribute_list(self, obj, attr):
        return getattr(obj, attr)

    def filter_grid_attribute(self, obj, attr):
        return attr.endswith(GridSearch.GRID_LIST_SUFFIX) and not attr.startswith('__') and not callable(getattr(obj,attr))

    def calc_grid_size(self, grid_attributes):
        grid_size = 1
        for attr, attr_list in grid_attributes.items():
            grid_size *= len(attr_list)
        return grid_size

    def grid_choice_to_str(self, grid_choice):
        grid_str = ''
        for attr, attr_value in grid_choice.items():
            if len(grid_str):
                grid_str += GridSearch.GRID_PARAM_SEPARATOR
            grid_str += attr + GridSearch.GRID_VALUE_SEPARATOR + str(attr_value)
        return grid_str

    def get_grid_choice(self, grid_attributes, grid_choice_history):
        history_size = len(grid_choice_history)
        while True:
            grid_choice = {}
            choice_ind = history_size
            for attr, attr_list in grid_attributes.items():
                attr_list_size = len(attr_list)
                if self.randomSearch:
                    attr_ind = random.randint(0, attr_list_size-1)
                else:
                    attr_ind = choice_ind%attr_list_size
                    choice_ind /= attr_list_size
                grid_choice[attr] = attr_list[attr_ind]
            choice_str = self.grid_choice_to_str(grid_choice)
            if choice_str not in grid_choice_history:
                return choice_str, grid_choice

    def set_grid_choice(self, choice_str, grid_choice):
        self.choice_str = choice_str
        for attr, attr_value in grid_choice.items():
            setattr(self.solution, attr, attr_value)

    def add_result(self, name, value):
        if not hasattr(self, 'results_cache'):
            self.results_cache = {}
        if name not in self.results_cache:
            self.results_cache[name] = {}
        if self.choice_str not in self.results_cache[name]:
            self.results_cache[name][self.choice_str] = []
        self.results_cache[name][self.choice_str].append(value)

    def get_results(self, name):
        return self.results_cache[name][self.choice_str]

    def get_stats(self, name):
        results = self.get_results(name)
        t = torch.FloatTensor(results)
        return t.mean().item(), t.std().item()

    def get_all_results(self, name):
        return self.results_cache[name]

    def search_model(self, case_data, solution, solution_manager):
        if self.enabled:
            grid_attributes = self.get_grid_attributes(self.solution)
            grid_size = self.calc_grid_size(grid_attributes)
            grid_choice_history = {}
            while len(grid_choice_history) <  grid_size:
                choice_str, grid_choice = self.get_grid_choice(grid_attributes, grid_choice_history)
                self.set_grid_choice(choice_str, grid_choice)
                if hasattr(solution, 'iter_number'):
                    for it in range(solution.iter_number):
                        solution.iter = it
                        case_data.manual_seed = it
                        solution_manager.train_model(solution, case_data)
                else:
                    solution_manager.train_model(solution, case_data)

                grid_choice_history[choice_str] = True
            print(solution_manager.accepted_string("[SEARCH COMPLETED]"))
            print("Specify case_number, if you want to search over other case data")
            exit(0)

    def log_step_value(self, writer, name, value, step):
        if self.enabled:
            if type(value) == dict:
                values = {self.choice_str + GridSearch.GRID_PARAM_SEPARATOR + key : val for key, val in value.items()}
            else:
                values = {self.choice_str: value}

            writer.add_scalars(name, values, step)

    @classmethod
    def run_case(self, case_data, solution, solution_manager):
        if '__grid_search__' in dir(solution) is not None:
            solution.__grid_search__.search_model(case_data, solution, solution_manager)
