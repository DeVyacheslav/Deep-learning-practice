# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        # sm.SolutionManager.print_hint("Hint[1]: Increase hidden size")
        self.hidden_size = solution.hidden_size
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, output_size)
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, output_size)
        self.loss_ = 'BCEloss'

    
    def forward(self, x):
        x = nn.Sequential(self.linear1,
                      nn.ReLU(),
                      self.linear2,
                      nn.Sigmoid()
            ).forward(x)
        return x

    def calc_error(self, output, target):
        # This is loss function
        if self.loss_ == 'BCEloss':
            result = nn.BCELoss()(output, target)
        elif self.loss_ == 'square':
            result = ((output-target)**2).sum()
        # result = (-1.0 * (target * torch.log(output) + (1.0 - target) * torch.log(1.0 - output))).sum()
        # result = ((output-target)**2).sum()
        return  result

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        self.algo_name = 'adadelta'
        self.mini_batch = False
        # Control speed of learning
        self.learning_rate = 5
        self.weight_decay = 0
        self.momentum = 0.95
        self.coef = 0.99
        self.step = 1
        self.epoch = 2
        # Control number of hidden neurons
        self.hidden_size = 93
        self.weight_init = False
        self.nesterov_moment = False
        

        # Grid search settings, see grid_search_tutorial
        self.coef_grid = [0.95, 0.99]
        self.step_grid = [3, 4]
        self.momentum_grid = [0.85, 0.95]
        self.learning_rate_grid = [0.1, 0.5, 0.9, 1.5, 2, 2.5]
        self.hidden_size_grid = [85, 95]
        # grid search will initialize this field
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 1

    def get_optimizer(self, optimization_name, model_param):
        name = str.lower(optimization_name)

        if name == 'adam':
            optimizer = optim.Adam(model_param, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name == 'sgd':
            optimizer = optim.SGD(model_param, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum, nesterov=self.nesterov_moment)
        elif name == 'adadelta':
            optimizer = optim.Adadelta(model_param, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name == 'adagrad':
            optimizer = optim.Adagrad(model_param, lr=self.learning_rate, weight_decay=self.weight_decay)

        return optimizer

    # Return trained model
    def train_model(self, train_data, train_target, context):
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        def weights_init_uniform_rule(m):
            classname = m.__class__.__name__
            # for every Linear layer in a model..
            if classname.find('Linear') != -1:
                # get the number of the inputs
                n = m.in_features
                y = 1.0/np.sqrt(n)

                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0.001)

        # Uncommend next line to understand grid search
        if run_grid_search:
            self.grid_search_tutorial()

        model = SolutionModel(train_data.size(1), train_target.size(1), self)

        if self.weight_init:
            model.apply(weights_init_uniform_rule)

        # Optimizer used for training neural network
        # sm.SolutionManager.print_hint("Hint[2]: Learning rate is too small", context.step)
        # optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum, nesterov=self.nesterov_moment)
        optimizer = self.get_optimizer(self.algo_name, model.parameters())
        scheduler = None
        if self.algo_name == 'sgd':
            scheduler = optim.lr_scheduler.StepLR(optimizer, self.step, self.coef)
        # model.parameters()...gradient set to zero
        # optimizer.zero_grad()
        while True:
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            if self.mini_batch:
                optimizer.zero_grad()

            # evaluate model => model.forward(data)

            temp = torch.cat((train_data, train_target), 1)
            temp = temp[torch.randperm(temp.size()[0])]

            train_data = temp[:, :-1]
            train_target = temp[:, -1:]

            output = model(train_data)
            
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1 or correct == total or context.step > 200:
                break
            # calculate error
            error = model.calc_error(output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            self.print_stats(context.step, error, correct, total)
            if  run_grid_search and context.step > 50:
                break
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            
            if scheduler:
                scheduler.step()

        if self.grid_search:
            res = context.step if correct == total else 1000000
            self.grid_search.add_result('steps', res)

        return model

    def print_stats(self, step, error, correct, total):
        if step % 1000 == 0:
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))
    
    def grid_search_tutorial(self):
        # During grid search every possible combination in field_grid, train_model will be called
        # iter_number times. This can be used for automatic parameters tunning.
        if self.grid_search:
            # print("[HelloXor] learning_rate={} iter={}".format(self.learning_rate, self.iter))
            self.grid_search.add_result('iter', self.iter)
            if self.iter == self.iter_number-1:
                # print("[HelloXor] chose_str={}".format(self.grid_search.choice_str))
                # print("[HelloXor] iters={}".format(self.grid_search.get_results('iter')))
                stats = self.grid_search.get_stats('iter')
                # print("[HelloXor] Mean={} Std={}".format(stats[0], stats[1]))
        else:
            print("Enable grid search: See run_grid_search in the end of file")
            exit(0)
###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
# run_grid_search = True
if run_grid_search:
    grid_search = gs.GridSearch()
    grid_search.run(Config(), case_number=3, random_order=False, verbose=True)
    results = grid_search.get_all_results('steps')
    results = sorted((sum(value)/len(value), key) for key, value in results.items())
    print(results)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
