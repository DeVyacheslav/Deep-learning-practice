# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # sm.SolutionManager.print_hint("Hint[1]: Increase hidden size")
        self.hidden_size = solution.hidden_size

        self.layer_count = solution.layer_count
        self.hidden_activation = solution.hidden_activation
        self.output_activation = solution.output_activation

        self.linears = nn.ModuleList(
            [nn.Linear(self.input_size if i == 0 else self.hidden_size, 
                self.hidden_size if i != self.layer_count - 1 else self.output_size) 
            for i in range(self.layer_count)]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(self.hidden_size, track_running_stats=False) for i in range(self.layer_count)]
        )

        args = []
        for i in range(self.layer_count):
            args.append(self.linears[i])
            if i != self.layer_count - 1:
                args.append(self.batch_norms[i]) 
            args.append(self.get_activation(self.hidden_activation if i != self.layer_count - 1 else self.output_activation))

        self.model = nn.Sequential(*args)

        self.loss_ = 'BCEloss'

    def get_activation(self, name):
        name = str.lower(name)

        if name == 'relu':
            return nn.ReLU()
        elif name == 'relu6':
            return nn.ReLU6()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        
        return nn.ReLU()

    def forward(self, x):
        return self.model.forward(x)

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
        self.layer_count = 5
        self.batch_size = 128
        self.hidden_activation = 'relu6'
        self.output_activation = 'sigmoid'
        self.algo_name = 'adam'
        self.mini_batch = True
        # Control speed of learning
        self.learning_rate = 0.01
        self.weight_decay = 0
        self.momentum = 0.9
        self.coef = 0.99
        self.step = 1
        self.epoch = 10
        # Control number of hidden neurons
        self.hidden_size = 105
        self.weight_init = True
        self.nesterov_moment = False
        self.rho = 0.8 # adadelta

        # Grid search settings, see grid_search_tutorial
        # self.coef_grid = [0.85, 0.9, 0.95, 0.99]
        # self.step_grid = [1, 2, 3, 4]
        # self.momentum_grid = [0.65, 0.9, 0.95, 0.99]
        # self.weight_decay_grid = [0, 0.01, 0.001, 0.00001]
        # self.rho_grid = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        self.learning_rate_grid = [0.021, 0.1, 0.01, 0.001, 0.03, 0.04, 0.2, 0.3, 0.4, 0.009, 0.09, 0.0009]#[0.0001, 0.001, 0.005, 0.01, 0.1, 0.05, 0.009] #[0.1, 0.01, 0.001,1.5, 1, 2, 3]
        # self.weight_init_grid = [True, False]
        # self.nesterov_moment_grid = [True, False]
        # self.hidden_size_grid = [i for i in range(25, 55)]#[37,38,39,40,41, 49,50, 51,52,53,54]
        # self.loss__grid = ['BCEloss', 'square']
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
            optimizer = optim.Adadelta(model_param, lr=self.learning_rate, weight_decay=self.weight_decay, rho=self.rho)
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
                m.bias.data.fill_(0.0)
        def weights_init_uniform_rule(m):
            classname = m.__class__.__name__
            # for every Linear layer in a model..
            if classname.find('Linear') != -1:
                # get the number of the inputs
                n = m.in_features
                y = 1.0/np.sqrt(n)

                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0.0)

        # Uncommend next line to understand grid search
        if run_grid_search:
            self.grid_search_tutorial()

        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        model.train()
        if self.weight_init:
            model.apply(init_weights)

        # Optimizer used for training neural network
        # sm.SolutionManager.print_hint("Hint[2]: Learning rate is too small", context.step)
        # optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum, nesterov=self.nesterov_moment)
        optimizer = self.get_optimizer(self.algo_name, model.parameters())

        number_of_batches = int(train_data.size(0)/self.batch_size)
        batches_counter = 0

        for i in range(self.epoch):
            # Report step, so we know how many steps
            context.increase_step()
            indices = torch.randperm(train_data.size()[0])

            train_data = train_data[indices]
            train_target = train_target[indices]

            train_data_batches = torch.utils.data.DataLoader(train_data, self.batch_size)
            train_target_batches = torch.utils.data.DataLoader(train_target, self.batch_size)

            for x, y in zip(train_data_batches, train_target_batches):
                # model.parameters()...gradient set to zero
                optimizer.zero_grad()

                # evaluate model => model.forward(data)
                output = model(x)

                # if x < 0.5 predict 0 else predict 1
                predict = model.calc_predict(output)
                # Number of correct predictions
                correct = predict.eq(y.view_as(predict)).long().sum().item()
                # Total number of needed predictions
                total = predict.view(-1).size(0)
                # No more time left or learned everything, stop training
                time_left = context.get_timer().get_time_left()
                if time_left < 0.1:
                    break
                # calculate error
                error = model.calc_error(output, y)

                residue = (output.data-y.data).abs()
                if residue.max() < 0.55:
                    batches_counter += 1
                    if batches_counter >= number_of_batches:
                        break
                else:
                    batches_counter = 0
                # calculate deriviative of model.forward() and put it in model.parameters()...gradient
                error.backward()
                    # update model: model.parameters() -= lr * gradient
                optimizer.step()
            # print progress of the learning
            self.print_stats(context.step, error, correct, total)
            # evaluate model => model.forward(data)
            output = model(train_data)

            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1 or correct == total:
                break

           
                
        if self.grid_search:
            res = context.step if correct == total else 1000000
            self.grid_search.add_result('steps', res)

        return model

    def print_stats(self, step, error, correct, total):
        if step % 10 == 0:
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
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
#run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
