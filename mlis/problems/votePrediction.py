# There are random function from 8 inputs.
# There are random input vector of size 8 * number of voters.
# We calculate function number of voters times and sum result.
# We return 1 if sum > voters/2, 0 otherwise
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
import numpy as np
from torch.autograd import Variable

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size_data = input_size
        self.ensemble_enabled = solution.ensemble_enabled

        self.input_size = input_size // 8
        self.output_size = output_size
        self.first_hidden_size = solution.first_hidden_size
        self.hidden_size = solution.hidden_size

        self.loss_ = solution.loss

        self.hidden_sizes = solution.hidden_sizes
        self.hidden_activations = solution.hidden_activations

        self.layer_count = solution.layer_count
        self.hidden_activation = solution.hidden_activation
        self.output_activation = solution.output_activation
        
        layer_sizes = [self.input_size, 45, self.output_size]
        layer_sizes2 = [8,*self.hidden_sizes, 1]    
        
        def get_module_list(layer_sizes, hidden_activations):
            args = []
            for i in range(1, len(layer_sizes)):
                args.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                args.append(nn.BatchNorm1d(layer_sizes[i], track_running_stats=False)) 
                args.append(self.get_activation(hidden_activations[i - 1] if i != len(layer_sizes) - 1 else self.output_activation))
            
            return args
            
        self.hidden_activations2 = ['relu'] + ['relu'] * self.hidden_size

        modules = get_module_list(layer_sizes, self.hidden_activations)
        modules2 = get_module_list(layer_sizes2, self.hidden_activations)

        self.model = nn.Sequential(*modules)
        self.model2 = nn.Sequential(*modules2)


        

    def get_activation(self, name):
        name = str.lower(name)

        if name == 'relu':
            return nn.ReLU()
        elif name == 'relu6':
            return nn.ReLU6()
        elif name == 'leakyrelu':
            return nn.LeakyReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        
        return nn.ReLU()

    def forward(self, x):
        x = (x - 0.5)*2.0
        output = self.model2(x.view(-1, 8))
        self.model2.output = output.view(-1, self.input_size)

        return self.model(self.model2.output)

    def calc_error(self, output, target):
        self.loss_ = str.lower(self.loss_)
        # This is loss function
        if self.loss_ == 'bceloss':
            result = nn.BCELoss()(output, target)
        elif self.loss_ == 'square':
            result = ((output-target)**2).sum()
        elif self.loss_ == 'mseloss':
            result = nn.MSELoss(reduction='mean')(output, target)
        else:
            raise 'Error: loss {} not found.'.format(self.loss_)

        return  result

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        self.layer_count = 3
        self.batch_size = 256
        self.ensemble_enabled = True
    
        self.algo_name = 'rmsprop'
        self.loss = 'bceloss'
        self.init_type = 'xavier'
        # Control speed of learning
        self.learning_rate = 0.01
        self.weight_decay = 0
        self.momentum = 0.9
        self.coef = 0.99
        self.step = 1
        self.epoch = 20
       
        # Control number of hidden neurons
        self.first_hidden_size = 45
        self.hidden_size = 40
        self.hidden_layer_count = self.layer_count - 1
        self.hidden_sizes = [self.first_hidden_size] + [self.hidden_size] * (self.hidden_layer_count - 1)
        
        self.first_hidden_activation = 'relu6'
        self.hidden_activation = 'relu6'
        self.hidden_activations = [self.first_hidden_activation] + [self.hidden_activation] * self.hidden_layer_count

        self.output_activation = 'sigmoid'

        self.weight_init = True
        self.nesterov_moment = False
        self.rho = 0.8 # adadelta
        
        # Grid search settings, see grid_search_tutorial
        # self.coef_grid = [0.85, 0.9, 0.95, 0.99]
        # self.step_grid = [1, 2, 3, 4]
        # self.momentum_grid = [0.65, 0.9, 0.95, 0.99]
        # self.weight_decay_grid = [0, 0.01, 0.001, 0.00001]
        # self.rho_grid = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        # self.batch_size_grid = [64, 128, 254, 512, 1024]
        # self.algo_name_grid = ['adam', 'sgd', 'rmsprop']
        self.learning_rate_grid = [0.011, 0.012, 0.013, 0.14, 0.15, 0.01, 0.001] #[0.1, 0.01, 0.001,0.003,0.004, 0.03, 0.04, 0.2, 0.3, 0.4] #[0.021, 0.1, 0.01, 0.001, 0.03, 0.04, 0.2, 0.3, 0.4, 0.009, 0.09, 0.0009]#[0.0001, 0.001, 0.005, 0.01, 0.1, 0.05, 0.009] #[0.1, 0.01, 0.001,1.5, 1, 2, 3]
        # self.weight_init_grid = [True, False]
        # self.nesterov_moment_grid = [True, False]
        # self.layer_count_grid = [3,4,5,6,7,8,9,10]
        # self.first_hidden_size_grid = [16,17,18,19,20,21,22,23,24,25,26, 39,40,41,42,43,44,45]
        # self.hidden_size_grid = [16, 32, 40, 43]
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
            optimizer = optim.Adam(model_param, lr=self.learning_rate,betas=(0.9, 0.999), weight_decay=self.weight_decay)
        elif name == 'sgd':
            optimizer = optim.SGD(model_param, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum, nesterov=self.nesterov_moment)
        elif name == 'adadelta':
            optimizer = optim.Adadelta(model_param, lr=self.learning_rate, weight_decay=self.weight_decay, rho=self.rho)
        elif name == 'adagrad':
            optimizer = optim.Adagrad(model_param, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name == 'rmsprop':
            optimizer = optim.RMSprop(model_param, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise 'Error: Optimizer \'{}\' not found.'.format(name)

        return optimizer

    # Return trained model
    def train_model(self, train_data, train_target, context):
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight, a=-1, b=1)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.uniform_(m.bias, a=-1, b=1)
                # m.bias.data.fill_(0.01)
        def weights_init_uniform_rule(m):
            classname = m.__class__.__name__
            # for every Linear layer in a model..
            if classname.find('Linear') != -1:
                # get the number of the inputs
                n = m.in_features
                y = 1.0/np.sqrt(n)

                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0.01)

        def get_init_type(key):
            types = {
                'uniform': init_normal,
                'xavier': init_weights,
                'uniform_rule': weights_init_uniform_rule
            }

            return types[key]
        # Uncommend next line to understand grid search
        if run_grid_search:
            self.grid_search_tutorial()

        model = SolutionModel(train_data.size(1), train_target.size(1), self)

        model.train()
        if self.weight_init:
            model.apply(get_init_type(self.init_type))

        # Optimizer used for training neural network
        optimizer = self.get_optimizer(self.algo_name, model.parameters())

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100000])
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        batches_count = train_data.size(0)//self.batch_size
        good_counter = 0
        good_limit = batches_count
        # epoch = 0
        while True:
            index = context.step % batches_count
            
            if index == batches_count-1: 
                # epoch += 1
                # print('Epoch {}'.format(epoch))
                indices = torch.randperm(train_data.size()[0])

                train_data, train_target = train_data[indices], train_target[indices]
            

            # train_dataset = torch.utils.data.TensorDataset(train_data, train_target)
            # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

            # for x, y in train_loader:
            # Report step, so we know how many steps
            context.increase_step()

            start = index * self.batch_size
            end = start + self.batch_size
            # print(end)
            x = train_data[start: end]
            y = train_target[start: end]

            # model.parameters()...gradient set to zero
            optimizer.zero_grad()

            # evaluate model => model.forward(data)
            output = model(x)
            with torch.no_grad():
                diff = (output - y).abs()

                if diff.max() <  0.4:
                    good_count += 1
                    if good_count >= good_limit:
                        break
                else:
                    good_count = 0
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

            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # update model: model.parameters() -= lr * gradient
            optimizer.step()


            # print progress of the learning
            self.print_stats(context.step, error, correct, total)

            time_left = context.get_timer().get_time_left()

            time_limit = 0.1 #if train_data.size(1) > 35 else 1.15 if train_data.size(1) > 23 else 1.5
            if time_left < time_limit:
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

    def get_index(self, tensor_index):
        index = 0
        for i in range(tensor_index.size(0)):
            index = 2*index + tensor_index[i].item()
        return index

    def calc_value(self, input_data, function_table, input_size, input_count_size):
        count = 0
        for i in range(input_count_size):
            count += function_table[self.get_index(input_data[i*input_size: (i+1)*input_size])].item()
        if count > input_count_size/2:
            return 1
        else:
            return 0

    def create_data(self, data_size, input_size, input_count_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_table = torch.ByteTensor(function_size).random_(0, 2)
        total_input_size = input_size*input_count_size
        data = torch.ByteTensor(data_size, total_input_size).random_(0, 2)
        target = torch.ByteTensor(data_size)
        for i in range(data_size):
            target[i] = self.calc_value(data[i], function_table, input_size, input_count_size)
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        input_size = 8
        data_size = (1<<input_size)*32
        input_count_size = case

        data, target = self.create_data(2*data_size, input_size, input_count_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs per voter and {} voters".format(input_size, input_count_size))

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
    # grid_search = gs.GridSearch()
    # grid_search.run(Config(), case_number=4, random_order=False, verbose=True)
    # results = grid_search.get_all_results('steps')
    # results = sorted((sum(value)/len(value), key) for key, value in results.items())
    # print(results)
    # exit()

    cases_results = {}
    for i in range(1, 11):
        grid_search = gs.GridSearch()
        grid_search.run(Config(), case_number=i, random_order=False, verbose=False)
        results = grid_search.get_all_results('steps')

        # results = sorted((sum(value)/len(value), key) for key, value in results.items())
        if i == 1:
            for key in results:
                cases_results[key] = results[key][0]
        else:
            for key in results:
                cases_results[key] += results[key][0]

    for key in cases_results:
        cases_results[key] = cases_results[key] / 10
    
    cases_results = sorted((value, key) for key, value in cases_results.items())
    print(cases_results)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
