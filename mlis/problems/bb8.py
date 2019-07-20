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

class InputTransform(nn.Module):
    def __init__(self, input_size):
        super(InputTransform, self).__init__()
        self.input_size = input_size

    def forward(self, x):
        return x.view(-1, self.input_size)

class InputTransformEmbed(nn.Module):
    def __init__(self):
        super(InputTransformEmbed, self).__init__()

    def forward(self, x):
        return x.t().long()
        
class InputDebug(nn.Module):
    def __init__(self, exit_enabled=False):
        super(InputDebug, self).__init__()
        self.exit_enabled = exit_enabled

    def forward(self, x):
        print(x, x.size())#, len(x[0]), len(x[0][0]) , len(x[0][0][0]))
        if self.exit_enabled:
            exit()
        return x

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size_data = input_size
        self.ensemble_enabled = solution.ensemble_enabled

        self.input_size = input_size 
        self.output_size = output_size
        self.first_hidden_size = solution.first_hidden_size
        self.hidden_size = solution.hidden_size

        self.loss_ = solution.loss

        self.hidden_sizes = solution.hidden_sizes
        self.hidden_activations = solution.hidden_activations

        self.layer_count = solution.layer_count
        self.hidden_activation = solution.hidden_activation
        self.output_activation = solution.output_activation
        
        self.embed_size = input_size // 4 if input_size // 4 <= 16 else 16
        self.conv = nn.Conv1d(self.input_size, self.embed_size, 3)
        self.embed = nn.Embedding(input_size, self.embed_size)

        # self.embed_size = self.input_size

        # self.embed_size = self.embed_size // 2 -1
        self.GRU = nn.GRU(self.embed_size, self.embed_size, self.layer_count, dropout=0.2)
        self.LSTM = nn.LSTM(self.embed_size, self.embed_size, self.layer_count, dropout=0.2)   
        # self.conv = nn.Conv1d(self.input_size, self.embed_size, 3)
        self.fc = nn.Linear(self.embed_size, 1)

        # layer_sizes = [self.input_size, *self.hidden_sizes, self.output_size]
        # layer_sizes = [self.embed_size, *self.hidden_sizes, self.output_size]
        # layer_sizes2 = [8,*self.hidden_sizes, 1]    
        
        # def get_module_list(layer_sizes, hidden_activations):
        #     args = []

        #     # args.append(self.embed)
        #     # args.append(nn.GRU(self.embed_size, 256))
        #     # args.append(InputTransformEmbed())
        #     # # args.append(InputDebug(True))
        #     # args.append(self.embed)
        #     # args.append(self.LSTM)
        #     # args.append(InputDebug(True))
        #     for i in range(1, len(layer_sizes)):
        #         if i == len(layer_sizes) - 1:
        #             args.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        #         else:
        #             pass
        #             # args.append(nn.LSTM(3, 3))
        #             # args.append(nn.GRUCell(layer_sizes[i - 1], layer_sizes[i]))
        #             # args.append(InputDebug(True))
        #             # args.append(nn.Dropout(0.8))
                
        #         # args.append(nn.BatchNorm1d(layer_sizes[i], track_running_stats=False))
        #     args.append(self.get_activation(hidden_activations[i - 1] if i != len(layer_sizes) - 1 else self.output_activation))
            
        #     return args
            
        # self.hidden_activations2 = ['relu'] + ['relu'] * self.hidden_size

        # modules = get_module_list(layer_sizes, self.hidden_activations)
        # modules2 = get_module_list(layer_sizes2, self.hidden_activations)

        # self.model = nn.Sequential(*modules)
        # self.model2 = nn.Sequential(*modules2)

        # print(self.model)
        # print(self.GRU)
        # exit()


        

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
        elif name == 'softmax':
            return nn.LogSoftmax(dim=1)
        
        return nn.ReLU()

    def forward(self, x):
        if not self.training and torch.cuda.is_available():
            x = x.cuda()

        batch_size = x.size(0)
        feature_size = x.size(1)

        x = x.float()
        # x = (x - 0.5)*2.0
        # x = x.view(-1,1)
        x = x.t()
        
        embed = self.embed(x.long())
        # print(embed.view(64,self.embed_size,256).size())
        # exit()

        # embed = self.conv(embed)
        # embed = nn.ReLU()(embed)
        

        # embed = nn.MaxPool1d(2)(embed)

        # print(embed.size())
        # exit()
        # embed = embed.view(x.size(1), x.size(0), self.embed_size)
        # print(embed.size())
        # exit()
        # embed = nn.Dropout(0.2)(embed)
        output, hidden = self.GRU(embed)
        # output = nn.Dropout(0.2)(output)
        # print(output.size(), hidden.size())
        # exit()
        fc_output = self.fc(output[feature_size - 1])
        fc_output = nn.Sigmoid()(fc_output)
        if not self.training and torch.cuda.is_available():
            fc_output = fc_output.cpu()

        return fc_output #self.model(x)

    def forward9(self, x):
        x = x.float()
        # x = (x - 0.5)*2.0

        return self.model(x)

    def calc_error(self, output, target):
        self.loss_ = str.lower(self.loss_)
        # This is loss function
        if self.loss_ == 'bceloss':
            result = nn.BCELoss()(output, target)
        elif self.loss_ == 'square':
            result = ((output-target)**2).sum()
        elif self.loss_ == 'mseloss':
            result = nn.MSELoss(reduction='mean')(output, target)
        elif self.loss_ == 'nll_loss':
            result = nn.NLLLoss()(output, target)
        elif self.loss_ == 'crossentropyloss':
            result = nn.CrossEntropyLoss()(output, target)
        else:
            raise 'Error: loss {} not found.'.format(self.loss_)

        return  result

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        self.layer_count = 1
        self.batch_size = 64
        self.ensemble_enabled = True
    
        self.algo_name = 'adam'
        self.loss = 'bceloss'
        self.init_type = 'xavier'
        # Control speed of learning
        self.learning_rate = 0.03
        self.weight_decay = 0
        self.momentum = 0.9
        self.coef = 0.99
        self.step = 1
        self.epoch = 20
       
        # Control number of hidden neurons
        self.first_hidden_size = 15
        self.hidden_size = 15
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

        train_target = train_target.float() #train_target.type(torch.LongTensor)
        # train_data = train_data.float() #train_data.type(torch.FloatTensor)

        # print(train_data, train_target)
        # exit()
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        # print(torch.cuda.is_available())
        if torch.cuda.is_available():
            model.cuda()
            train_data = train_data.cuda()
            train_target = train_target.cuda()
            
        model.train()
        if self.weight_init:
            model.apply(get_init_type(self.init_type))

        # Optimizer used for training neural network
        optimizer = self.get_optimizer(self.algo_name, model.parameters())

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100000])
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        batches_count = train_data.size(0)//self.batch_size
        good_counter = 0
        good_limit = 6 #batches_count
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
            # print(output[0], output.size(0), output[0].size(0))
            # exit()
            # with torch.no_grad():
            #     # print(output, y)
            #     # exit()
            #     diff = (output - y).abs()

            #     if diff.max() <  0.4:
            #         good_count += 1
            #         if good_count >= good_limit:
            #             break
            #     else:
            #         good_count = 0
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
            # self.print_stats(context.step, error, correct, total)

            # time_left = context.get_timer().get_time_left()

            # time_limit = 0.1 #if train_data.size(1) > 35 else 1.15 if train_data.size(1) > 23 else 1.5
            # if time_left < time_limit:
            #     break

           
                
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
        self.test_limit = 0.8

# There are 2 languages.
class Language:
    def __init__(self, states_count, letters_count):
        self.states_count = states_count
        with torch.no_grad():
            self.state_to_state_prob = torch.FloatTensor(states_count, states_count).uniform_()
            self.state_to_state_prob = self.state_to_state_prob/torch.sum(self.state_to_state_prob, dim=1).view(-1,1)
            self.state_to_state_cum_prob = torch.cumsum(self.state_to_state_prob, dim=1)

    def balance(self, sentences):
        letters_id, counts = torch.unique(sentences.view(-1), sorted=True, return_counts=True)
        perm = torch.randperm(letters_id.size(0))
        letters_id = letters_id[perm]
        counts = counts[perm]
        total = counts.sum().item()
        x = torch.ByteTensor(total+1).zero_()
        x[0] = 1
        xs = [x]
        for letter_id, count in zip(letters_id, counts):
            cc = count.item()
            nx = xs[-1].clone()
            nx[cc:][xs[-1][:-cc]] = 1
            xs.append(nx)
        best_balance = total//2
        while xs[-1][best_balance].item() == 0:
            best_balance -= 1
        #if best_balance != total//2:
        #    print("UNBALANCED")
        current_balance = best_balance
        balance_set = [False for _ in range(letters_id.size(0))]
        last_ind = len(xs)-1
        while current_balance != 0:
            while xs[last_ind-1][current_balance].item() == 1:
                last_ind -= 1
            balance_set[last_ind-1] = True
            current_balance -= counts[last_ind-1].item()
        b_sentences = sentences.clone()
        self.state_to_state_letter = self.state_to_state_letter.view(-1)
        for ind, set_id in enumerate(balance_set):
            val = 0
            if set_id:
                val = 1
            b_sentences[sentences == letters_id[ind]] = val
            self.state_to_state_letter[letters_id[ind]] = val
        assert b_sentences.view(-1).sum() == best_balance
        self.state_to_state_letter = self.state_to_state_letter.view(self.states_count, self.states_count)
        return b_sentences

    def gen(self, count, length):
        with torch.no_grad():
            self.state_to_state_letter = torch.arange(self.states_count*self.states_count).view(self.states_count, self.states_count)
            #self.state_to_state_letter.random_(0,2)
            sentences = torch.LongTensor(count, length)
            states = torch.LongTensor(count).random_(0, self.states_count)
            for i in range(length):
                res = torch.FloatTensor(count).uniform_()
                probs = self.state_to_state_cum_prob[states]
                next_states = self.states_count-(res.view(-1,1) < probs).sum(dim=1)
                next_states = next_states.clamp(max=self.states_count-1)
                letters_ind = self.state_to_state_letter[states, next_states]
                sentences[:,i] = letters_ind
                states = next_states
            sentences = self.balance(sentences)
            return sentences

    def calc_probs(self, sentences):
        size = sentences.size(0)
        states_count = self.state_to_state_prob.size(0)
        length = sentences.size(1)
        with torch.no_grad():
            state_to_prob = torch.FloatTensor(size, states_count).double()
            state_to_prob[:,:] = 1.0
            for i in range(length):
                letters = sentences[:,i]
                s1 = self.state_to_state_letter.size()
                s2 = letters.size()
                sf = s2+s1

                t1 = self.state_to_state_letter.view((1,)+s1).expand(sf)
                t2 = letters.view(s2+(1,1)).expand(sf)
                t3 = self.state_to_state_prob.view((1,)+s1).expand(sf).double()
                t4 = (t1 == t2).double()
                t5 = torch.mul(t3, t4)
                t6 = state_to_prob
                next_state_to_prob = torch.matmul(t6.view(t6.size(0), 1, t6.size(1)), t5).view_as(t6)
                state_to_prob = next_state_to_prob
            return state_to_prob.sum(dim=1)

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, length, states_count, letters_count, seed):
        while True:
            torch.manual_seed(seed)
            languages = [Language(states_count, letters_count), Language(states_count, letters_count)]
            data_size_per_lang = data_size//len(languages)
            datas = []
            targets = []
            for ind, lan in enumerate(languages):
                datas.append(lan.gen(data_size_per_lang, length))
                t = torch.LongTensor(data_size_per_lang)
                t[:] = ind
                targets.append(t)
            bad_count = 0
            good_count = 0
            for ind, data in enumerate(datas):
                probs = [lan.calc_probs(data) for lan in languages]
                bad_count += (probs[ind] <= probs[1-ind]).long().sum().item()
                good_count += (probs[ind] > probs[1-ind]).long().sum().item()
            best_prob = good_count/(bad_count+good_count)
            if best_prob > 0.95:
                break
            print("Low best prob = {}, seed = {}".format(best_prob, seed))
            seed += 1

        data = torch.cat(datas, dim=0)
        target = torch.cat(targets, dim=0)
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data, target.view(-1, 1).float(), best_prob)

    def create_case_data(self, case):
        data_size = 256*4
        case_configs = [
                (8, 2, 7),
                (16, 3, 34),
                (32, 4, 132),
                (64, 5, 13),
                (128, 6, 1),
                (256, 7, 5),
                (256, 7, 6),
                (256, 7, 71),
                (256, 7, 19),
                (256, 7, 40)
                ]
        case_config = case_configs[min(case, 10)-1]
        length = case_config[0]
        states_count = case_config[1]
        # seed help generate data faster
        seed = 1000*case + case_config[2]
        letters_count = 2
        data, target, best_prob = self.create_data(2*data_size, length, states_count, letters_count, seed)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("States = {} Length = {} Seed = {} Best prob = {:.3}".format(states_count, length, seed, best_prob))

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
    exit()

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
     # this part is needed for GPU initialization (so that it doesn't take place
    # inside the first test)
    a = torch.tensor(np.ones((1, 10000)), dtype=torch.float32)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    a.to(device)
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
