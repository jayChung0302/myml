import torch
import torch.nn as nn

import numpy as np

__all__ = ['Pruner']

class Pruner:
    def __init__(self, net, rank_type='l2_weight', num_class=1000, \
        safeguard=0, random=False, device='cuda', resource='FLOPs'):
        self.net = net
        self.rank_type = rank_type
        self.chains = {} # chainning conv (use activation index 2 present a conv)
        self.y = None
        self.safeguard = safeguard
        self.random = random
        self.device = device
        self.resoure_type = resource
        self.reset()

    def forward(self, x):
        raise NotImplementedError
    
    def get_valid_filters(self):
        raise NotImplementedError
    
    def get_valid_flops(self):
        raise NotImplementedError

    def count_params(self):
        '''Count a number of network's trainable parameters'''
        params_conv, params_all = 0, 0

        for module in self.net.modules():
            if isinstance(module, nn.Conv2d):
                params_all += np.prod(module.weight.size())
                params_conv += np.prod(module.weight.size())
            if isinstance(module, nn.Linear):
                params_all += np.prod(module.weight.size())
                

        return params_all, params_conv

    def reset(self):
        self.cur_flops = 0
        self.base_flops = 0
        self.cur_size, conv_size = self.count_params()
        self.base_size = self.cur_size - conv_size
        self.quota = None
        self.filter_ranks = {}
        self.rates = {}
        self.cost_map = {}
        self.in_params = {}
        self.omap_size = {}
        self.conv_in_channels = {}
        self.conv_out_channels = {}

    def flop_regularize(self, l):
        for key in self.filter_ranks:
            self.filter_ranks[key] -= l * self.rates[key]

    def compute_rank(self, grad):
        activation_idx = len(self.activations) - self.grad_idx - 1
        activation = self.activations[activation_idx]
        if self.rank_type == 'analysis':
            if activation_idx not in self.filter_ranks:
                self.filter_ranks[activation_idx] = activation * grad
            else:
                self.filter_ranks[activation_idx] = torch.cat((self.filter_ranks[activation_idx], activation*grad), 0)

        else:
            if self.rank_type == 'meanAbsMeanImpact':
                values = torch.abs((grad * activation).sum((2, 3)) / np.prod(activation.shape[2:]))

            # NxC to C
            values = values.sum(0) / activation.size(0)

            if activation_idx not in self.filter_ranks:
                self.filter_ranks[activation_idx] = torch.zeros(activation.size(1), device=self.device)
            
            self.filter_ranks[activation_idx] += values
        
        self.grad_idx += 1        

    def calculate_cost(self, encoding):
        pass

    def get_unit_flops_for_layer(self, layer_id):
        pass

    def get_unit_filters_for_layer(self, layer_id):
        pass

    def one_shot_lowest_ranking_filters(self, target):
        # Consolidation of chained channels
        # Use the maximum rank among the chained channels for the criteria for those channels
        # Greedily pick from the lowest rank.
        # 
        # This return list of [layers act_index, filter_index, rank]
        data = []
        chained = []
        
        # keys of filter_ranks are activation index
        checked = []
        org_filter_size = {}
        new_filter_size = {}
        for i in sorted(self.filter_ranks.keys()):
            org_filter_size[i] = self.filter_ranks[i].size(0)
            if i in checked:
                continue
            current_chain = []
            k = i
            while k in self.chains:
                current_chain.append(k)
                k = self.chains[k]
            current_chain.append(k)
            checked.append(k)

            sizes = np.array([self.filter_ranks[x].size(0) for x in current_chain])
            max_size = np.max(sizes)
            
            for k in current_chain:
                new_filter_size[k] = max_size

            ranks = [self.filter_ranks[x].to(self.device) for x in current_chain]
            cnt = torch.zeros(int(max_size), device=self.device)

            for idx in range(len(ranks)):
                pass

    def one_shot_lowest_ranking_filters_multi_targets(self, targets):
        pass
    
    def pruning_with_transformations(self, original_dist, \
        perturbation, target, masking=False):
        pass
    
    def pruning_with_transformations_multi_target(self, original_dist,\
        perturbation, target, masking=False):
        pass

    def normalize_ranks_per_layer(self):
        pass
    
    def get_pruning_plan_from_layer_budget(self, layer_budget):
        pass

    def sort_weights(self):
        pass

    def get_pruning_plan_from_importance(self, target, importance):
        pass

    def pack_pruning_target(self, filters_to_prune_per_layer, get_segment=True,\
        progressive=True):
        pass

    def get_pruning_plan(self, num_filters_to_prune, progressive=True,\
         get_segment=False):
        pass

    def get_uniform_ratio(self, target):
        pass

    def uniform_grow(self, growth_rate):
        pass

    def get_pruning_plan_multi_target(self, targets):
        pass

