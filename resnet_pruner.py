from filter_pruner import Pruner
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from model.resnet_cifar10 import BasicBlock, DownsampleA
import numpy as np

def is_leaf_module(module):
    return len(module.children()) == 0

class pruner_resnet(Pruner):
    def forward(self, x):
        self.activation_idx = 0
        self.grad_idx = 0
        self.activations = []
        self.linear = None
        # activation index to the instance of conv layer
        self.activation_to_conv = {}
        self.conv_to_idx = {}
        # retrieve next immediate bn layer using activation index of conv
        self.bn_for_conv = {}
        self.cur_flops = 0

        def modify_forward(net):
            for module in net.children():
                if is_leaf_module(module):
                    def new_forward(module):
                        def lambda_forward(x):
                            return self.trace_layer(module, x)
                        return lambda_forward
                    module.old_forward = module.forward
                    module.forward = new_forward(module)
                else:
                    modify_forward(module)

        def restore_forward(net):
            for module in net.children():
                if is_leaf_module(module) and hasattr(module, 'old_forward'):
                    module.forward = module.old_forward
                    module.old_forward = None
                else:
                    restore_forward(module)

        modify_forward(self.net)
        y = self.net(x)
        restore_forward(self.net)

        self.btnk = False
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                self.linear = module
            if isinstance(module, Bottleneck):
                self.btnk = True
            pass
        if self.btnk:
            self.parse_dependency_btnk()
        else:
            self.parse_dependency()
        self.resoure_usage = self.cur_flops

        return y

    def trace_layer(self, layer, x):
        y = layer.old_forward(x)
        if isinstance(layer, nn.Conv2d):
            # Conv2d.weight.size() -> [out_ch, in_ch, kernel_w, kernel_h]
            self.conv_in_channels[self.activation_idx] = layer.weight.size(1)
            self.conv_out_channels[self.activation_idx] = layer.weight.size(0)
            h, w = y.size()[2:]
            self.omap_size[self.activation_idx] = (h, w)
            self.cost_map[self.activation_idx] = h * w * np.prod(layer.weight.size()[2:]) / layer.groups
            self.in_params[self.activation_idx] = np.prod(layer.weight.size()[1:])
            self.cur_flops += h * w * np.prod(layer.weight.size()[0:])

            if self.rank_type == 'l1_weight':
                pass
            elif self.rank_type == 'l2_weight':
                pass
            else:
                y.register_hook(self.compute_rank)
                self.activations.append(y)
            
            self.rates[self.activation_idx] = self.conv_in_channels[self.activation_idx] * \
                self.cost_map[self.activation_idx]
            self.activation_to_conv[self.activation_idx] = layer
            self.conv_to_idx[layer] = self.activation_index
            self.activation_idx += 1
        elif isinstance(layer, nn.BatchNorm2d):
            self.bn_for_conv[self.activation_idx-1] = layer
            if self.rank_type == 'l2_bn':
                pass
            if self.rank_type == 'l2_bn_param':
                pass

        elif isinstance(layer, nn.Linear):
            self.base_flops += np.prod(layer.weight.size())
            self.cur_flops += np.prod(layer.weight.size())
        
        self.org_conv_in_channels = self.conv_in_channels.copy()
        self.org_conv_out_channels = self.conv_out_channels.copy()

        return y

    def parse_dependency_btnk(self):
        self.downsample_conv = []
        self.pre_padding = {}
        self.next_conv = {}
        prev_conv_idx = 0
        cur_conv_idx = 0
        prev_res = -1
        for module in self.net.modules():
            if isinstance(module, Bottleneck):
                if prev_res > -1:
                    self.next_conv[prev_res] = [self.conv_to_idx[module.conv1]]
                self.next_conv[cur_conv_idx] = [self.conv_to_idx[module.conv1]]
                self.next_conv[self.conv_to_idx[module.conv1]] = [self.conv_to_idx[module.conv2]]
                self.next_conv[self.conv_to_idx[module.conv2]] = [self.conv_to_idx[module.conv3]]
                cur_conv_idx = self.conv_to_idx[module.conv3]
                if module.downsample is not None:
                    residual_conv_idx = self.conv_to_idx[module.downsample[0]]
                    self.downsample_conv.append(residual_conv_idx)
                    self.next_conv[prev_conv_idx].append(residual_conv_idx)
                    prev_res = residual_conv_idx
                    self.chains[cur_conv_idx] = residual_conv_idx
                else:
                    if (prev_res > -1) and (not prev_res in self.chains):
                        self.chains[prev_res] = cur_conv_idx
                    elif prev_conv_idx not in self.chains:
                        self.chains[prev_conv_idx] = cur_conv_idx
                prev_conv_idx = cur_conv_idx

    def parse_dependency(self):
        self.downsample_conv = []
        self.pre_padding = {}
        self.next_conv = {}
        prev_conv_idx = 0
        prev_res = -1
        for module in self.net.modules():
            if isinstance(module, BasicBlock):
                cur_conv_idx = self.conv_to_idx[module.conv[3]]
                if isinstance(module.shortcut, DownsampleA):
                    self.pre_padding[cur_conv_idx] = module.shortcut
                self.chains[prev_conv_idx] = cur_conv_idx
                prev_conv_idx = cur_conv_idx
        last_idx = -1
        for module in self.net.modules():
            if isinstance(module, nn.Conv2d) and module.weight.size(2) == 3:
                idx = self.conv_to_idx(module)
                if (last_idx > -1) and (not last_idx in self.next_conv):
                    self.next_conv[last_idx] = [idx]
                elif (last_idx > -1):
                    self.next_conv[last_idx].append(idx)
                last_idx = idx

    def get_valid_filters(self):
        pass
    
    def get_valid_flops(self):
        pass
    
    def mask_conv_layer_segment(self, layer_index, filter_range):
        pass

    def prune_conv_layer_segment(self, layer_index, filter_range):
        pass
    
    def amc_filter_compress(self):
        pass

    def amc_compress(self):
        pass
