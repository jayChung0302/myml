from filter_pruner import Pruner
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

def is_leaf_module(module):
    return len(module.children()) == 0

class pruner_resnet(Pruner):
    def trace_layer(self, layer, x):
        y = layer.old_forward(x)
        if isinstance(layer, nn.Conv2d):
            # Conv2d.weight.size() -> [out_ch, in_ch, kernel_w, kernel_h]
            self.conv_in_channels[self.activation_idx] = layer.weight.size(1)
            self.conv_out_channels[self.activation_idx] = layer.weight.size(0)
            h, w = y.size()[2:]
            self.omap_size[self.activation_idx] = (h, w)
            self.cost_map[self.activation_idx] = h * w * layer.weight.size(2) * layer.weight.size(3) / layer.groups
            pass

    def parse_dependency_btnk(self):
        pass

    def parse_dependency(self):
        pass

    def forward(self, x):
        self.activation_idx = 0
        self.grad_idx = 0
        self.activations = []
        self.linear = None
        # activation index to the instance of conv layer
        self.activation_to_conv = {}
        self.conv_to_index = {}
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
