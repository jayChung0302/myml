import torch
import torchvision
import argparse
from model.resnet_cifar10 import *

from resnet_pruner import pruner_resnet

def main():
    net = ResNet56(7)
    pruner = pruner_resnet(net)
    pruner.reset()
    pruner.net.eval()
    pruner.forward(torch.randn((1, 3, 224, 224), device='cpu'))
    print(pruner.cur_flops)

    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    from IPython import embed;embed()
    


if __name__ == '__main__':
    
    main()
