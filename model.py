from queue import Empty
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_config(config_file):
    """
    Parse the yolov3 config file

    return a dictionary of necessary building blocks
    """

    with open(config_file, 'r') as f:
        lines = f.read().split('\n')
        lines = [l for l in lines if len(l) > 0] # remove empty lines
        lines = [l for l in lines if l[0] != '#'] # remove comments
        lines = [l.strip() for l in lines] # remove white spaces

        block = {}
        blocks = []

        for line in lines:
            if line[0] == "[":
                if len(block) != 0:
                    blocks.append(block)
                    block={}

                block["type"] = line[1:-1].strip()
            
            else:
                k, v = line.split('=')
                block[k.strip()] = v.strip()
        
        blocks.append(block)
    return blocks

def create_modules(blocks):
    """
    Creating modules from parsed blocks

    Possible blocks:
    1. convolutional
    2. upsample
    3. route
    4. shortcut
    5. yolo
    6. net
    """

    net_info = blocks[0] # net block from cfg
    module_list = nn.ModuleList()
    input_channels = 3 # initialize input channels as 3 (RBG)
    output_filters = []

    for i, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        if block['type'] == 'convolutional':
            activation = block['activation']

            try:
                batch_norm = int(block['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True
            
            filters = int(block['filters'])
            padding = int(block['pad'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])

            pad = (kernel_size - 1) if padding else 0

            conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=bias
            )
            module.add_module(f"conv{i}", conv)

            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm{i}", bn)
            
            if activation == 'leaky':
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky{i}", act)

        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            module.add_module(f"upsample{i}", upsample)

        elif block['type'] == 'route':
            layers = block['layers'].split(',')

            start = int(layers[0]) 
            end = int(layers[1]) if len(layers) > 1 else 0

            if start > 0:
                start = start - i
            if end > 0:
                end = end - i

            route = EmptyLayer()
            module.add_module(f"route{i}", route)

            if end < 0:
                filters = output_filters[i + start] + output_filters[i + end]
            else:
                filters = output_filters[i + start]

        elif block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f"shortcut{i}", shortcut)

        elif block['type'] == 'yolo':
            mask = block['mask'].split(",")
            mask = [int(i) for i in mask]

            anchors = block['anchors'].split(',')
            anchors = [int(i) for i in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection{i}", detection)
            
        module_list.append(module)
        input_channels = filters
        output_filters.append(filters)
    
    return (net_info, module_list)



if __name__ == "__main__":
    blocks = parse_config('./cfg/yolov3.cfg')
    model = create_modules(blocks)

    print(model)
