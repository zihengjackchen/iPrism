from pytorchfi.core import fault_injection as pfi_core
import random
import logging
import struct
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class PyTorchFICarlaUtils(object):
    def __init__(self):
        pass
    
    @staticmethod
    def floatToBits(f):
        s = struct.pack('>f', f)
        return struct.unpack('>l', s)[0]

    @staticmethod
    def bitsToFloat(b):
        s = struct.pack('>l', b)
        return struct.unpack('>f', s)[0]

    @staticmethod
    def is_conv_weight_layer(name):
        # Note layers 'network.classifier.4.weight' and 'network.classifier.1.weight' are conv layers too
        if "weight" not in name:
            return False
        elif "conv" in name:
            return True
        elif "network.classifier.1.weight" in name:
            return True
        elif "network.classifier.4.weight" in name:
            return True
        return False

    @staticmethod
    def random_weight_location(pfi_model, conv=-1):
        loc = list()

        if conv == -1:
            corrupt_layer = random.randint(0, pfi_model.get_total_conv() - 1)
        else:
            corrupt_layer = conv
        loc.append(corrupt_layer)

        curr_layer = 0
        for name, param in pfi_model.get_original_model().named_parameters():
            if PyTorchFICarlaUtils.is_conv_weight_layer(name):
                if curr_layer == corrupt_layer:
                    for dim in param.size():
                        loc.append(random.randint(0, dim - 1))
                curr_layer += 1

        assert curr_layer == pfi_model.get_total_conv()
        assert len(loc) == 5

        return tuple(loc)

    @staticmethod
    def random_neuron_location(pfi_model, conv=-1):
        if conv == -1:
            conv = random.randint(0, pfi_model.get_total_conv() - 1)

        c = random.randint(0, pfi_model.get_fmaps_num(conv) - 1)
        h = random.randint(0, pfi_model.get_fmaps_H(conv) - 1)
        w = random.randint(0, pfi_model.get_fmaps_W(conv) - 1)

        return (conv, c, h, w)
    
    def random_value(min_val=-1, max_val=1):
        return random.uniform(min_val, max_val)

    # NOTE not used, random generation is handled by the campaign manager
    def random_single_weight_injection(pfi_model, conv_id=-1, min_val=-1, max_val=1):
        # Permenant or 'Stuck at' fault for a weight.
        loc = PyTorchFICarlaUtils.random_weight_location(pfi_model, conv_id)
        (conv_idx, k, c_in, kH, kW)  = loc
        val = PyTorchFICarlaUtils.random_value(min_val, max_val)

        print("\033[0;32m* FI Params-> Type: Weight, Loc:{}, Val:{} \033[0m\n".format(loc, val))

        return pfi_model.declare_weight_fi(conv_num=conv_idx, k=k, c=c_in, h=kH, w=kW, value=val)

    # NOTE not used, random generation is handled by the campaign manager
    def random_single_neuron_injection(pfi_model, batch=-1, conv_id=-1, min_val=-1, max_val=1):
        # Permenant or 'Stuck at' fault for a neuron output
        loc = PyTorchFICarlaUtils.random_neuron_location(pfi_model, conv_id)
        (conv_idx, c, h, w)  = loc
        val = PyTorchFICarlaUtils.random_value(min_val, max_val)

        print("\033[0;32m* FI Params-> Type: Neuron, Loc:{}, Val:{} \033[0m\n".format(loc, val))

        return pfi_model.declare_neuron_fi(batch=batch, conv_num=conv_idx, c=c, h=h, w=w, value=val)

    @staticmethod
    def single_weight_injection(pfi_model, layer, k, c, h, w, min_val, max_val, layer_name, usemask):
        value = random.randint(min_val, max_val)
        
        for name, param in pfi_model.get_original_model().named_parameters():
            if name == layer_name:
                curval = param[k,c,h,w]
        if usemask:
            value = PyTorchFICarlaUtils.floatToBits(curval.item())^value
            value = PyTorchFICarlaUtils.bitsToFloat(value)
        print("\033[0;32m* FI Params-> Type: Weight, Loc:{}, Name:{} \033[0m\n".format((layer, k, c, h, w), layer_name))
        print("Original value: {}, bit value: {}, change to value: {}".format(curval.item(), PyTorchFICarlaUtils.floatToBits(curval.item()), value))
        return pfi_model.declare_weight_fi(conv_num=layer, k=k, c=c, h=h, w=w, value=value)

    @staticmethod
    def single_neuron_injection(pfi_model, layer, c, h, w, min_val, max_val, layer_name, usemask):
        value = random.randint(min_val, max_val)
        print("\033[0;32m* FI Params-> Type: Weight, Loc:{}, Val:{} \033[0m\n".format((layer, c, h, w), value))
        return pfi_model.declare_neuron_fi(batch=0, conv_num=layer, c=c, h=h, w=w, value=value)

    @staticmethod
    def print_pfi_model(pfi_model):
        print("creating network features and weights size dump file at ./")
        with open("./net_features_size.csv", 'w') as features_file:
            features_file.write("idx,c,h,w,name\n")
        
        with open("./net_weights_size.csv", 'w') as weights_file:
            weights_file.write("idx,k,c,h,w,name\n")
        
        count = 0
        conv2d_namelist = list()
        for name, layer in pfi_model.get_original_model().named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                name += ".weight"
                conv2d_namelist.append(name)
                sizes = layer.weight.size()
                
                # Layer Weight Dim
                k = sizes[0]
                c = sizes[1]
                h = sizes[2]
                w = sizes[3]
                line = "{},{},{},{},{},{}\n".format(count,k,c,h,w,name)
                with open("./net_weights_size.csv", "a") as weights_file:
                    weights_file.write(line)
                count += 1

        count = 0
        for name, param in pfi_model.get_original_model().named_parameters():
            if name in conv2d_namelist:
                # Layer Output Dim
                c = pfi_model.get_fmaps_num(count)
                w = pfi_model.get_fmaps_W(count)
                h = pfi_model.get_fmaps_H(count)
                line = "{},{},{},{},{}\n".format(count,c,h,w,name)
                with open("./net_features_size.csv", "a") as features_file:
                    features_file.write(line)
                count+=1

        print("Total conv:",pfi_model.get_total_conv())
        print("Total conv from instance checks:", len(conv2d_namelist))
        print("Total conv by feature map:", count)
        assert(pfi_model.get_total_conv() == count == len(conv2d_namelist))

    @staticmethod
    def get_weight_distribution(pfi_model):
        n_bins = 100
        conv_weights = np.array([])
        for name, param in pfi_model.get_original_model().named_parameters():
            if PyTorchFICarlaUtils.is_conv_weight_layer(name):
                conv_weights = np.append(conv_weights, param.data.cpu().numpy())

        fig, ax = plt.subplots(1,1)
        ax.hist(conv_weights, bins=n_bins)
        ax.set_yscale('log')
        ax.set_title('Weight Distribution for Conv Layers (ImageAgent)')
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Count (log scale)")
        plt.show()