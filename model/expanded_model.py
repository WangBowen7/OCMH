import copy

import torch
import torch.nn as nn

from collections import OrderedDict
import math

class ExpandedImgNet(nn.Module):
    def __init__(self, base_mlp, expansion_factor = 2, expand_size = None):
        super(ExpandedImgNet, self).__init__()
        self.expansion_factor = expansion_factor

        self.base_state_dict = base_mlp.state_dict()
        self.expand_image_layers, self.image_mask = self.expand_layers(base_mlp, expand_size=expand_size)

        self.expand_state_dict = self.expand_image_layers.state_dict()

        self.current_expand_I = None
        self.current_origin_I = None
        self.current_I = None
        
        self.alpha = 1.0


    def save_origin_grad(self, ratio=1):
        self.origin_I_grad = OrderedDict()
        for name, param in self.expand_image_layers.named_parameters():
            self.origin_I_grad[name] = ratio * copy.deepcopy(param.grad)


    def save_expand_grad(self, ratio=1):
        self.expand_I_grad = OrderedDict()
        for name, param in self.expand_image_layers.named_parameters():
            self.expand_I_grad[name] = ratio * copy.deepcopy(param.grad)

    def merge_grad(self, weight_origin=0.9, weight_expand=0.1, weight_origin_e=0.9, weight_expand_e=0.1):
        for name, param in self.expand_image_layers.named_parameters():
            base_param_data = self.base_state_dict[name]
            base_param_shape = base_param_data.shape

            origin_I_grad_data = self.origin_I_grad[name]
            origin_I_grad_data_shape = origin_I_grad_data.shape

            expand_I_grad_data = self.expand_I_grad[name]
            expand_I_grad_data_shape = expand_I_grad_data.shape

            param_shape = param.data.shape

            # 判断是weight还是bias
            if len(param_shape) == 2:

                if (param_shape[1] - base_param_shape[1]) > 0:
                    param.data[:base_param_shape[0], base_param_shape[1]:] = 0

                # TODO 原始部分的参数
                param.grad[:base_param_shape[0], :base_param_shape[1]] = weight_origin * origin_I_grad_data[ :base_param_shape[0], :base_param_shape[1]] \
                                                                         + weight_expand * expand_I_grad_data[: base_param_shape[0], :base_param_shape[1]]

                # TODO 膨胀部分的参数 (这里也能加权)
                # param.grad[base_param_shape[0]:, :] = expand_I_grad_data[base_param_shape[0]:, :]
                param.grad[base_param_shape[0]:, :] = weight_origin_e * origin_I_grad_data[base_param_shape[0]:, :] + weight_expand_e * expand_I_grad_data[base_param_shape[0]:, :]

            elif len(param_shape) == 1:
                param.grad[:base_param_shape[0]] = weight_origin * origin_I_grad_data[:base_param_shape[0]] + weight_expand * expand_I_grad_data[:base_param_shape[0]]

                # param.grad[base_param_shape[0]:] = expand_I_grad_data[base_param_shape[0]:]
                param.grad[base_param_shape[0]:] = weight_origin_e * origin_I_grad_data[base_param_shape[0]:] + weight_expand_e * expand_I_grad_data[base_param_shape[0]:]


    def save_current(self):
        self.current_I = copy.deepcopy(self.expand_image_layers.state_dict())  # 深拷贝/内存何时释放
        # self.current_I = self.expand_image_layers.state_dict().clone().detach()


    def update_current_expand(self):
        self.current_expand_I = copy.deepcopy(self.expand_image_layers.state_dict()) # 深拷贝
        # print('r----------------')
        # print(self.current_I)
        # 参数reset
        for name, param in self.expand_image_layers.named_parameters():
            param.data = self.current_I[name]
            # print('reset')

    def update_current_origin(self):
        self.current_origin_I = copy.deepcopy(self.expand_image_layers.state_dict()) # 深拷贝

        # print(self.current_I)
        # 参数reset
        for name, param in self.expand_image_layers.named_parameters():
            param.data = self.current_I[name]

    def update_parameters(self):

        for name, param in self.expand_image_layers.named_parameters():
            base_param_data = self.base_state_dict[name]
            base_param_shape = base_param_data.shape

            current_expand_I_data = self.current_expand_I[name]
            current_expand_I_shape = current_expand_I_data.shape

            current_origin_I_data = self.current_origin_I[name]
            current_origin_I_shape = current_origin_I_data.shape

            param_shape = param.data.shape

            # 判断是weight还是bias
            if len(param_shape) == 2:

                if (param_shape[1] - base_param_shape[1]) > 0:
                    # param.data[:base_param_shape[0], :(param_shape[1]-base_param_shape[1])] = current_expand_I_data[:base_param_shape[0], :(param_shape[1]-base_param_shape[1])]
                    # param.data[:base_param_shape[0], :(param_shape[1]-base_param_shape[1])] = 0
                    param.data[:base_param_shape[0], base_param_shape[1]:] = 0

                param.data[base_param_shape[0]:, :] = current_expand_I_data[base_param_shape[0]:, :]


                param.data[:base_param_shape[0], :base_param_shape[1]] = current_origin_I_data[:base_param_shape[0], :base_param_shape[1]]


            elif len(param_shape) == 1:
                param.data[:base_param_shape[0]] = current_origin_I_data[:base_param_shape[0]]
                param.data[base_param_shape[0]:] = current_expand_I_data[base_param_shape[0]:]


    def update_parameters_mix(self, weight_origin=0.9):

        for name, param in self.expand_image_layers.named_parameters():
            base_param_data = self.base_state_dict[name]
            base_param_shape = base_param_data.shape

            current_expand_I_data = self.current_expand_I[name]
            current_expand_I_shape = current_expand_I_data.shape

            current_origin_I_data = self.current_origin_I[name]
            current_origin_I_shape = current_origin_I_data.shape

            param_shape = param.data.shape

            # 判断是weight还是bias
            if len(param_shape) == 2:

                if (param_shape[1] - base_param_shape[1]) > 0:

                    param.data[:base_param_shape[0], base_param_shape[1]:] = 0

                # TODO 原始部分的参数
                param.data[:base_param_shape[0], :base_param_shape[1]] = weight_origin * current_origin_I_data[:base_param_shape[0], :base_param_shape[1]] + (1-weight_origin) * current_expand_I_data[:base_param_shape[0], :base_param_shape[1]]

                # TODO 膨胀部分的参数
                param.data[base_param_shape[0]:, :] = current_expand_I_data[base_param_shape[0]:, :]

            elif len(param_shape) == 1:
                param.data[:base_param_shape[0]] = weight_origin * current_origin_I_data[:base_param_shape[0]] + (1 - weight_origin) * current_expand_I_data[:base_param_shape[0]]

                param.data[base_param_shape[0]:] = current_expand_I_data[base_param_shape[0]:]

    # 保持未膨胀部分参数不变，并更新膨胀模型的参数
    def reset_parameter_origin(self):
        for name, param in self.expand_image_layers.named_parameters():
            base_param_data = self.base_state_dict[name]
            base_param_shape = base_param_data.shape
            param_shape = param.data.shape

            # 判断是weight还是bias
            if len(base_param_shape) == 2:
                param.data[:base_param_shape[0], (param_shape[1]-base_param_shape[1]):] = base_param_data

                if (param_shape[1]-base_param_shape[1]) > 0:
                    param.data[:base_param_shape[0], :(param_shape[1] - base_param_shape[1])] = 0

            elif len(base_param_shape) == 1:
                param.data[:base_param_shape[0]] = base_param_data

        # 更新膨胀部分参数字典
        self.expand_state_dict = self.expand_image_layers.state_dict()

    def reset_parameter_expand(self):
        for name, param in self.expand_image_layers.named_parameters():
            base_param_data = self.base_state_dict[name]
            base_param_shape = base_param_data.shape

            expand_param_data = self.expand_state_dict[name]
            expand_param_shape = expand_param_data.shape

            param_shape = param.data.shape

            # 判断是weight还是bias
            if len(base_param_shape) == 2:
                param.data[base_param_shape[0]:, (param_shape[1]-base_param_shape[1]):] = expand_param_data[base_param_shape[0]:, (param_shape[1]-base_param_shape[1]):]

                if (param_shape[1]-base_param_shape[1]) > 0:
                    param.data[:base_param_shape[0], :(param_shape[1] - base_param_shape[1])] = 0

            elif len(base_param_shape) == 1:
                param.data[base_param_shape[0]:] = expand_param_data[base_param_shape[0]:]

    def expand_layers(self, model, expand_size = [4096, 16]):

        mask = dict()
        # expanded_layers = nn.ModuleList()
        # expanded_layers = nn.Sequential()
        layer_num = 0

        for name_layer, layer in model.named_children():
            
            
            if isinstance(layer, nn.Sequential):
                
                if len(layer) < len(expand_size):
                    continue
                
                for name_layer_c, layer_c in layer.named_children():
                    if isinstance(layer_c, nn.Linear):
                        layer_num = layer_num + 1
                        # 复制权重
                        weight_data_copy = layer_c.weight.data.clone().detach()
                        bias_data_copy = layer_c.bias.data.clone().detach()

                        if layer_num == 1:

                            if expand_size != None:
                                # 创建新的线性层
                                expanded_layer = nn.Linear(layer_c.in_features,
                                                           layer_c.out_features + expand_size[layer_num - 1])
                                # print(expanded_layer.weight.data.shape)
                                expanded_layer.weight.data[:layer_c.out_features, :] = weight_data_copy
                                # print(expanded_layer.bias.data.shape)
                                expanded_layer.bias.data[:layer_c.out_features] = bias_data_copy  # 复用原始层的偏置
                                # for expand_length in expand_size:

                            else:

                                # 创建新的线性层
                                expanded_layer = nn.Linear(layer_c.in_features,
                                                           layer_c.out_features * self.expansion_factor)
                                # print(expanded_layer.weight.data.shape)
                                expanded_layer.weight.data[:layer_c.out_features, :] = weight_data_copy
                                # print(expanded_layer.bias.data.shape)
                                expanded_layer.bias.data[:layer_c.out_features] = bias_data_copy  # 复用原始层的偏置

                            # # 添加
                            # expanded_layers.add_module(name=name_layer, module=expanded_layer)\
                            # TODO
                            setattr(layer, name_layer_c, expanded_layer)

                        else:

                            if expand_size != None:
                                # 创建新的线性层
                                expanded_layer = nn.Linear(layer_c.in_features + expand_size[layer_num - 2],
                                                           layer_c.out_features + expand_size[layer_num - 1])

                                # expanded_layer.weight.data[:layer.out_features, expand_size[layer_num-2]:] = weight_data_copy
                                expanded_layer.weight.data[:layer_c.out_features, :layer_c.in_features] = weight_data_copy

                                # expanded_layer.weight.data[:layer.out_features, :expand_size[layer_num-2]] = 0
                                expanded_layer.weight.data[:layer_c.out_features, layer_c.in_features:] = 0

                                expanded_layer.bias.data[:layer_c.out_features] = bias_data_copy
                            else:
                                # 创建新的线性层
                                expanded_layer = nn.Linear(layer_c.in_features * self.expansion_factor,
                                                           layer_c.out_features * self.expansion_factor)

                                expanded_layer.weight.data[:layer_c.out_features, :layer_c.in_features] = weight_data_copy

                                expanded_layer.weight.data[:layer_c.out_features, layer_c.in_features:] = 0

                                expanded_layer.bias.data[:layer_c.out_features] = bias_data_copy
                            # 添加
                            # expanded_layers.add_module(name=name_layer, module=expanded_layer)
                            setattr(layer, name_layer_c, expanded_layer)

                    if isinstance(layer_c, nn.BatchNorm1d):
                        # layer_c.num_features = layer_c.num_features + 64
                        # layer_c.num_features = 64
                        setattr(layer, name_layer_c, nn.BatchNorm1d(num_features=layer_c.num_features + expand_size[layer_num-1]))
            if isinstance(layer, nn.Linear):
                layer_num = layer_num + 1
                # 复制权重
                weight_data_copy = layer.weight.data.clone().detach()
                bias_data_copy = layer.bias.data.clone().detach()

                if layer_num == 1:

                    if expand_size != None:
                        # 创建新的线性层
                        expanded_layer = nn.Linear(layer.in_features, layer.out_features + expand_size[layer_num-1])
                        # print(expanded_layer.weight.data.shape)
                        expanded_layer.weight.data[:layer.out_features, :] = weight_data_copy
                        # print(expanded_layer.bias.data.shape)
                        expanded_layer.bias.data[:layer.out_features] = bias_data_copy  # 复用原始层的偏置
                        # for expand_length in expand_size:

                    else:

                        # 创建新的线性层
                        expanded_layer = nn.Linear(layer.in_features, layer.out_features * self.expansion_factor)
                        # print(expanded_layer.weight.data.shape)
                        expanded_layer.weight.data[:layer.out_features, :] = weight_data_copy
                        # print(expanded_layer.bias.data.shape)
                        expanded_layer.bias.data[:layer.out_features] = bias_data_copy  # 复用原始层的偏置

                    # # 添加
                    # expanded_layers.add_module(name=name_layer, module=expanded_layer)
                    setattr(model, name_layer, expanded_layer)

                else:

                    if expand_size != None:
                        # 创建新的线性层
                        expanded_layer = nn.Linear(layer.in_features + expand_size[layer_num-2], layer.out_features + expand_size[layer_num-1])

                        # expanded_layer.weight.data[:layer.out_features, expand_size[layer_num-2]:] = weight_data_copy
                        expanded_layer.weight.data[:layer.out_features, :layer.in_features] = weight_data_copy

                        # expanded_layer.weight.data[:layer.out_features, :expand_size[layer_num-2]] = 0
                        expanded_layer.weight.data[:layer.out_features, layer.in_features:] = 0

                        expanded_layer.bias.data[:layer.out_features] = bias_data_copy
                    else:
                        # 创建新的线性层
                        expanded_layer = nn.Linear(layer.in_features * self.expansion_factor, layer.out_features * self.expansion_factor)

                        expanded_layer.weight.data[:layer.out_features, :layer.in_features] = weight_data_copy

                        expanded_layer.weight.data[:layer.out_features, layer.in_features:] = 0

                        expanded_layer.bias.data[:layer.out_features] = bias_data_copy
                    # # 添加
                    # expanded_layers.add_module(name=name_layer, module=expanded_layer)
                    setattr(model, name_layer, expanded_layer)
            else:
                # # 添加
                # expanded_layers.add_module(name=name_layer, module=layer)
                # setattr(model, name_layer, layer)
                pass
            
            # generate mask
            if isinstance(layer, nn.Linear):
                if layer_num == 1:
                    for name_param, param in layer.named_parameters():
                        key = f'{name_layer}.{name_param}'

                        if name_param.startswith('weight'):
                            mask[key] = torch.zeros_like(expanded_layer.weight)
                            mask[key][layer.out_features:, :] = 1

                        if name_param.startswith('bias'):
                            mask[key] = torch.zeros_like(expanded_layer.bias)
                            mask[key][layer.out_features:] = 1

                else:
                    for name_param, param in layer.named_parameters():
                        key = f'{name_layer}.{name_param}'

                        if name_param.startswith('weight'):
                            mask[key] = torch.zeros_like(expanded_layer.weight)
                            mask[key][layer.out_features:, :] = 1

                        if name_param.startswith('bias'):
                            mask[key] = torch.zeros_like(expanded_layer.bias)
                            mask[key][layer.out_features:] = 1

        # print(model)
        return model, mask

    
    def set_alpha(self, epoch):
        # self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
        self.expand_image_layers.set_alpha(epoch)
        
    def forward(self, x):
        
        return self.expand_image_layers(x)


class ExpandedTxtNet(nn.Module):
    def __init__(self, base_mlp, expansion_factor = 2, expand_size = None):
        super(ExpandedTxtNet, self).__init__()
        self.expansion_factor = expansion_factor
        # self.mask = dict()

        # self.base_layers = base_mlp
        # 复制并堆叠每个线性层的权重
        # self.base_layers = self.expand_layers(base_mlp)
        self.base_state_dict = base_mlp.state_dict()
        self.expand_text_layers, self.text_mask = self.expand_layers(base_mlp, expand_size=expand_size)

        self.current_expand_T = None
        self.current_origin_T = None
        self.current_T = None
        
        self.alpha = 1.0

        self.save_current()

    def save_current(self):
        self.current_T = copy.deepcopy(self.expand_text_layers.state_dict())

    def save_origin_grad(self, ratio=1):
        self.origin_T_grad = OrderedDict()
        for name, param in self.expand_text_layers.named_parameters():
            self.origin_T_grad[name] = ratio * copy.deepcopy(param.grad)

    def save_expand_grad(self, ratio=1):
        self.expand_T_grad = OrderedDict()
        for name, param in self.expand_text_layers.named_parameters():
            self.expand_T_grad[name] = ratio * copy.deepcopy(param.grad)

    def merge_grad(self, weight_origin=0.9, weight_expand=0.1, weight_origin_e=0.9, weight_expand_e=0.1):
        for name, param in self.expand_text_layers.named_parameters():
            base_param_data = self.base_state_dict[name]
            base_param_shape = base_param_data.shape

            origin_T_grad_data = self.origin_T_grad[name]
            origin_T_grad_data_shape = origin_T_grad_data.shape

            expand_T_grad_data = self.expand_T_grad[name]
            expand_T_grad_data_shape = expand_T_grad_data.shape

            param_shape = param.data.shape

            # 判断是weight还是bias
            if len(param_shape) == 2:

                if (param_shape[1] - base_param_shape[1]) > 0:
                    param.data[:base_param_shape[0], base_param_shape[1]:] = 0

                # TODO 原始部分的参数
                param.grad[:base_param_shape[0], :base_param_shape[1]] = weight_origin * origin_T_grad_data[ :base_param_shape[0], :base_param_shape[1]] \
                                                                         + weight_expand * expand_T_grad_data[: base_param_shape[0], :base_param_shape[1]]

                # TODO 膨胀部分的参数
                # param.grad[base_param_shape[0]:, :] = expand_T_grad_data[base_param_shape[0]:, :]
                param.grad[base_param_shape[0]:, :] = weight_origin_e * origin_T_grad_data[base_param_shape[0]:, :] + weight_expand_e * expand_T_grad_data[base_param_shape[0]:, :]

            elif len(param_shape) == 1:
                param.grad[:base_param_shape[0]] = weight_origin * origin_T_grad_data[:base_param_shape[0]] + weight_expand * expand_T_grad_data[:base_param_shape[0]]

                # param.grad[base_param_shape[0]:] = expand_T_grad_data[base_param_shape[0]:]
                param.grad[base_param_shape[0]:] = weight_origin_e * origin_T_grad_data[base_param_shape[0]:] + weight_expand_e * expand_T_grad_data[base_param_shape[0]:]


    def update_current_expand(self):
        self.current_expand_T = copy.deepcopy(self.expand_text_layers.state_dict())

        # 参数reset
        for name, param in self.expand_text_layers.named_parameters():
            param.data = self.current_T[name]

    def update_current_origin(self):
        self.current_origin_T = copy.deepcopy(self.expand_text_layers.state_dict())

        # 参数reset
        for name, param in self.expand_text_layers.named_parameters():
            param.data = self.current_T[name]

    def update_parameters(self):
        for name, param in self.expand_text_layers.named_parameters():
            base_param_data = self.base_state_dict[name]
            base_param_shape = base_param_data.shape

            current_expand_T_data = self.current_expand_T[name]
            current_expand_T_shape = current_expand_T_data.shape

            current_origin_T_data = self.current_origin_T[name]
            current_origin_T_shape = current_origin_T_data.shape

            param_shape = param.data.shape

            # 判断是weight还是bias
            if len(param_shape) == 2:

                if (param_shape[1] - base_param_shape[1]) > 0:
                    # param.data[:base_param_shape[0], :(param_shape[1]-base_param_shape[1])] = current_expand_T_data[:base_param_shape[0], :(param_shape[1]-base_param_shape[1])]
                    # param.data[:base_param_shape[0], :(param_shape[1]-base_param_shape[1])] = 0
                    param.data[:base_param_shape[0], base_param_shape[1]:] = 0

                param.data[base_param_shape[0]:, :] = current_expand_T_data[base_param_shape[0]:, :]

                param.data[:base_param_shape[0], :base_param_shape[1]] = current_origin_T_data[:base_param_shape[0], :base_param_shape[1]]

            elif len(param_shape) == 1:
                param.data[:base_param_shape[0]] = current_origin_T_data[:base_param_shape[0]]
                param.data[base_param_shape[0]:] = current_expand_T_data[base_param_shape[0]:]

    def update_parameters_mix(self, weight_origin=0.9):

        for name, param in self.expand_text_layers.named_parameters():
            base_param_data = self.base_state_dict[name]
            base_param_shape = base_param_data.shape

            current_expand_T_data = self.current_expand_T[name]
            current_expand_T_shape = current_expand_T_data.shape

            current_origin_T_data = self.current_origin_T[name]
            current_origin_T_shape = current_origin_T_data.shape

            param_shape = param.data.shape

            # 判断是weight还是bias
            if len(param_shape) == 2:

                if (param_shape[1] - base_param_shape[1]) > 0:

                    param.data[:base_param_shape[0], base_param_shape[1]:] = 0

                # TODO 原始部分的参数
                param.data[:base_param_shape[0], :base_param_shape[1]] = weight_origin * current_origin_T_data[:base_param_shape[0], :base_param_shape[1]] + (1-weight_origin) * current_expand_T_data[:base_param_shape[0], :base_param_shape[1]]

                # TODO 膨胀部分的参数
                param.data[base_param_shape[0]:, :] = current_expand_T_data[base_param_shape[0]:, :]

            elif len(param_shape) == 1:
                param.data[:base_param_shape[0]] = weight_origin * current_origin_T_data[:base_param_shape[0]] + (1 - weight_origin) * current_expand_T_data[:base_param_shape[0]]

                param.data[base_param_shape[0]:] = current_expand_T_data[base_param_shape[0]:]

    def reset_parameter_orign(self):

        for name, param in self.expand_text_layers.named_parameters():
            base_param_data = self.base_state_dict[name]
            base_param_shape = base_param_data.shape
            param_shape = param.data.shape

            # 判断是weight还是bias
            # weight
            if len(base_param_shape) == 2:
                param.data[:base_param_shape[0], (param_shape[1]-base_param_shape[1]):] = base_param_data

                if (param_shape[1]-base_param_shape[1]) > 0:
                    param.data[:base_param_shape[0], :(param_shape[1] - base_param_shape[1])] = 0
            # bias
            elif len(base_param_shape) == 1:
                param.data[:base_param_shape[0]] = base_param_data

    def expand_layers(self, model, expand_size = [4096, 16]):

        mask = dict()
        # expanded_layers = nn.ModuleList()
        # expanded_layers = nn.Sequential()
        layer_num = 0

        for name_layer, layer in model.named_children():
        # for name_layer, layer in model.named_modules():
        


            if isinstance(layer, nn.Sequential):
                
                if len(layer) < len(expand_size):
                    continue
                
                for name_layer_c, layer_c in layer.named_children():
                    if isinstance(layer_c, nn.Linear):
                        layer_num = layer_num + 1
                        # 复制权重
                        weight_data_copy = layer_c.weight.data.clone().detach()
                        bias_data_copy = layer_c.bias.data.clone().detach()

                        if layer_num == 1:

                            if expand_size != None:
                                # 创建新的线性层
                                expanded_layer = nn.Linear(layer_c.in_features,
                                                           layer_c.out_features + expand_size[layer_num - 1])
                                # print(expanded_layer.weight.data.shape)
                                expanded_layer.weight.data[:layer_c.out_features, :] = weight_data_copy
                                # print(expanded_layer.bias.data.shape)
                                expanded_layer.bias.data[:layer_c.out_features] = bias_data_copy  # 复用原始层的偏置
                                # for expand_length in expand_size:

                            else:

                                # 创建新的线性层
                                expanded_layer = nn.Linear(layer_c.in_features,
                                                           layer_c.out_features * self.expansion_factor)
                                # print(expanded_layer.weight.data.shape)
                                expanded_layer.weight.data[:layer_c.out_features, :] = weight_data_copy
                                # print(expanded_layer.bias.data.shape)
                                expanded_layer.bias.data[:layer_c.out_features] = bias_data_copy  # 复用原始层的偏置

                            # # 添加
                            # expanded_layers.add_module(name=name_layer, module=expanded_layer)\
                            # TODO
                            setattr(layer, name_layer_c, expanded_layer)

                        else:

                            if expand_size != None:
                                # 创建新的线性层
                                expanded_layer = nn.Linear(layer_c.in_features + expand_size[layer_num - 2],
                                                           layer_c.out_features + expand_size[layer_num - 1])

                                # expanded_layer.weight.data[:layer.out_features, expand_size[layer_num-2]:] = weight_data_copy
                                expanded_layer.weight.data[:layer_c.out_features, :layer_c.in_features] = weight_data_copy

                                # expanded_layer.weight.data[:layer.out_features, :expand_size[layer_num-2]] = 0
                                expanded_layer.weight.data[:layer_c.out_features, layer_c.in_features:] = 0

                                expanded_layer.bias.data[:layer_c.out_features] = bias_data_copy
                            else:
                                # 创建新的线性层
                                expanded_layer = nn.Linear(layer_c.in_features * self.expansion_factor,
                                                           layer_c.out_features * self.expansion_factor)

                                expanded_layer.weight.data[:layer_c.out_features, :layer_c.in_features] = weight_data_copy

                                expanded_layer.weight.data[:layer_c.out_features, layer_c.in_features:] = 0

                                expanded_layer.bias.data[:layer_c.out_features] = bias_data_copy
                            # 添加
                            # expanded_layers.add_module(name=name_layer, module=expanded_layer)
                            setattr(layer, name_layer_c, expanded_layer)
                    
                    
                    if isinstance(layer_c, nn.BatchNorm1d):
                        setattr(layer, name_layer_c, nn.BatchNorm1d(num_features=layer_c.num_features + expand_size[layer_num-1]))
                    

            if isinstance(layer, nn.Linear):
                layer_num = layer_num + 1
                # 复制权重
                weight_data_copy = layer.weight.data.clone().detach()
                bias_data_copy = layer.bias.data.clone().detach()

                if layer_num == 1:

                    if expand_size != None:
                        # 创建新的线性层
                        expanded_layer = nn.Linear(layer.in_features, layer.out_features + expand_size[layer_num-1])
                        # print(expanded_layer.weight.data.shape)
                        expanded_layer.weight.data[:layer.out_features, :] = weight_data_copy
                        # print(expanded_layer.bias.data.shape)
                        expanded_layer.bias.data[:layer.out_features] = bias_data_copy  # 复用原始层的偏置
                        # for expand_length in expand_size:

                    else:

                        # 创建新的线性层
                        expanded_layer = nn.Linear(layer.in_features, layer.out_features * self.expansion_factor)
                        # print(expanded_layer.weight.data.shape)
                        expanded_layer.weight.data[:layer.out_features, :] = weight_data_copy
                        # print(expanded_layer.bias.data.shape)
                        expanded_layer.bias.data[:layer.out_features] = bias_data_copy  # 复用原始层的偏置

                    # # 添加
                    # expanded_layers.add_module(name=name_layer, module=expanded_layer)
                    setattr(model, name_layer, expanded_layer)

                else:

                    if expand_size != None:
                        # 创建新的线性层
                        expanded_layer = nn.Linear(layer.in_features + expand_size[layer_num-2], layer.out_features + expand_size[layer_num-1])

                        # expanded_layer.weight.data[:layer.out_features, expand_size[layer_num-2]:] = weight_data_copy
                        expanded_layer.weight.data[:layer.out_features, :layer.in_features] = weight_data_copy

                        # expanded_layer.weight.data[:layer.out_features, :expand_size[layer_num-2]] = 0
                        expanded_layer.weight.data[:layer.out_features, layer.in_features:] = 0

                        expanded_layer.bias.data[:layer.out_features] = bias_data_copy
                    else:
                        # 创建新的线性层
                        expanded_layer = nn.Linear(layer.in_features * self.expansion_factor, layer.out_features * self.expansion_factor)

                        expanded_layer.weight.data[:layer.out_features, :layer.in_features] = weight_data_copy

                        expanded_layer.weight.data[:layer.out_features, layer.in_features:] = 0

                        expanded_layer.bias.data[:layer.out_features] = bias_data_copy
                    # # 添加
                    # expanded_layers.add_module(name=name_layer, module=expanded_layer)
                    setattr(model, name_layer, expanded_layer)
            else:
                # # 添加
                # expanded_layers.add_module(name=name_layer, module=layer)
                # setattr(model, name_layer, layer)
                pass

            # generate mask
            if isinstance(layer, nn.Linear):
                if layer_num == 1:
                    for name_param, param in layer.named_parameters():
                        key = f'{name_layer}.{name_param}'

                        if name_param.startswith('weight'):
                            mask[key] = torch.zeros_like(expanded_layer.weight)
                            mask[key][layer.out_features:, :] = 1

                        if name_param.startswith('bias'):
                            mask[key] = torch.zeros_like(expanded_layer.bias)
                            mask[key][layer.out_features:] = 1

                else:
                    for name_param, param in layer.named_parameters():
                        key = f'{name_layer}.{name_param}'

                        if name_param.startswith('weight'):
                            mask[key] = torch.zeros_like(expanded_layer.weight)
                            mask[key][layer.out_features:, :] = 1

                        if name_param.startswith('bias'):
                            mask[key] = torch.zeros_like(expanded_layer.bias)
                            mask[key][layer.out_features:] = 1

        # print(model)
        return model, mask

    def set_alpha(self, epoch):
        # self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
        self.expand_text_layers.set_alpha(epoch)
        
    def forward(self, x):
        return self.expand_text_layers(x)


if __name__ == '__main__':

    pass