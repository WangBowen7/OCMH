import importlib

import torch
import os
import torch.nn as nn

UCMH_MODEL = ['djsrh', 'jdsh', 'dgcpn', 'cirh']
CMH_MODEL = ['dcmh', 'cpah']



class PretrainedModel:

    def __init__(self):

        self.model_name = {
            'flickr': 'MIRFlickr',
            'nuswide': 'NUSWIDE',
            'coco': 'MSCOCO'
        }
        pass


    def get_finetune_model(self, name, code_length, dataset=None, split_ratio=37, model_type='finetune'):
        if name in CMH_MODEL:
            module = importlib.import_module(f'.{name}', package='model.cmh')
            image_model = getattr(module, 'ImgNet')(code_len = code_length)
            text_model = getattr(module, 'TxtNet')(code_len = code_length)

            base_path = '/data/WangBoWen/OpenSet/'
            path = '{}_model/openset{}/cmh/{}/'.format(model_type, split_ratio, name)
            file_name = '{}_{}_{}bit_best_epoch.pth'.format(name.upper(), dataset, str(code_length))
            model_state_dict = os.path.join(base_path, path, file_name)

            return image_model, text_model

        elif name in UCMH_MODEL:
            module = importlib.import_module(f'.{name}', package='model.ucmh')
            image_model = getattr(module, 'ImgNet')(code_len = code_length)
            text_model = getattr(module, 'TxtNet')(code_len = code_length)

            base_path = '/data/WangBoWen/OpenSet/'
            path = '{}_model/test/ucmh/{}/'.format(model_type, split_ratio, name)

            file_name = '{}_{}_{}bit_best_epoch.pth'.format(name.upper(), dataset, str(code_length))

            model_state_dict = os.path.join(base_path, path, file_name)
            return image_model, text_model

        else:
            raise ValueError("No this model")
        
        
    def get_model(self, name, code_length, dataset=None, split_ratio=37, model_type='pretrained'):
        if name in CMH_MODEL:
            module = importlib.import_module(f'.{name}', package='model.cmh')
            # module = importlib.import_module(f'.{name}', package='cmh')
            image_model = getattr(module, 'ImgNet')(code_len = code_length)
            text_model = getattr(module, 'TxtNet')(code_len = code_length)

            base_path = '/data/WangBoWen/OpenSet/'
            path = '{}_model/openset{}/cmh/{}/'.format(model_type, split_ratio, name)
            # file_name = '{}_{}_{}bit_best_epoch.pth'.format(name.upper(), self.model_name[dataset], str(code_length))
            file_name = '{}_{}_{}bit_best_epoch.pth'.format(name.upper(), dataset, str(code_length))
            model_state_dict = os.path.join(base_path, path, file_name)
            
            if name == 'cpah':
                state = torch.load(model_state_dict, map_location='cpu') # 模型参数加载到cpu
                image_model.load_state_dict(state['ImgNet'], strict=False)
                text_model.load_state_dict(state['TxtNet'], strict=False)
                image_model.hashF = nn.Sequential(*(list(image_model.image_module.children()) + list(image_model.hash_module.children())))
                image_model.image_module = nn.Sequential()
                image_model.hash_module = nn.Sequential()
                image_model.mask_module = nn.Sequential()
                
                text_model.hashF = nn.Sequential(*(list(text_model.text_module.children()) + list(text_model.hash_module.children())))
                text_model.text_module = nn.Sequential()
                text_model.hash_module = nn.Sequential()
                text_model.mask_module = nn.Sequential()
            else:
                state = torch.load(model_state_dict, map_location='cpu') # 模型参数加载到cpu
                image_model.load_state_dict(state['ImgNet'])
                text_model.load_state_dict(state['TxtNet'])

            return image_model, text_model

        elif name in UCMH_MODEL:
            module = importlib.import_module(f'.{name}', package='model.ucmh')
            image_model = getattr(module, 'ImgNet')(code_len = code_length)
            text_model = getattr(module, 'TxtNet')(code_len = code_length)

            base_path = '/data/WangBoWen/OpenSet/'
            path = '{}_model/openset{}/ucmh/{}'.format(model_type, split_ratio, name)


            file_name = '{}_{}_{}bit_best_epoch.pth'.format(name.upper(), dataset, str(code_length))
            model_state_dict = os.path.join(base_path, path, file_name)

            state = torch.load(model_state_dict, map_location='cpu') # 模型参数加载到cpu
            image_model.load_state_dict(state['ImgNet'])
            text_model.load_state_dict(state['TxtNet'])
            return image_model, text_model

        else:
            raise ValueError("No this model")
        
