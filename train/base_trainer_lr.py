import os.path as osp

import torch.utils.data as data
import torch

# from utils import load_ucmh, load_data
# from dataset import MyDataset
# from utils import compress, calculate_top_map

import numpy as np
from dataset.dataset import load_dataset, CustomDataSet
from model.expanded_model import ExpandedImgNet, ExpandedTxtNet
import bisect
import pickle
from utils import get_model_feature, calculate_top_map_cuda, compress_cuda, compress_cuda_composite, calculate_top_map_cuda_u
from model.model_load import PretrainedModel, UCMH_MODEL, CMH_MODEL
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
import os
from sklearn.cluster import KMeans

IMAGE = 'image'
TEXT = 'text'
LABEL = 'label'

class Base_Trainer:
    def __init__(self, args, logger) -> None:
        self.args = args
        
        self.config = args
        # 加载训练数据
        self.I_tr_target, self.T_tr_target, self.L_tr_target = load_dataset(dataname=self.config.dataname, split_ratio=self.config.split_ratio, train_mode='target')

        self.best = 0

        # 加载测试数据（four task）
        # source
        self.I_te_source, self.I_db_source, self.T_te_source, self.T_db_source, self.L_te_source, self.L_db_source = load_dataset(
            dataname=self.config.dataname, split_ratio=self.config.split_ratio, test_mode='source', dataset_mode='source')
        # target
        self.I_te_target, self.I_db_target, self.T_te_target, self.T_db_target, self.L_te_target, self.L_db_target = load_dataset(
            dataname=self.config.dataname, split_ratio=self.config.split_ratio, test_mode='target', dataset_mode='target')

        if self.config.dataname == 'NUSWIDE' or self.config.dataname == 'MSCOCO':
            # target only
            self.I_te_target_only, self.I_db_target_only, self.T_te_target_only, self.T_db_target_only, self.L_te_target_only, self.L_db_target_only = load_dataset(
                dataname=self.config.dataname, split_ratio=self.config.split_ratio, test_mode='target_only', dataset_mode='target_only')


        # 构建dataloader
        self.build_dataloader()

        # 加载pre-trained模型
        self.CodeNet_I, self.CodeNet_T = PretrainedModel().get_model(name=self.config.model, code_length=self.config.bits, dataset=self.config.dataname)

        self.CodeNet_I_bk, self.CodeNet_T_bk = copy.deepcopy(self.CodeNet_I), copy.deepcopy(self.CodeNet_T)
        
        self.CodeNet_I_origin, self.CodeNet_T_origin = copy.deepcopy(self.CodeNet_I), copy.deepcopy(self.CodeNet_T)


        if self.config.model == 'djsrh' or self.config.model == 'jdsh':  
            self.Expand_CodeNet_I = ExpandedImgNet(self.CodeNet_I, expand_size=[int(self.config.bits/8)])
            self.Expand_CodeNet_T = ExpandedTxtNet(self.CodeNet_T, expand_size=[512, int(int(self.config.bits/8))])
        
        if self.config.model == 'dgcpn':
            self.Expand_CodeNet_I = ExpandedImgNet(self.CodeNet_I, expand_size=[512, int(self.config.bits/8)])
            self.Expand_CodeNet_T = ExpandedTxtNet(self.CodeNet_T, expand_size=[512, int(int(self.config.bits)/8)])
            
        if self.config.model == 'cirh':
            self.Expand_CodeNet_I = ExpandedImgNet(self.CodeNet_I, expand_size=[512, int(self.config.bits/8)])
            self.Expand_CodeNet_T = ExpandedTxtNet(self.CodeNet_T, expand_size=[64, int(self.config.bits/8)])            

        if self.config.model == 'dcmh':  
            self.Expand_CodeNet_I = ExpandedImgNet(self.CodeNet_I, expand_size=[int(self.config.bits/8)])
            self.Expand_CodeNet_T = ExpandedTxtNet(self.CodeNet_T, expand_size=[1024, int(self.config.bits/8)])      
                      
        if self.config.model == 'cpah':  
            self.Expand_CodeNet_I = ExpandedImgNet(self.CodeNet_I, expand_size=[64, 64, int(self.config.bits/8)])
            self.Expand_CodeNet_T = ExpandedTxtNet(self.CodeNet_T, expand_size=[64, 64, int(self.config.bits/8)])    



        
        if self.args.log != None and self.args.log_best != None:
            if self.args.mode != None:
                self.log_path = './results/{}/{}_{}.txt'.format(self.args.mode, self.config.model, self.args.log)
                self.log_best_path = './results/{}/{}_{}.txt'.format(self.args.mode, self.config.model, self.args.log_best)
            else:
                self.log_path = './results/{}_{}.txt'.format(self.config.model, self.args.log)
                self.log_best_path = './results/{}_{}.txt'.format(self.config.model, self.args.log_best)
            pass
        else:
            self.log_path = './results/{}_0304.txt'.format(self.config.model)
            self.log_best_path = './results/{}_best_0304.txt'.format(self.config.model)

    def build_dataloader(self, ):
        test_images = {'source': self.I_te_source, 'target': self.I_te_target}
        test_texts = {'source': self.T_te_source, 'target': self.T_te_target}
        test_labels = {'source': self.L_te_source, 'target': self.L_te_target}

        database_images = {'source': self.I_db_source, 'target': self.I_db_target}
        database_texts = {'source': self.T_db_source, 'target': self.T_db_target}
        database_labels = {'source': self.L_db_source, 'target': self.L_db_target}

        if self.config.dataname == 'NUSWIDE' or self.config.dataname == 'MSCOCO':
            test_images.update({'target_only': self.I_te_target_only})
            test_texts.update({'target_only': self.T_te_target_only})
            test_labels.update({'target_only': self.L_te_target_only})

            database_images.update({'target_only': self.I_db_target_only})
            database_texts.update({'target_only': self.T_db_target_only})
            database_labels.update({'target_only': self.L_db_target_only})
            mode_list = ['source', 'target', 'target_only']
        else:
            mode_list = ['source', 'target']

        self.test_dataset = {x: CustomDataSet(images=test_images[x], texts=test_texts[x], labels=test_labels[x]) for x in mode_list}
        self.database_dataset = {
            x: CustomDataSet(images=database_images[x], texts=database_texts[x], labels=database_labels[x]) for x in mode_list}

        # train
        self.train_dataset = CustomDataSet(self.I_tr_target, self.T_tr_target, self.L_tr_target)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                                 batch_size=self.args.BATCH_SIZE,
                                                 drop_last=self.args.DROP_LAST,
                                                 shuffle=True,
                                                 num_workers=4)
        

        self.config.batch_size = 512 # 测试时batch_size
        self.evaluate_dataloader = dict()


        for name in mode_list:
            self.evaluate_dataloader['{}_test'.format(name)] = DataLoader(dataset=self.test_dataset[name], batch_size=self.config.batch_size, shuffle=False, num_workers=2)
            self.evaluate_dataloader['{}_database'.format(name)] = DataLoader(dataset=self.database_dataset[name], batch_size=self.config.batch_size, shuffle=False, num_workers=2)


    def train(self):
        pass

    
    # 得到hash模型的特征
    def get_hash_model_feature(self, train_data, hash_model):
        
        if isinstance(train_data, list):
            #　local feature
            tr_data_list = []
            with torch.no_grad():
                for data in  train_data:
                    _, tr_data = hash_model(data)
                    tr_data_list.append(tr_data)
            return tr_data_list        
            
        else:
            #　global feature    
            train_data = torch.as_tensor(train_data).cuda()
            with torch.no_grad():
                _, tr_data = hash_model(train_data)
            return tr_data
    
    #　量化
    def compress(self, data):
        data_code = torch.sign(data)
        return data_code
    
    # 计算hamming距离
    def calc_hamming_dist(self, B1, B2):
        q = B2.shape[1]
        if len(B1.shape) < 2:
            B1 = B1.unsqueeze(0)
        distH = 0.5 * (q - B1.mm(B2.t()))
        return distH


    
    def eval_base(self, mode='source', save_log = False, epoch=0):
        self.Expand_CodeNet_I.cuda().eval()
        self.Expand_CodeNet_T.cuda().eval()

        if mode == 'source':
            self.CodeNet_I_bk.cuda().eval()
            self.CodeNet_T_bk.cuda().eval()

            re_BI_source, re_BT_source, re_L_source, qu_BI, qu_BT, qu_L = \
                compress_cuda_composite(self.evaluate_dataloader['source_database'], None, self.evaluate_dataloader['source_test'], self.CodeNet_I_bk, self.CodeNet_T_bk, self.Expand_CodeNet_I, self.Expand_CodeNet_T, bits=self.config.bits, mode='source')
            
            MAP_I2T = calculate_top_map_cuda_u(qu_B=qu_BI, re_B=re_BT_source, qu_L=qu_L, re_L=re_L_source,  bits=self.config.bits, topk=50)
            MAP_T2I = calculate_top_map_cuda_u(qu_B=qu_BT, re_B=re_BI_source, qu_L=qu_L, re_L=re_L_source,  bits=self.config.bits, topk=50)
            
            self.result['source'] = [MAP_I2T.item(), MAP_T2I.item()]

            now = datetime.datetime.now()
            print('[%s %s-%d-%s] MAP@I2T = %.4f, MAP@T2I = %.4f' % (
                now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, mode, MAP_I2T, MAP_T2I))
            
            if save_log:
                with open(self.log_path, 'a+') as f:
                    f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, MAP_I2T, MAP_T2I))    
            

        if mode == 'target':

            re_BI_target, re_BT_target, re_L_target, qu_BI, qu_BT, qu_L = \
                compress_cuda_composite(None, self.evaluate_dataloader['target_database'], self.evaluate_dataloader['target_test'], None, None, self.Expand_CodeNet_I, self.Expand_CodeNet_T, bits=self.config.bits, mode='target')

            MAP_I2T = calculate_top_map_cuda_u(qu_B=qu_BI, re_B=re_BT_target, qu_L=qu_L, re_L=re_L_target, bits=self.config.bits, topk=50)
            MAP_T2I = calculate_top_map_cuda_u(qu_B=qu_BT, re_B=re_BI_target, qu_L=qu_L, re_L=re_L_target, bits=self.config.bits, topk=50)

            self.result['target'] = [MAP_I2T.item(), MAP_T2I.item()]
            
            now = datetime.datetime.now()
            print('[%s %s-%d-%s] MAP@I2T = %.4f, MAP@T2I = %.4f' % (
                now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, mode, MAP_I2T, MAP_T2I))

            if save_log:            
                with open(self.log_path, 'a+') as f:
                    # f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (settings.DATASET_NAME, settings.CODE_LEN, MAP_I2T, MAP_T2I))
                    f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, MAP_I2T, MAP_T2I))
        
        if mode == 'target_only':

            re_BI_target, re_BT_target, re_L_target, qu_BI, qu_BT, qu_L = \
                compress_cuda_composite(None, self.evaluate_dataloader['target_only_database'], self.evaluate_dataloader['target_only_test'], None, None, self.Expand_CodeNet_I, self.Expand_CodeNet_T, bits=self.config.bits, mode='target_only')

            MAP_I2T = calculate_top_map_cuda_u(qu_B=qu_BI, re_B=re_BT_target, qu_L=qu_L, re_L=re_L_target, bits=self.config.bits, topk=50)
            MAP_T2I = calculate_top_map_cuda_u(qu_B=qu_BT, re_B=re_BI_target, qu_L=qu_L, re_L=re_L_target, bits=self.config.bits, topk=50)

            self.result['target_only'] = [MAP_I2T.item(), MAP_T2I.item()]
            
            now = datetime.datetime.now()
            print('[%s %s-%d-%s] MAP@I2T = %.4f, MAP@T2I = %.4f' % (
                now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, mode, MAP_I2T, MAP_T2I))
            
            if save_log:   
                with open(self.log_path, 'a+') as f:
                    # f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (settings.DATASET_NAME, settings.CODE_LEN, MAP_I2T, MAP_T2I))
                    f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, MAP_I2T, MAP_T2I))
        
        if mode == 'mix_to':
            self.CodeNet_I_bk.cuda().eval()
            self.CodeNet_T_bk.cuda().eval()

            re_BI_source, re_BT_source, re_L_source, re_BI_target, re_BT_target, re_L_target, re_L, qu_BI, qu_BT, qu_L = \
                compress_cuda_composite(self.evaluate_dataloader['source_database'], self.evaluate_dataloader['target_only_database'], self.evaluate_dataloader['target_only_test'], self.CodeNet_I_bk, self.CodeNet_T_bk, self.Expand_CodeNet_I, self.Expand_CodeNet_T, bits=self.config.bits, mode='mix')

            MAP_I2T = calculate_top_map_cuda_u(qu_B=qu_BI, re_B=re_BT_source, qu_L=qu_L, re_L=re_L, re_B2=re_BT_target, bits=self.config.bits, topk=50)
            MAP_T2I = calculate_top_map_cuda_u(qu_B=qu_BT, re_B=re_BI_source, qu_L=qu_L, re_L=re_L, re_B2=re_BI_target, bits=self.config.bits, topk=50)

            self.result['mix_to'] = [MAP_I2T.item(), MAP_T2I.item()]
            
            now = datetime.datetime.now()
            print('[%s %s-%d-%s] MAP@I2T = %.4f, MAP@T2I = %.4f' % (
                now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, mode, MAP_I2T, MAP_T2I))
            
            if save_log:       
                with open(self.log_path, 'a+') as f:
                    # f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (settings.DATASET_NAME, settings.CODE_LEN, MAP_I2T, MAP_T2I))
                    f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, MAP_I2T, MAP_T2I))

    def eval(self, save_log=False, epoch=0):
        
        self.result = dict()
        
        print('-----------source-------------')
        self.eval_base(mode='source', save_log=save_log, epoch=epoch)
        
        print('-----------target-------------')
        self.eval_base(mode='target', save_log=save_log, epoch=epoch)
        # print('-----------cross-------------')
        # self.eval_base(mode='cross')
        # print('-----------mix-------------')
        # self.eval_base(mode='mix')
        
        if self.config.dataname == 'NUSWIDE' or self.config.dataname == 'MSCOCO':  
        
            print('-----------target_only-------------')
            self.eval_base(mode='target_only', save_log=save_log, epoch=epoch)
            
            print('-----------mix_to-------------')
            self.eval_base(mode='mix_to', save_log=save_log, epoch=epoch)
  
        if save_log:
            with open(self.log_path, 'a+') as f:
                # f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (settings.DATASET_NAME, settings.CODE_LEN, MAP_I2T, MAP_T2I))
                f.write('\n\n\n')   
        
            tmp_map = 0
            now = datetime.datetime.now()
            for k,v in self.result.items():
                tmp_map = tmp_map + v[0] + v[1]
            
            if tmp_map > self.best:
                self.best = tmp_map
                with open(self.log_best_path, 'a+') as f:
                    for k, v in self.result.items():
                        f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, v[0], v[1]))
                    
                    f.write('\n\n')  
                    
                # 保存模型
                    
    def eval_base_baseline(self, mode='source', save_log = False, epoch=0):
            
        self.CodeNet_I_bk.cuda().eval()
        self.CodeNet_T_bk.cuda().eval()

        if mode == 'source':
            self.CodeNet_I_origin.cuda().eval()
            self.CodeNet_T_origin.cuda().eval()

            re_BI_source, re_BT_source, re_L_source, qu_BI, qu_BT, qu_L = \
                compress_cuda_composite(self.evaluate_dataloader['source_database'], None, self.evaluate_dataloader['source_test'], self.CodeNet_I_origin, self.CodeNet_T_origin, self.CodeNet_I_bk, self.CodeNet_T_bk, bits=self.config.bits, mode='source')
            
            MAP_I2T = calculate_top_map_cuda_u(qu_B=qu_BI, re_B=re_BT_source, qu_L=qu_L, re_L=re_L_source,  bits=self.config.bits, topk=50)
            MAP_T2I = calculate_top_map_cuda_u(qu_B=qu_BT, re_B=re_BI_source, qu_L=qu_L, re_L=re_L_source,  bits=self.config.bits, topk=50)
            
            self.result['source'] = [MAP_I2T.item(), MAP_T2I.item()]

            now = datetime.datetime.now()
            print('[%s %s-%d-%s] MAP@I2T = %.4f, MAP@T2I = %.4f' % (
                now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, mode, MAP_I2T, MAP_T2I))
            
            if save_log:
                with open(self.log_path, 'a+') as f:
                    # f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (settings.DATASET_NAME, settings.CODE_LEN, MAP_I2T, MAP_T2I))
                    f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, MAP_I2T, MAP_T2I))    
            

        if mode == 'target':

            re_BI_target, re_BT_target, re_L_target, qu_BI, qu_BT, qu_L = \
                compress_cuda_composite(None, self.evaluate_dataloader['target_database'], self.evaluate_dataloader['target_test'], None, None, self.CodeNet_I_bk, self.CodeNet_T_bk, bits=self.config.bits, mode='target')

            MAP_I2T = calculate_top_map_cuda_u(qu_B=qu_BI, re_B=re_BT_target, qu_L=qu_L, re_L=re_L_target, bits=self.config.bits, topk=50)
            MAP_T2I = calculate_top_map_cuda_u(qu_B=qu_BT, re_B=re_BI_target, qu_L=qu_L, re_L=re_L_target, bits=self.config.bits, topk=50)

            self.result['target'] = [MAP_I2T.item(), MAP_T2I.item()]
            
            now = datetime.datetime.now()
            print('[%s %s-%d-%s] MAP@I2T = %.4f, MAP@T2I = %.4f' % (
                now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, mode, MAP_I2T, MAP_T2I))

            if save_log:            
                with open(self.log_path, 'a+') as f:
                    # f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (settings.DATASET_NAME, settings.CODE_LEN, MAP_I2T, MAP_T2I))
                    f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, MAP_I2T, MAP_T2I))
        
        if mode == 'target_only':

            re_BI_target, re_BT_target, re_L_target, qu_BI, qu_BT, qu_L = \
                compress_cuda_composite(None, self.evaluate_dataloader['target_only_database'], self.evaluate_dataloader['target_only_test'], None, None, self.CodeNet_I_bk, self.CodeNet_T_bk, bits=self.config.bits, mode='target_only')

            MAP_I2T = calculate_top_map_cuda_u(qu_B=qu_BI, re_B=re_BT_target, qu_L=qu_L, re_L=re_L_target, bits=self.config.bits, topk=50)
            MAP_T2I = calculate_top_map_cuda_u(qu_B=qu_BT, re_B=re_BI_target, qu_L=qu_L, re_L=re_L_target, bits=self.config.bits, topk=50)

            self.result['target_only'] = [MAP_I2T.item(), MAP_T2I.item()]
            
            now = datetime.datetime.now()
            print('[%s %s-%d-%s] MAP@I2T = %.4f, MAP@T2I = %.4f' % (
                now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, mode, MAP_I2T, MAP_T2I))
            
            if save_log:   
                with open(self.log_path, 'a+') as f:
                    # f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (settings.DATASET_NAME, settings.CODE_LEN, MAP_I2T, MAP_T2I))
                    f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, MAP_I2T, MAP_T2I))
        
        if mode == 'mix_to':
            self.CodeNet_I_bk.cuda().eval()
            self.CodeNet_T_bk.cuda().eval()

            re_BI_source, re_BT_source, re_L_source, re_BI_target, re_BT_target, re_L_target, re_L, qu_BI, qu_BT, qu_L = \
                compress_cuda_composite(self.evaluate_dataloader['source_database'], self.evaluate_dataloader['target_only_database'], self.evaluate_dataloader['target_only_test'], self.CodeNet_I_origin, self.CodeNet_T_origin, self.CodeNet_I_bk, self.CodeNet_T_bk, bits=self.config.bits, mode='mix')

            MAP_I2T = calculate_top_map_cuda_u(qu_B=qu_BI, re_B=re_BT_source, qu_L=qu_L, re_L=re_L, re_B2=re_BT_target, bits=self.config.bits, topk=50)
            MAP_T2I = calculate_top_map_cuda_u(qu_B=qu_BT, re_B=re_BI_source, qu_L=qu_L, re_L=re_L, re_B2=re_BI_target, bits=self.config.bits, topk=50)

            self.result['mix_to'] = [MAP_I2T.item(), MAP_T2I.item()]
            
            now = datetime.datetime.now()
            print('[%s %s-%d-%s] MAP@I2T = %.4f, MAP@T2I = %.4f' % (
                now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, mode, MAP_I2T, MAP_T2I))
            
            
            if save_log:       
                with open(self.log_path, 'a+') as f:
                    # f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (settings.DATASET_NAME, settings.CODE_LEN, MAP_I2T, MAP_T2I))
                    f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, MAP_I2T, MAP_T2I))                
    def eval_baseline(self, save_log=False, epoch=0):
        
        self.result = dict()
        
        print('-----------source-------------')
        self.eval_base_baseline(mode='source', save_log=save_log, epoch=epoch)
        
        print('-----------target-------------')
        self.eval_base_baseline(mode='target', save_log=save_log, epoch=epoch)
        
        if self.config.dataname == 'NUSWIDE' or self.config.dataname == 'MSCOCO':  
        
            print('-----------target_only-------------')
            self.eval_base_baseline(mode='target_only', save_log=save_log, epoch=epoch)
            
            print('-----------mix_to-------------')
            self.eval_base_baseline(mode='mix_to', save_log=save_log, epoch=epoch)
  
        if save_log:
            with open(self.log_path, 'a+') as f:
                f.write('\n\n\n')   
        
            tmp_map = 0
            now = datetime.datetime.now()
            for k,v in self.result.items():
                tmp_map = tmp_map + v[0] + v[1]
            
            if tmp_map > self.best:
                self.best = tmp_map
                with open(self.log_best_path, 'a+') as f:
                    for k, v in self.result.items():
                        f.write('[%s %s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (now.strftime("%Y-%m-%d %H:%M:%S"), self.config.dataname, self.config.bits, epoch, v[0], v[1]))
                    
                    f.write('\n\n') 
        
        return self.result  
                
    def performance_eval(self):
        self.imgNet.eval().cuda()
        self.txtNet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_cuda(self.retrieval_dataloader, self.test_dataloader, self.imgNet, self.txtNet)

        MAP_I2T = calculate_top_map_cuda(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map_cuda(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        return MAP_I2T, MAP_T2I


    def save_checkpoints(self, epoch):
        ckp_path = osp.join('/data/WangBoWen/OpenSet/expand_model/openset37/ucmh/', self.args.model, '{}_{}_{}bit_best_epoch.pth'.format(self.args.model.upper(), self.args.dataname, self.args.bits))
        obj = {
            'ImgNet': self.Expand_CodeNet_I.state_dict(),
            'TxtNet': self.Expand_CodeNet_T.state_dict(),
            'epoch': epoch,
        }
        torch.save(obj, ckp_path)