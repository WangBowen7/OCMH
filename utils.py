import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
import logging
import os.path as osp


def p_topK(qB, rB, query_label, retrieval_label, K=None):
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p

def compress_wiki(train_loader, test_loader, modeli, modelt, train_dataset, test_dataset, classes=10):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(train_loader):
        var_data_I = Variable(data_I.cuda())
        _,_,code_I = modeli(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(target)
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = modelt(var_data_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        _,_,code_I = modeli(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(target)
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = modelt(var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.eye(classes)[np.array(re_L)]

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.eye(classes)[np.array(qu_L)]
    
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compress(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    with torch.no_grad():
        for _, (data_I, data_T, _, _) in enumerate(train_loader):
            var_data_I = Variable(data_I.cuda())
            _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            re_BI.extend(code_I.cpu().data.numpy())

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])

    with torch.no_grad():
        for _, (data_I, data_T, _, _) in enumerate(test_loader):
            var_data_I = Variable(data_I.cuda())
            _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            qu_BI.extend(code_I.cpu().data.numpy())

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = train_dataset.labels

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = test_dataset.labels
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compress_cuda_composite(train_loader_source, train_loader_target, test_loader, model_I, model_T, model_I_expand, model_T_expand , bits=None, mode=None):

    re_BI_source = list([])
    re_BT_source = list([])
    re_L_source = list([])
    # origin source code: 不变
    if mode == 'source' or mode == 'cross' or mode == 'mix':
        with torch.no_grad():
            for _, (data_I, data_T, data_L, _) in enumerate(train_loader_source):
                var_data_I = Variable(data_I.cuda())
                _, code_I = model_I(var_data_I)
                code_I = torch.sign(code_I)
                re_BI_source.extend(code_I)

                var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
                # var_data_T = Variable(torch.FloatTensor(data_T.numpy()).unsqueeze(1).unsqueeze(-1).cuda())
                _, code_T = model_T(var_data_T)
                code_T = torch.sign(code_T)
                re_BT_source.extend(code_T)
                
                re_L_source.extend(data_L.type(torch.float32).cuda())
                # re_L_source.extend(data_L)

    re_BI_target = list([])
    re_BT_target = list([])
    re_L_target = list([])
    if mode == 'mix' or mode == 'target' or mode == 'target_only':

        with torch.no_grad():
            for _, (data_I, data_T, data_L, _) in enumerate(train_loader_target):
                var_data_I = Variable(data_I.cuda())
                _, code_I = model_I_expand(var_data_I)
                code_I = torch.sign(code_I)
                re_BI_target.extend(code_I)

                var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
                # var_data_T = Variable(torch.FloatTensor(data_T.numpy()).unsqueeze(1).unsqueeze(-1).cuda())
                _, code_T = model_T_expand(var_data_T)
                code_T = torch.sign(code_T)
                re_BT_target.extend(code_T)
                
                re_L_target.extend(data_L.type(torch.float32).cuda())
                # re_L_target.extend(data_L)


    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])

    with torch.no_grad():
        for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
            var_data_I = Variable(data_I.cuda())
            _, code_I = model_I_expand(var_data_I)
            code_I = torch.sign(code_I)
            qu_BI.extend(code_I)

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            # var_data_T = Variable(torch.FloatTensor(data_T.numpy()).unsqueeze(1).unsqueeze(-1).cuda())
            _, code_T = model_T_expand(var_data_T)
            code_T = torch.sign(code_T)
            qu_BT.extend(code_T)
            
            qu_L.extend(data_L.type(torch.float32).cuda())
            # qu_L.extend(data_L)

    qu_BI = torch.stack(qu_BI)
    qu_BT = torch.stack(qu_BT)
    qu_L = torch.stack(qu_L)

    # print('----qu_L----')
    # print(qu_L)
    
    if mode == 'source' or mode == 'cross':
        re_BI_source = torch.stack(re_BI_source)
        re_BT_source = torch.stack(re_BT_source)
        re_L_source = torch.stack(re_L_source)

        qu_BI = qu_BI[:, :bits]
        qu_BT = qu_BT[:, :bits]
        
        # print(re_L_source)
        return re_BI_source, re_BT_source, re_L_source, qu_BI, qu_BT, qu_L

    if mode == 'mix':
        re_BI_source = torch.stack(re_BI_source)
        re_BT_source = torch.stack(re_BT_source)
        re_L_source = torch.stack(re_L_source)

        re_BI_target = torch.stack(re_BI_target)
        re_BT_target = torch.stack(re_BT_target)
        re_L_target = torch.stack(re_L_target)

        re_L = torch.cat([re_L_source, re_L_target], dim=0)
        return re_BI_source, re_BT_source, re_L_source, re_BI_target, re_BT_target, re_L_target, re_L, qu_BI, qu_BT, qu_L

    if mode == 'target' or mode == 'target_only':
        re_BI_target = torch.stack(re_BI_target)
        re_BT_target = torch.stack(re_BT_target)
        re_L_target = torch.stack(re_L_target)
        return re_BI_target, re_BT_target, re_L_target, qu_BI, qu_BT, qu_L



def compress_cuda(train_loader, test_loader, model_I, model_T, source_retrieval_index = None, bits=None, mode=None):

    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    with torch.no_grad():
        for _, (data_I, data_T, data_L, _) in enumerate(train_loader):
            var_data_I = Variable(data_I.cuda())
            _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            # re_BI.extend(code_I.cpu().data.numpy())
            re_BI.extend(code_I)

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            # re_BT.extend(code_T.cpu().data.numpy())
            re_BT.extend(code_T)
            # re_L.extend(data_L.cpu().data.numpy())
            re_L.extend(data_L)

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])

    with torch.no_grad():
        for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
            var_data_I = Variable(data_I.cuda())
            _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            # qu_BI.extend(code_I.cpu().data.numpy())
            qu_BI.extend(code_I)

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            # qu_BT.extend(code_T.cpu().data.numpy())
            qu_BT.extend(code_T)
            # qu_L.extend(data_L.cpu().data.numpy())
            qu_L.extend(data_L)

    re_BI = torch.stack(re_BI)
    re_BT = torch.stack(re_BT)
    re_L = torch.stack(re_L)
    if source_retrieval_index is not None:
        re_BI[source_retrieval_index, bits:] = 1
        re_BT[source_retrieval_index, bits:] = 1

    qu_BI = torch.stack(qu_BI)
    qu_BT = torch.stack(qu_BT)
    qu_L = torch.stack(qu_L)

    # if mode == 'source' or mode == 'target':
    if mode == 'source':

        re_BI = re_BI[:, :bits]
        re_BT = re_BT[:, :bits]
        # re_BI[:, bits:] = 1
        # re_BT[:, bits:] = 1

        qu_BI = qu_BI[:, :bits]
        qu_BT = qu_BT[:, :bits]

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compress_cuda_ft(train_loader, test_loader, model_I, model_T):

    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    with torch.no_grad():
        for _, (data_I, data_T, data_L, _) in enumerate(train_loader):
            var_data_I = Variable(data_I.cuda())
            _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            # re_BI.extend(code_I.cpu().data.numpy())
            re_BI.extend(code_I)

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            # re_BT.extend(code_T.cpu().data.numpy())
            re_BT.extend(code_T)
            # re_L.extend(data_L.cpu().data.numpy())
            re_L.extend(data_L)

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])

    with torch.no_grad():
        for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
            var_data_I = Variable(data_I.cuda())
            _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            # qu_BI.extend(code_I.cpu().data.numpy())
            qu_BI.extend(code_I)

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            # qu_BT.extend(code_T.cpu().data.numpy())
            qu_BT.extend(code_T)
            # qu_L.extend(data_L.cpu().data.numpy())
            qu_L.extend(data_L)

    re_BI = torch.stack(re_BI)
    re_BT = torch.stack(re_BT)
    re_L = torch.stack(re_L)


    qu_BI = torch.stack(qu_BI)
    qu_BT = torch.stack(qu_BT)
    qu_L = torch.stack(qu_L)

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def compress_cuda_ft_split(data_loader, model_I, model_T):

    data_BI = list([])
    data_BT = list([])
    data_Label = list([])
    with torch.no_grad():
        for _, (data_I, data_T, data_L, _) in enumerate(data_loader):
            var_data_I = Variable(data_I.cuda())
            _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            # re_BI.extend(code_I.cpu().data.numpy())
            data_BI.extend(code_I)

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            # re_BT.extend(code_T.cpu().data.numpy())
            data_BT.extend(code_T)
            # re_L.extend(data_L.cpu().data.numpy())

            data_Label.extend(data_L)



    data_BI = torch.stack(data_BI)
    data_BT = torch.stack(data_BT)
    data_Label = torch.stack(data_Label)


    return data_BI, data_BT, data_Label

def get_model_feature(train_I, train_T, model_I, model_T):


    train_I = torch.as_tensor(train_I).cuda()
    train_T = torch.as_tensor(train_T).cuda()
    with torch.no_grad():
        _, tr_BI = model_I(train_I)
        _, tr_BT = model_T(train_T)

    return tr_BI, tr_BT


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk=None):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    if topk is None:
        topk = re_L.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

# origin
def calculate_top_map_cuda(qu_B, re_B, qu_L, re_L, topk=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}

    num_query = qu_L.shape[0]

    map = 0
    if topk is None:
        topk = re_L.shape[0]
    for iter in range(num_query):
        q_L = qu_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(re_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hamming_dist(qu_B[iter, :], re_B)
        _, ind = torch.sort(hamm, stable=True)  # 默认稳定排序
        ind.squeeze_()
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = torch.sum(tgnd)
        if tsum == 0:
            continue

        count = torch.arange(1, int(tsum) + 1).type(torch.float32)
        tindex = torch.nonzero(tgnd).squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map

# update
def calculate_top_map_cuda_u(qu_B, re_B ,qu_L, re_L, re_B2=None,  bits=None, topk=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    
    # print(qu_B.shape[1])
    # print(re_B.shape[1])
    # print(re_B2.shape[1])

    num_query = qu_L.shape[0]
    # print(re_B.shape)
    map = 0
    if topk is None:
        topk = re_L.shape[0]
    for iter in range(num_query):
        q_L = qu_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(re_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)

        if re_B2 is not None:
            # 归一化
            hamm1 = calc_hamming_dist(qu_B[iter, :bits], re_B) / bits
            hamm2 = calc_hamming_dist(qu_B[iter, :], re_B2) / qu_B.shape[1]
            # hamm2 = calc_hamming_dist(qu_B[iter, :bits], re_B2[:, :bits]) / bits
            hamm = torch.cat([hamm1, hamm2], dim=1)

        else:
            hamm = calc_hamming_dist(qu_B[iter, :], re_B)

        _, ind = torch.sort(hamm, stable=True)  # 默认稳定排序
        ind.squeeze_()
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = torch.sum(tgnd)
        if tsum == 0:
            continue

        count = torch.arange(1, int(tsum) + 1).type(torch.float32)
        tindex = torch.nonzero(tgnd).squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def get_logger(filename='logs'):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    log_name = '{}.txt'.format(filename)
    log_dir = './logs'

    # # 判断文件是否存在,存在移除
    # if osp.isfile(osp.join(log_dir, log_name)):
    #     os.remove(osp.join(log_dir, log_name))

    # 创建一个handler,用于写入日志文件
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    # 定义输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    # 创建一个handler,用于输出到控制台
    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    # 定义输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger, txt_log, stream_log
