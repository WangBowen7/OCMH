import torch
import torch.nn.functional as F
from train.base_trainer_lr import Base_Trainer
from utils import get_logger
from transformers import set_seed
import numpy as np

class OURS_Trainer(Base_Trainer):
    def __init__(self, args, logger) -> None:
        super().__init__(args, logger)
        
        self.imgNet = self.Expand_CodeNet_I.cuda()
        self.txtNet = self.Expand_CodeNet_T.cuda()
        
        self.opt_I_origin = torch.optim.SGD(self.imgNet.parameters(), lr=args.LR_ORIGIN * args.LR_IMG, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        self.opt_T_origin = torch.optim.SGD(self.txtNet.parameters(), lr=args.LR_ORIGIN * args.LR_TXT, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        
        self.opt_I_expand = torch.optim.SGD(self.imgNet.parameters(), lr=args.LR_EXPAND * args.LR_IMG, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        self.opt_T_expand = torch.optim.SGD(self.txtNet.parameters(), lr=args.LR_EXPAND * args.LR_TXT, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        
        
        # TODO        
        # 生成用于自蒸馏的初始continues hash code
        self.CodeNet_I_bk.cuda().eval()
        self.CodeNet_T_bk.cuda().eval()
        

        with torch.no_grad():
            _, self.continues_hash_I = self.CodeNet_I_bk(torch.as_tensor(self.I_tr_target).cuda())
            _, self.continues_hash_T = self.CodeNet_T_bk(torch.as_tensor(self.T_tr_target).cuda())
            
    def get_ID(self, ):
        self.I_tr_target_pt = F.normalize(torch.as_tensor(self.I_tr_target).cuda())
        self.T_tr_target_pt = F.normalize(torch.as_tensor(self.T_tr_target).cuda())
        
        S_IT = torch.matmul(self.I_tr_target_pt, self.T_tr_target_pt.T)
        self.mask = (~torch.eye(self.I_tr_target_pt.shape[0], self.I_tr_target_pt.shape[0], dtype=bool).cuda()).float()
        S_IT = S_IT * self.mask 
        values, indices = S_IT.topk(self.args.batch_size-1)
        
        images = []
        texts = []
        loss_I_list = []
        loss_T_list = []
        for idx, indice in enumerate(indices):
            cur_image = self.continues_hash_I[idx].unsqueeze(0)
            ner_images = self.continues_hash_I[indice]
            batch_images = torch.cat([cur_image, ner_images], dim=0)
        
            cur_text = self.continues_hash_T[idx].unsqueeze(0)
            ner_texts = self.continues_hash_T[indice]
            batch_texts = torch.cat([cur_text, ner_texts], dim=0)
            
            images.append(batch_images)
            texts.append(batch_texts)
            
            loss, loss_I, loss_T = self.contrastiveLoss(batch_images, batch_texts, temperature=0.5)  # 温度系数越大，logits越平缓
            
            loss_I_list.append(loss_I[0].cpu().numpy())
            loss_T_list.append(loss_T[0].cpu().numpy())
        
        

        loss_I_list = np.array(loss_I_list)
        loss_T_list = np.array(loss_T_list)
        
        loss_I_mean = np.mean(loss_I_list) 
        loss_T_mean = np.mean(loss_T_list)     
        
        
        self.index = np.arange(self.I_tr_target.shape[0])

        self.ID_index = list(set(self.index[loss_I_list < loss_I_mean]) & set(self.index[loss_T_list < loss_T_mean]))
        self.ID_loss_I = loss_I_list[self.ID_index]
        self.ID_loss_T = loss_T_list[self.ID_index]
        
        self.OOD_index = list(set(self.index[loss_I_list > loss_I_mean]) & set(self.index[loss_T_list > loss_T_mean]))
        self.OOD_loss_I = loss_I_list[self.OOD_index]
        self.OOD_loss_T = loss_T_list[self.OOD_index]
        
        return self.ID_index, self.ID_loss_I, self.ID_loss_T, self.OOD_index, self.OOD_loss_I, self.OOD_loss_T
    

    def train(self):
        self.imgNet.train()
        self.txtNet.train()
        self.ID_index, self.ID_loss_I, self.ID_loss_T, self.OOD_index, self.OOD_loss_I, self.OOD_loss_T = self.get_ID()
        self.ID_index_dict = dict()
        for idx, ID_idx in enumerate(self.ID_index):
            self.ID_index_dict[ID_idx] = self.ID_loss_I[idx] + self.ID_loss_T[idx]
            
        self.OOD_index_dict = dict()
        for idx, OOD_idx in enumerate(self.OOD_index):
            self.OOD_index_dict[OOD_idx] = self.OOD_loss_I[idx] + self.OOD_loss_T[idx]

        for epoch in range(self.args.EPOCHS):
            
            if self.args.set_alpha == 1:  
                self.imgNet.set_alpha(epoch)
                self.txtNet.set_alpha(epoch)
            
            
            self.I_threod = np.mean(self.ID_loss_I)
            self.T_threod = np.mean(self.ID_loss_T)
            self.loss_I_epoch_ID = []
            self.loss_T_epoch_ID = []
            self.ID_index_epoch = []
            
            self.loss_I_epoch_OOD = []
            self.loss_T_epoch_OOD = []
            self.OOD_index_epoch = []
            
            self.ID_index_dict_list = list(self.ID_index_dict.keys())
            self.OOD_index_dict_list = list(self.OOD_index_dict.keys())
            
            
            for idx, (image_feature, text_feature, label, index) in enumerate(self.train_dataloader):
                image_feature = image_feature.cuda()
                text_feature = text_feature.cuda()
        

                ## 获取每个batch的ID样本
                index_s = [] # 对应每个batch训练数据的索引
                # index_s_origin = [] # 对应训练数据的原始索引
                
                index_s_origin = list(set(self.ID_index_dict.keys()) & set(index.numpy()))
                
                for i in range(len(index.numpy())):
                    if index.numpy()[i] in self.ID_index_dict.keys():
                        index_s.append(i)
                                          
                        
                ## 获取每个batch的OOD样本
                index_dis = [] # 对应每个batch训练数据的索引
                # index_s_origin = [] # 对应训练数据的原始索引
                
                index_dis_origin = list(set(self.OOD_index_dict.keys()) & set(index.numpy()))
                             
                   
                _, code_I = self.imgNet(image_feature)
                _, code_T = self.txtNet(text_feature)
                
                image_feature_aug = image_feature + torch.randn_like(image_feature) * 0.1
                text_feature_aug = text_feature + torch.randn_like(text_feature) * 0.1
                
                _, code_I_aug = self.imgNet(image_feature_aug)
                _, code_T_aug = self.txtNet(text_feature_aug)
                
                # TODO添加（得到相似样本的数据）
                code_I_s = code_I[index_s]
                code_T_s = code_T[index_s]
                
                code_I_aug_s = code_I_aug[index_s]
                code_T_aug_s = code_T_aug[index_s]
                
                loss_cb, _, _ = self.contrastiveLoss(code_I_s[:, :self.args.bits], code_T_s[:,:self.args.bits], temperature=0.5) # 原来是0.1
                
                
                code_I_dis = code_I[index_dis]
                code_T_dis = code_T[index_dis]
                
                
                loss_cl_inter, _, _  = self.contrastiveLoss(code_I, code_T, temperature=0.5)
                loss_cl_I, _, _  = self.contrastiveLoss(code_I, code_I_aug, temperature=0.5)
                loss_cl_T, _, _  = self.contrastiveLoss(code_T, code_T_aug, temperature=0.5)
                loss_cl_intra = loss_cl_I + loss_cl_T
                

                loss = self.args.LAMBDA0*loss_cl_inter + self.args.LAMBDA1*loss_cl_intra + self.args.LAMBDA2*loss_cb 
         
                self.opt_T_expand.zero_grad()
                self.opt_I_expand.zero_grad()

                self.opt_T_origin.zero_grad()
                self.opt_I_origin.zero_grad()

                loss.backward()

                self.imgNet.save_current()
                self.txtNet.save_current()

                # TODO 更新expand部分参数
                self.opt_T_expand.step()
                self.opt_I_expand.step()

                self.imgNet.update_current_expand()
                self.txtNet.update_current_expand()

                self.imgNet.save_current()
                self.txtNet.save_current()
                # TODO 更新origin部分参数
                self.opt_T_origin.step()
                self.opt_I_origin.step()

                self.imgNet.update_current_origin()
                self.txtNet.update_current_origin()

                self.imgNet.update_parameters()
                self.txtNet.update_parameters()
                
                
                # 计算新模型的loss值，更新ID bank
                with torch.no_grad():
                    _, code_I = self.imgNet(image_feature)
                    _, code_T = self.txtNet(text_feature)
                    
                # print('bs {}'.format(image_feature.shape))
                # 使用origin计算loss
                loss, loss_I, loss_T = self.contrastiveLoss(code_I[:,:self.args.bits], code_T[:,:self.args.bits], temperature=0.5) # 原来0.5
                  
                idx_batch = np.arange(len(index))
                
                ID_idx_batch = list(set(idx_batch[loss_I.cpu().numpy() < self.I_threod]) & set(idx_batch[loss_T.cpu().numpy() < self.T_threod]))
                self.loss_I_epoch_ID.extend(loss_I.cpu().numpy()[ID_idx_batch])
                self.loss_T_epoch_ID.extend(loss_T.cpu().numpy()[ID_idx_batch])
                self.ID_index_epoch.extend(ID_idx_batch)
                
                OOD_idx_batch = list(set(idx_batch[loss_I.cpu().numpy() > self.I_threod]) & set(idx_batch[loss_T.cpu().numpy() > self.T_threod]))
                self.loss_I_epoch_OOD.extend(loss_I.cpu().numpy()[OOD_idx_batch])
                self.loss_T_epoch_OOD.extend(loss_T.cpu().numpy()[OOD_idx_batch])
                self.OOD_index_epoch.extend(OOD_idx_batch)                     

            ## 更新bank
            # 求ID差集并加入dict中
            extend_ID_idx_list = list(set(self.ID_index_epoch) - set(self.ID_index))
            extend_ID_idx_num = len(extend_ID_idx_list)
            
            for extend_ID_idx in extend_ID_idx_list:
                self.ID_index_dict[extend_ID_idx] = self.loss_I_epoch_ID[self.ID_index_epoch.index(extend_ID_idx)] + self.loss_T_epoch_ID[self.ID_index_epoch.index(extend_ID_idx)]
            # print(len(self.ID_index_dict.keys()))
            # 按值排序 (返回元组列表)
            ID_index_sorted = sorted(self.ID_index_dict.items(), key=lambda x: x[1], reverse=False)[:extend_ID_idx_num]
            
            # 删除指定的不相似的值
            for k, v in ID_index_sorted:
                self.ID_index_dict.pop(k)
                
                
            # 求OOD差集并加入dict中
            extend_OOD_idx_list = list(set(self.OOD_index_epoch) - set(self.OOD_index))
            extend_OOD_idx_num = len(extend_OOD_idx_list)
            
            for extend_OOD_idx in extend_OOD_idx_list:
                self.OOD_index_dict[extend_OOD_idx] = self.loss_I_epoch_OOD[self.OOD_index_epoch.index(extend_OOD_idx)] + self.loss_T_epoch_OOD[self.OOD_index_epoch.index(extend_OOD_idx)]
            # print(len(self.ID_index_dict.keys()))
            # 按值排序 (返回元组列表)
            OOD_index_sorted = sorted(self.OOD_index_dict.items(), key=lambda x: x[1], reverse=False)[:extend_OOD_idx_num]
            
            # 删除指定的不相似的值
            for k, v in OOD_index_sorted:
                self.OOD_index_dict.pop(k)
            
                
            # print('epoch:{}  loss:{}'.format(epoch, loss.item()))         
            # break
            if (epoch+1) % 10 == 0:
                print('------------{}----------'.format(epoch+1))
                self.eval(save_log=self.args.save_log, epoch=epoch+1) 
                # self.eval_baseline(save_log=self.args.save_log, epoch=epoch+1) 
                # self.eval(save_log=False) 

    def contrastiveLoss(self, emb_i, emb_j, temperature=0.5):	
        
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        
        self.batch_size = z_i.shape[0]
        
        self.negatives_mask = (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool).cuda()).float()	
        self.temperature = torch.tensor(temperature).cuda()
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        # loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        # loss = torch.sum(loss_partial) / (2 * self.batch_size)
        loss_I_partial = -torch.log(nominator[:self.batch_size] / torch.sum(denominator[:self.batch_size], dim=1))        # 1*bs
        loss_T_partial = -torch.log(nominator[self.batch_size:] / torch.sum(denominator[self.batch_size:], dim=1))        # 1*bs
        loss_I = torch.sum(loss_I_partial) / (1 * self.batch_size)
        loss_T = torch.sum(loss_T_partial) / (1 * self.batch_size)
        
        loss = loss_I + loss_T
        
        return loss, loss_I_partial, loss_T_partial
    
         
         