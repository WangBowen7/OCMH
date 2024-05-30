import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

class ImgNet(nn.Module):
    # def __init__(self, image_dim, text_dim, hidden_dim, hash_dim, label_dim):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.module_name = 'CPAH'
        self.image_dim = 512
        self.hidden_dim = 512
        self.hash_dim = code_len
        
        # if dataset == 'MIRFlickr':
        #     self.label_dim = 24
            
        # if dataset == 'NUSWIDE':
        #     self.label_dim = 21
            
        # if dataset == 'MSCOCO':
        #     self.label_dim = 80
            
        class Unsqueezer(nn.Module):
            """
            Converts 2d input into 4d input for Conv2d layers
            """
            def __init__(self):
                super(Unsqueezer, self).__init__()

            def forward(self, x):
                return x.unsqueeze(1).unsqueeze(-1)
            
        
        self.image_module = nn.Sequential(
            nn.Linear(self.image_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(True),
        )
        
        
        self.hash_module = nn.Sequential(
                nn.Linear(512, self.hash_dim, bias=True),
                # nn.Tanh()
        )
        
        
        self.mask_module = nn.Sequential(
                Unsqueezer(),
                # nn.Conv1d(1, hidden_dim, kernel_size=(hidden_dim, 1), stride=(1, 1)),
                nn.Conv2d(1, self.hidden_dim, kernel_size=(self.hidden_dim, 1), stride=(1, 1)),
                nn.Sigmoid()
        )
        
        
        self.hashF = nn.Sequential()
                 
        
        # self.feature_dis = nn.Sequential(
        #             nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
        #             nn.ReLU(True),
        #             nn.Linear(hidden_dim // 8, 1, bias=True)
        #         )

        # # C (consistency classification)
        # self.consistency_dis = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
        #     nn.ReLU(True),
        #     nn.Linear(hidden_dim // 8, 3, bias=True)
        # )    
        
        # self.classifier =  nn.Sequential(
        #         nn.Linear(self.hidden_dim, self.label_dim, bias=True),
        #         nn.Sigmoid()
        # )
        
        self.alpha = 1.0
        
    def forward(self, x):
        h_i = self.hashF(x)
        code = torch.tanh(self.alpha * h_i)
        # return f_i, self.hash_module(f_i.detach())
        return h_i, code  
          
    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)    
    # def forward(self, x):
    #     f_i = self.image_module(x)
    #     h_i = self.hash_module(f_i.detach())
    #     code = torch.tanh(self.alpha * h_i)
    #     # return f_i, self.hash_module(f_i.detach())
    #     return f_i, code
        
    # def forward(self, r_img):
    #     f_r_img = self.image_module(r_img)  # image feature
    #     # f_r_txt = self.text_module(r_txt)  # text feature

    #     # MASKING
    #     mc_img = self.get_mask(f_r_img, 'img')  # modality common mask for img
    #     # mc_txt = self.get_mask(f_r_txt, 'txt')  # modality common mask for txt
    #     mp_img = 1 - mc_img  # modality private mask for img
    #     # mp_txt = 1 - mc_txt  # modality private mask for txt

    #     f_rc_img = f_r_img * mc_img  # modality common feature for img
    #     # f_rc_txt = f_r_txt * mc_txt  # modality common feature for txt
    #     f_rp_img = f_r_img * mp_img  # modality private feature for img
    #     # f_rp_txt = f_r_txt * mp_txt  # modality private feature for txt

    #     # HASHING
    #     h_img = self.get_hash(f_rc_img, 'img')  # img hash
    #     # h_txt = self.get_hash(f_rc_txt, 'txt')  # txt hash

    #     # return h_img, h_txt, f_rc_img, f_rc_txt, f_rp_img, f_rp_txt
    #     return h_img, f_rc_img, f_rp_img

    # def get_mask(self, x):
    #     return self.mask_module(x).squeeze()

    # def get_hash(self, x):
    #     return self.hash_module(x).squeeze()

    # def generate_img_code(self, i):
    #     f_i = self.image_module(i)
    #     return f_i, self.hash_module(f_i.detach())

    # def dis_D(self, f):
    #     return self.feature_dis(f).squeeze()

    # def dis_C(self, f):
    #     return self.consistency_dis(f).squeeze()

    # def dis_classify(self, f):
    #     return self.classifier(f).squeeze()
    

class TxtNet(nn.Module):
    def __init__(self, code_len):
        super(TxtNet, self).__init__()
        self.module_name = 'CPAH'
        self.image_dim = 512
        self.text_dim = 512
        self.hidden_dim = 512
        self.hash_dim = code_len
        # if dataset == 'MIRFlickr':
        #     self.label_dim = 24
            
        # if dataset == 'NUSWIDE':
        #     self.label_dim = 21
            
        # if dataset == 'MSCOCO':
        #     self.label_dim = 80
        
        class Unsqueezer(nn.Module):
            """
            Converts 2d input into 4d input for Conv2d layers
            """
            def __init__(self):
                super(Unsqueezer, self).__init__()

            def forward(self, x):
                return x.unsqueeze(1).unsqueeze(-1)
            
        
        self.text_module = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(True),
        )
        
        self.hash_module = nn.Sequential(
                nn.Linear(512, self.hash_dim, bias=True),
                # nn.Tanh()
        )
        
        self.mask_module = nn.Sequential(
                Unsqueezer(),
                # nn.Conv1d(1, hidden_dim, kernel_size=(hidden_dim, 1), stride=(1, 1)),
                nn.Conv2d(1, self.hidden_dim, kernel_size=(self.hidden_dim, 1), stride=(1, 1)),
                nn.Sigmoid()
        )
        
        self.hashF = nn.Sequential()
        
        # self.feature_dis = nn.Sequential(
        #             nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
        #             nn.ReLU(True),
        #             nn.Linear(hidden_dim // 8, 1, bias=True)
        #         )

        # # C (consistency classification)
        # self.consistency_dis = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
        #     nn.ReLU(True),
        #     nn.Linear(hidden_dim // 8, 3, bias=True)
        # )      
        
        # self.classifier =  nn.Sequential(
        #         nn.Linear(self.hidden_dim, label_dim, bias=True),
        #         nn.Sigmoid()
        # )

        self.alpha = 1.0

    def forward(self, x):
        h_t = self.hashF(x)
        code = torch.tanh(self.alpha * h_t)
        # return f_i, self.hash_module(f_i.detach())
        return h_t, code  
    
    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
    
    # def forward(self, x):
    #     f_t = self.text_module(x)
    #     h_t = self.hash_module(f_t.detach())
    #     code = torch.tanh(self.alpha * h_t)
    #     # return f_t, self.hash_module(f_t.detach())
    #     return f_t, code
    
    # def forward(self, r_txt):
    #     # f_r_img = self.image_module(r_img)  # image feature
    #     f_r_txt = self.text_module(r_txt)  # text feature

    #     # MASKING
    #     # mc_img = self.get_mask(f_r_img, 'img')  # modality common mask for img
    #     mc_txt = self.get_mask(f_r_txt)  # modality common mask for txt
    #     # mp_img = 1 - mc_img  # modality private mask for img
    #     mp_txt = 1 - mc_txt  # modality private mask for txt

    #     # f_rc_img = f_r_img * mc_img  # modality common feature for img
    #     f_rc_txt = f_r_txt * mc_txt  # modality common feature for txt
    #     # f_rp_img = f_r_img * mp_img  # modality private feature for img
    #     f_rp_txt = f_r_txt * mp_txt  # modality private feature for txt

    #     # HASHING
    #     # h_img = self.get_hash(f_rc_img, 'img')  # img hash
    #     h_txt = self.get_hash(f_rc_txt)  # txt hash

    #     # return h_img, h_txt, f_rc_img, f_rc_txt, f_rp_img, f_rp_txt
    #     return h_txt, f_rc_txt, f_rp_txt

    # def get_mask(self, x):
    #     return self.mask_module(x).squeeze()

    # def get_hash(self, x):
    #     return self.hash_module(x).squeeze()

    # def generate_txt_code(self, t):
    #     f_t = self.text_module(t)
    #     return f_t, self.hash_module(f_t.detach())

    # def dis_D(self, f):
    #     return self.feature_dis(f).squeeze()

    # def dis_C(self, f):
    #     return self.consistency_dis(f).squeeze()

    # def dis_classify(self, f):
    #     return self.classifier(f).squeeze()    