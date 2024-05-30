import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    '''
        GCN layer.
    '''
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        output = torch.mm(input, self.weight)
        output = torch.mm(adj, output)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HGCN(nn.Module):

    def __init__(self, gcn_input=512, args=None):
        super(HGCN, self).__init__()

        self.gcn_input = gcn_input
        self.args = args

        self.latent_dim = int(args.bits / 4)
        self.hash_dim = int(args.bits / 4)

        self.act = nn.Tanh()

        ## project the common space for Img and Txt
        # common space is 512-D.
        self.img_net = nn.Sequential(
            nn.Linear(args.img_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Linear(1024, self.gcn_input),
            nn.BatchNorm1d(self.gcn_input),
            nn.Tanh()
        )
        # print(args.txt_dim)
        # self.Embedding = init_embedding_layer(args.txt_dim, self.args.dataset)
        self.txt_net = nn.Sequential(
            nn.Linear(args.txt_dim, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, self.gcn_input),
            nn.BatchNorm1d(self.gcn_input),
            nn.Tanh()
        )

        ## HGCN
        # img-gcn
        self.gc1_img = GraphConvolution(self.gcn_input, self.latent_dim, args.dropout, act=lambda x: x)

        # txt-gcn
        self.gc1_txt = GraphConvolution(self.gcn_input, self.latent_dim, args.dropout, act=lambda x: x)

        # cross-modal gcn
        self.gc1_cross = GraphConvolution(self.gcn_input, self.latent_dim, args.dropout, act=lambda x: x)

        ## Hash
        self.hash_txt = nn.Sequential(
            nn.Linear(self.latent_dim, self.hash_dim),
            self.act
        )
        self.hash_img = nn.Sequential(
            nn.Linear(self.latent_dim, self.hash_dim),
            self.act
        )
        self.hash_txt_sp = nn.Sequential(
            nn.Linear(self.latent_dim, self.hash_dim),
            self.act
        )
        self.hash_img_sp = nn.Sequential(
            nn.Linear(self.latent_dim, self.hash_dim),
            self.act
        )

    def forward(self, img, txt, adj, cross_adj, now_size):
        ## project in common space
        img_common = self.img_net(img)
        # txt_common = self.txt_net(self.Embedding(txt))
        txt_common = self.txt_net(txt)

        # GCN for single modal
        img_gcn_1 = self.gc1_img(img_common, adj)
        txt_gcn_1 = self.gc1_txt(txt_common, adj)

        # GCN for cross modal
        comb_1 = self.gc1_cross(torch.cat((img_common, txt_common), 0), cross_adj)
        img_gcn_1_sp = comb_1[:now_size, :]
        txt_gcn_1_sp = comb_1[now_size:, :]

        img_z = self.hash_img(img_gcn_1)
        txt_z = self.hash_txt(txt_gcn_1)
        img_sp_z = self.hash_img_sp(img_gcn_1_sp)
        txt_sp_z = self.hash_txt_sp(txt_gcn_1_sp)

        hash = torch.cat((img_z, img_sp_z, txt_z, txt_sp_z), 1)

        return hash

class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(args.bits, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.discriminator(x)

class ImgNet(nn.Module):

    def __init__(self, code_len):
        super(ImgNet, self).__init__()

        self.hashFunc = nn.Sequential(
            # nn.Linear(args.img_dim, 2048),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, code_len),
            nn.BatchNorm1d(code_len),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.hashFunc(x)
        return x, output

class TxtNet(nn.Module):

    def __init__(self, code_len, args=None):
        super(TxtNet, self).__init__()

        # print(args.txt_dim)
        # self.Embedding = init_embedding_layer(args.txt_dim, args.dataset)

        self.hashFunc = nn.Sequential(
            # nn.Linear(args.txt_dim, 512),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, code_len),
            nn.BatchNorm1d(code_len),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.Embedding(x)
        # print(x)

        output = self.hashFunc(x)
        return x, output

class HashFusion(nn.Module):
    def __init__(self, args):
        super(HashFusion, self).__init__()
        self.bits = args.bits
        self.factor_param = nn.Parameter(torch.ones((1, self.bits)) * 0.5)

    def forward(self, imgHash: torch.Tensor, txtHash: torch.Tensor):
        return imgHash.mul(self.factor_param) + txtHash.mul(1 - self.factor_param)


# def init_embedding_layer(input_dim, dset_name):
#     Embedding = nn.Linear(input_dim, 300)
#     init_weights = None
#     if dset_name == 'NUSWIDE':
#         init_weights = pkl.load(open('models/nuswide_weights.pkl', 'rb'))['weights'].T
#     elif dset_name == 'Flickr':
#         init_weights = pkl.load(open('models/flickr_weights.pkl', 'rb'))['weights'].T
#     elif dset_name == 'COCO':
#         init_weights = pkl.load(open('models/coco_weights.pkl', 'rb'))['weights'].T
#     else:
#         pass
#     Embedding.weight = nn.Parameter(torch.Tensor(init_weights))
#     return Embedding

if __name__ == '__main__':

    pass