import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel, AutoProcessor

class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        # self.hash_layer = nn.Linear(4096, code_len)
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.hash_layer = nn.Linear(512, code_len)
        self.alpha = 1.0

    def forward(self, x):

        # with torch.no_grad():
        #     x = self.alexnet.features(x)
        #     x = x.view(x.size(0), -1)
        #     feat = self.alexnet.classifier(x)

        # with torch.no_grad():
        #     feat = self.model.get_image_features(pixel_values=x)
        feat = x

        hid = self.hash_layer(feat)
        code = torch.tanh(self.alpha * hid)

        # return feat, hid, code
        return hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len=512):
        super(TxtNet, self).__init__()

        self.net = nn.Sequential(nn.Linear(txt_feat_len, 4096),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(4096, code_len),
                                 )

        self.alpha = 1.0

    def forward(self, x):
        hid = self.net(x)
        code = torch.tanh(self.alpha * hid)
        # return code
        return hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


if __name__ == '__main__':
    pass