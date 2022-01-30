import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import Parameter
from attention.CrossModalAttention import CrossModalfAttention 
from attention.SelfAttention import ScaledDotProductAttention
from model.vit import image_net
from pytorch_transformers import BertModel


class ImgTextNet(nn.Module):
    def __init__(self, code_len, txt_feat_len, image_size=1024):
        super(ImgNet, self).__init__()
        self.image_net = image_net()  # 1024
        self.fc_encode = nn.Linear(image_size, code_len)
        self.decode = nn.Linear(code_len, txt_feat_len)
        self.alpha = 1.0
        self.alexnet = torchvision.models.alexnet(pretrained=True)

        self.fc1 = nn.Linear(txt_feat_len, image_size)
        self.fc2 = nn.Linear(image_size, code_len)
        self.textdecode = nn.Linear(code_len, image_size)
        self.fc3 = nn.Linear(txt_feat_len, 3)
        self.fc4 = nn.Linear(image_size, 512)
        self.fc5 = nn.Linear(512, image_size)
        self.fc6 = nn.Linear(512 * 7 * 7, image_size)
        self.fc7 = nn.Linear(image_size, 512)
        self.fc8 = nn.Linear(2 * image_size, image_size)
        self.fc9 = nn.Linear(2 * code_len, code_len)
        self.fc = nn.Linear(2 * txt_feat_len, txt_feat_len)
        self.attention = SequentialPolarizedSelfAttention(channel=512)
        self.sattention = ScaledDotProductAttention(d_model=1024, d_k=1024, d_v=1024, h=8)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # new add
        # self.attention = SpatialGroupEnhance(groups=8)


    def forward(self, x, y):
        x = self.image_net(x)
        feat = x.view(x.size(0), -1)
        imgfeat = feat
        # hid = self.fc_encode(feat)
        # code = torch.tanh(self.alpha * hid)
        # code_s = torch.sign(code)
        # # decoded = self.decode(code_s)
        # ConvTrans2d = nn.ConvTranspose2d()
        # ConvTrans1d = nn.ConvTranspose1d()
        txtfeat1 = self.fc1(y)
        txtfeat = F.relu(txtfeat1)

        # -------------------------------
        txtfeatt = txtfeat.unsqueeze(1)
        txtfeatt = self.sattention(txtfeatt, txtfeatt, txtfeatt)
        txtfeatt = txtfeatt.squeeze()
        txtfeat = txtfeat + txtfeatt
        # -------------------------------
        # txthid = self.fc2(txtfeat)
        txtfeatb = self.fc4(txtfeat)
        txtfeatb = F.relu(txtfeatb)

        feat = self.fc4(feat)
        feat = F.relu(feat)
        feat = torch.cat((txtfeatb, feat), 1)
        # -------------------------------
        feat = feat.unsqueeze(1)
        feat = self.sattention(feat, feat, feat)
        feat = feat.squeeze()
        # -------------------------------
        feat = self.fc7(feat)
        feat = feat.unsqueeze(-1).unsqueeze(-1)
        feat = self.avg(feat)
        feat = feat.repeat(1, 1, 7, 7)
        # upsample = nn.UpsamplingBilinear2d(scale_factor=7)
        #
        # feat = upsample(feat)
        size = feat.size(1)
        size1 = feat.size(-1)
        # feat = feat.flatten(2).transpose(1, 2) # 64*49*512
        # feat = feat.transpose(1, 2) # 64*512*49
        # feat = feat.view(feat.size(0), size, size1, -1)  # yong yu kuo zhan weidu
        feat = self.attention(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc6(feat)
        fusionfeat = feat
        # txtfeatb = txtfeatb.unsqueeze(-1).unsqueeze(-1)
        # upsample = nn.UpsamplingBilinear2d(scale_factor=7)
        # txtfeatb = upsample(txtfeatb)
        # txtfeatb = self.attention(txtfeatb)
        # txtfeatb = txtfeatb.view(txtfeatb.size(0), -1)
        # txtfeatb = self.fc6(txtfeatb)
        # cat image
        feat = torch.cat((feat, imgfeat), 1)
        feat = self.fc8(feat)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)
        code_s = torch.sign(code)
        imagedecoded = self.decode(code_s)  # 128 to 1386

        # cat text
        textfeat = torch.cat((feat, txtfeat), 1)
        textfeat = self.fc8(textfeat)
        txthid = self.fc2(textfeat)
        txtcode = torch.tanh(self.alpha * txthid)
        txtcode_s = torch.sign(txtcode)
        textdecoded = self.textdecode(txtcode_s)  # 128 to 1024
        # totalcode_s = (code_s + txtcode_s)
        totalcode_s = torch.cat((code_s, txtcode_s), 1)
        totalcode_s = self.fc9(totalcode_s)
        decoded = self.decode(totalcode_s)
        txtdecoded = self.textdecode(totalcode_s)

        decoded = torch.cat((imagedecoded, decoded), 1) # 2772
        txtdecoded = torch.cat((txtdecoded, textdecoded), 1) # 2048
        decoded = self.fc(decoded)
        txtdecoded = self.fc8(txtdecoded)

        return (x, feat), hid, code, decoded, textfeat, txthid, txtcode, txtdecoded, fusionfeat

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


    def forward(self, x):
        feat1 = self.fc1(x)
        feat = F.relu(feat1)
        hid = self.fc2(feat)

        code = torch.tanh(self.alpha * hid)
        code_s = torch.sign(code)
        decoded = self.decode(code_s)
        return feat, hid, code, decoded

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

