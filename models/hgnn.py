import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
affine_par = True
import sys
from .ConvGRU import ConvGRUCell
import time
import math
import cv2
import numpy as np
import os
import torch.utils.model_zoo as model_zoo
from .util import remove_layer
from .util import initialize_weights

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        out_static = self.forward_static_gcn(x)
        x = x + out_static
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x

class Postion_Att(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Postion_Att,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.agnostic_conv = nn.Conv2d(in_channels = in_dim, out_channels = 1, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)
        energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)

        out_agnostic = out.clone()
        out_final = out.clone()
        out_agnostic = self.agnostic_conv(out_agnostic)
        out_agnostic = self.sigmoid(out_agnostic)
        out_B = 1 - out_agnostic
        out_A = out_final * out_B

        out = self.gamma*out + x
        return out, out_A

class CoattentionModel(nn.Module):
    def  __init__(self, features, num_classes, all_channel=512, att_dir='./runs/', training_epoch=10, **kwargs):	
        super(CoattentionModel, self).__init__()
        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)    
        )
        self.extra_cls = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(512, 20, 1)
        )

        self.extra_classifier = nn.Conv2d(all_channel, num_classes, (1,1), bias=False)
        self.extra_gcn = DynamicGraphConvolution(all_channel, all_channel, num_classes)
        self.extra_branch = nn.Conv2d(all_channel, all_channel, (1,1))
        self.extra_gap = nn.AdaptiveAvgPool2d(1)

        self.extra_linear_e = nn.Linear(all_channel, all_channel, bias = False)
        self.channel = all_channel
        self.extra_gate = nn.Conv2d(all_channel, 1, kernel_size = 1, bias = False)
        self.extra_gate_s = nn.Sigmoid()
        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.extra_conv_fusion = nn.Conv2d(all_channel*3, all_channel, kernel_size=3, padding=1, bias= True)
        self.extra_relu_fusion = nn.ReLU(inplace=True)
        self.softmax = nn.Sigmoid()
        self.propagate_layers = 5  
        self.extra_edge = nn.Sequential(
            nn.Conv2d(all_channel*2, num_classes, 1)
        )

        self.att_p = Postion_Att(all_channel*2)

        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()     
    
    def forward_feature(self, x):
        '''
        backbone network
        '''
        x = self.features(x)
        x = self.extra_convs(x)
        return x
    
    def forward_sam(self, x):

        mask = self.extra_classifier(x) 
        mask = mask.view(mask.size(0), mask.size(1), -1) 
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.extra_branch(x)
        x_branch = x.clone()
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x, x_branch

    def forward_dgcn(self, x):
        x = self.extra_gcn(x)
        return x

    def generate_satt(self, exemplar): 
        s_sam, s_branch = self.forward_sam(exemplar)
        batch, channel, height, width = s_branch.shape
        s_dgcn = self.forward_dgcn(s_sam)
        s_dgcn = s_dgcn.transpose(1,2)

        s_final = torch.zeros(batch,20,channel,height,width).cuda()
        for i in range(20):
            s_i = s_dgcn[0][i]
            s_expand = s_i.unsqueeze(1).unsqueeze(2)
            s = s_branch[0] * s_expand
            s = self.softmax(s)
            s_final[0][i] = s

        s_att, _ = torch.max(s_final, dim=1, keepdim=True)
        s_att = s_att.squeeze(-4)
        s_fi = s_att + exemplar
        return s_fi
		
    def forward(self, input1, input2, input3, epoch=1, label=None, index=None): 
        x1 = self.forward_feature(input1)
        x1c = self.extra_cls(x1)
        self.map = x1c.clone()
        x11 = x1c.clone()
        x1ss = self.extra_gap(x11)
        x1ss = x1ss.view(-1, 20)

        x2 = self.forward_feature(input2)
        x2c = self.extra_cls(x2)
        x22 = x2c.clone()
        x2ss = self.extra_gap(x22)
        x2ss = x2ss.view(-1, 20)

        x3 = self.forward_feature(input3)
        x3c = self.extra_cls(x3)
        x33 = x3c.clone()
        x3ss = self.extra_gap(x33)
        x3ss = x3ss.view(-1, 20)
        
        exemplar = x1.clone()
        query = x2.clone()
        query1 = x3.clone()

        for ii in range(1):
            for passing_round in range(self.propagate_layers):
                s1_att = self.generate_satt(exemplar)
                s2_att = self.generate_satt(query)
                s3_att = self.generate_satt(query1)

                exemplar_1, _ = self.generate_attention(exemplar, query)
                exemplar_2, _ = self.generate_attention(exemplar, query1)

                query_1, _ = self.generate_attention(query, exemplar)
                query_2, _ = self.generate_attention(query, query1)

                query1_1, _ = self.generate_attention(query1, exemplar)
                query1_2, _ = self.generate_attention(query1, query)

                attention1 = self.message_fun(torch.cat([exemplar_1, exemplar_2, s1_att],1))
                attention2 = self.message_fun(torch.cat([query_1, query_2, s2_att],1))
                attention3 = self.message_fun(torch.cat([query1_1, query1_2, s3_att],1))
             
                h_v1 = self.extra_ConvGRU(attention1, exemplar)
                h_v2 = self.extra_ConvGRU(attention2, query)
                h_v3 = self.extra_ConvGRU(attention3, query1)
                
                exemplar = h_v1.clone()
                query = h_v2.clone()
                query1 = h_v3.clone()

                if passing_round == self.propagate_layers -1:
                    edge1 = torch.cat([exemplar,query],1)
                    e1_p, e1_A = self.att_p(edge1)
                    e1_p = self.extra_edge(e1_p)
                    e1_A = self.extra_edge(e1_A)
                    e1_p = self.extra_gap(e1_p)
                    e1_A = self.extra_gap(e1_A)
                    e1_p = e1_p.view(-1,20)
                    e1_A = e1_A.view(-1,20)

                    edge2 = torch.cat([exemplar,query1],1)
                    e2_p, e2_A = self.att_p(edge2)
                    e2_p = self.extra_edge(e2_p)
                    e2_A = self.extra_edge(e2_A)
                    e2_p = self.extra_gap(e2_p)
                    e2_A = self.extra_gap(e2_A)
                    e2_p = e2_p.view(-1,20)
                    e2_A = e2_A.view(-1,20)

                    edge3 = torch.cat([query,query1],1)
                    e3_p, e3_A = self.att_p(edge3)
                    e3_p = self.extra_edge(e3_p)
                    e3_A = self.extra_edge(e3_A)
                    e3_p = self.extra_gap(e3_p)
                    e3_A = self.extra_gap(e3_A)
                    e3_p = e3_p.view(-1,20)
                    e3_A = e3_A.view(-1,20)

                    exemplar = self.extra_cls(exemplar)
                    self.map1 = exemplar.clone()
                    x1s = self.extra_gap(exemplar)
                    x1s = x1s.view(-1, 20)

                    query = self.extra_cls(query)
                    x2s = self.extra_gap(query)
                    x2s = x2s.view(-1, 20)

                    query1 = self.extra_cls(query1)
                    x3s = self.extra_gap(query1)
                    x3s = x3s.view(-1, 20)

                    pre_probs = x1s.clone() 
                    probs = torch.sigmoid(pre_probs)
                    if index != None and epoch > 0:
                        atts = (self.map1 + self.map) / 2
                        atts[atts < 0] = 0
                        ind = torch.nonzero(label)
            
                        for i in range(ind.shape[0]):
                            batch_index, la = ind[i]
                            accu_map_name = '{}/{}_{}.png'.format(self.att_dir, batch_index+index, la)
                            att = atts[0, la].cpu().data.numpy()
                            att = np.rint(att / (att.max()  + 1e-8) * 255)

                            if epoch == self.training_epoch - 1 and not os.path.exists(accu_map_name):
                                cv2.imwrite(accu_map_name, att)
                                continue
                
                            if probs[0, la] < 0.1:  
                                continue
                            
                            try:
                                if not os.path.exists(accu_map_name):
                                    cv2.imwrite(accu_map_name, att)
                                else:
                                    accu_at = cv2.imread(accu_map_name, 0)
                                    accu_at_max = np.maximum(accu_at, att)
                                    cv2.imwrite(accu_map_name,  accu_at_max)
                            except Exception as e:
                                print(e)
       
        return x1ss,x1s, x2ss,x2s, x3ss,x3s, e1_p, e1_A, e2_p, e2_A, e3_p, e3_A

    def message_fun(self,input):
        input1 = self.extra_conv_fusion(input)
        input1 = self.extra_relu_fusion(input1)
        return input1

    def generate_attention(self, exemplar, query):
        fea_size = query.size()[2:]	 
        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0]*fea_size[1])
        query_flat = query.view(-1, self.channel, fea_size[0]*fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()
        exemplar_corr = self.extra_linear_e(exemplar_t)
        A = torch.bmm(exemplar_corr, query_flat)
        
        B = F.softmax(torch.transpose(A,1,2),dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        
        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  
        input1_att1 = input1_att.clone()
        input1_mask = self.extra_gate(input1_att)
        input1_mask = self.extra_gate_s(input1_mask)
        input1_att = input1_att * input1_mask

        return input1_att, input1_att1

        
    def get_heatmaps(self):
        return (self.map1 + self.map) / 2
    
    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model

    
def load_pretrained_model(model, path=None):

    state_dict = model_zoo.load_url(model_urls['vgg16'], progress=True)

    state_dict = remove_layer(state_dict, 'classifier.')
    state_dict = adjust_pretrained_model(state_dict, model)

    model.load_state_dict(state_dict, strict=False)
    return model


def make_layers(cfg, batch_norm=False,**kwargs):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)            
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'D2':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def HGNN_Net(pretrained=True, **kwargs):
    model = CoattentionModel(make_layers(cfg['D1'],**kwargs),**kwargs)  
    if pretrained:
         model = load_pretrained_model(model,path=None)
    return model
