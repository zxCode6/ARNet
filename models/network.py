import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math
import sys
import numpy as np
import torch.nn.functional as F
sys.dont_write_bytecode = True

'''

	This Network is designed for Few-Shot Learning Problem.

'''


###############################################################################
# Functions
###############################################################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_DN4Net(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal',
                  use_gpu=True, shot=None, parameter_t=None, parameter_h=None,**kwargs):
    DN4Net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model == 'Conv64F':
        DN4Net = FourLayer_64F(norm_layer=norm_layer, num=shot, parameter_t=parameter_t,parameter_h=parameter_h, **kwargs)
    elif which_model == 'ResNet256F':
        net_opt = {'userelu': False, 'in_planes': 3, 'dropout': 0.5, 'norm_layer': norm_layer}
        DN4Net = ResNetLike(net_opt, num=shot, parameter_t=parameter_t,parameter_h=parameter_h)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)
    init_weights(DN4Net, init_type=init_type)

    if use_gpu:
        DN4Net.cuda()

    if pretrained:
        DN4Net.load_state_dict(model_root)

    return DN4Net


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)





# #############################################################################
# Classes: FourLayer_64F
# #############################################################################

# Model: FourLayer_64F
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21






class FourLayer_64F(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3, num=None, parameter_t=None,parameter_h=None):
        super(FourLayer_64F, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(  # 3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*42*42

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21


            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21

        )

        self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k, num=num,
                                            parameter_t=parameter_t,parameter_h=parameter_h)

    def forward(self, input1, input2):

        # extract features of input1--query image   75
        q = self.features(input1)

        # extract features of input2--support set   5

        S = []

        for i in range(len(input2)):
            support_set_sam = self.features(input2[i])
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            S.append(support_set_sam)

        x = self.imgtoclass(q, S)  # get Batch*num_classes

        return x


# ========================== Define an image-to-class layer ==========================#



def average_pooling_to_vector(tensor):
        output = torch.mean(tensor,dim=1);
        return output


def average_pooling_to_vector1(tensor):
    output = torch.mean(tensor, dim=0);
    return output



def gaussian_kernel(x, y, sigma=1.0):
    exponent = -torch.pow(torch.norm(x - y, 2) / (2.0 * sigma), 2)
    return torch.exp(exponent)

def calculate_mmd_for_prototype(prototype, all_prototypes, sigma=1.0):
    mmd_for_prototype = 0.0

    for i in range(all_prototypes.size(0)):
        mmd_for_prototype += gaussian_kernel(prototype, all_prototypes[i], sigma)

    return mmd_for_prototype / all_prototypes.size(0)


class WeightGenerationNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(WeightGenerationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, query_sam, support_pro):
        support_pro = support_pro.unsqueeze(0).expand(query_sam.size(0), -1)
        concatenated_input = torch.cat((query_sam, support_pro), dim=1)
        x = self.fc1(concatenated_input)
        x = self.relu(x)
        output_vector = self.fc2(x)

       
        softmaxed_vector = F.softmax(output_vector, dim=1)

        return softmaxed_vector


class ImgtoClass_Metric(nn.Module):
    def __init__(self, neighbor_k=3, num=None, parameter_t=None,parameter_h=None):
        super(ImgtoClass_Metric, self).__init__()
        self.neighbor_k = neighbor_k
        self.num = num
        self.t = parameter_t
        self.h=parameter_h

        self.weight_generators = nn.ModuleList(
        [WeightGenerationNetwork(input_size=64, output_size=441) for _ in range(num)])


    def cal_cosinesimilarity(self, input1, input2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        shotList = ['shot1', 'shot2', 'shot3', 'shot4', 'shot5', 'shot6', 'shot7', 'shot8', 'shot9', 'shot10']
        support_pro=['p1', 'p2', 'p3', 'p4', 'p5']
        support_Prototypical = ['P1', 'P2', 'P3', 'P4', 'P5']
        listM = ['matrix_1', 'matrix_2', 'matrix_3', 'matrix_4', 'matrix_5']
        listS = ['matrix_6', 'matrix_7', 'matrix_8', 'matrix_9', 'matrix_10']
        Prototypical=[]
        shot_num = self.num
        t = self.t
        H=self.h
        B, C, h, w = input1.size()  # 50 64 21 21
        Similarity_list = []
        Similarity_list_local=[]


        for i in range(len(input2)):
                # tensor = torch.zeros(64).cuda()
                # mmd_weight=[]
                if(shot_num==1):
                    support_Prototypical[i]=average_pooling_to_vector((input2[i]))
                else:
                    for k in range(shot_num):
                        support_pro[k]=average_pooling_to_vector((input2[i])[:,k*441:(k+1)*441])
                    final_prototype = torch.zeros_like(support_pro[0])
                    for k in range(shot_num):
                        final_prototype+=support_pro[k]
                    final_prototype=final_prototype/shot_num
                    support_Prototypical[i]=final_prototype
                    tensor=tensor+support_Prototypical[k]
                
                for k in range(shot_num):
                    mmd_for_prototype = calculate_mmd_for_prototype(support_Prototypical[k], torch.cat(support_Prototypical))
                    weight = mmd_for_prototype.item()  
                    mmd_weight.append(1-weight)
        mmd_weight_tensor = torch.tensor(mmd_weight)
        total_weight = torch.sum(mmd_weight_tensor)
        
        for i in range(len(input2)):
            final_prototype = torch.zeros_like(support_Prototypical[0])
            for k in range(shot_num):
                final_prototype += mmd_weight[k] * support_Prototypical[k]
            final_prototype=final_prototype/total_weight
            support_pro[i]=final_prototype

        for i in range(B):
            wdata = []
            query_sam = input1[i]
            query_sam = query_sam.view(C, -1)  # 64 441
            query_sam = torch.transpose(query_sam, 0, 1)  # 441 64
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            query_Prototypical = average_pooling_to_vector1(query_sam)
            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, len(input2)).cuda()
                inner_sim_local=torch.zeros(1, len(input2)).cuda()
            for t in range(len(input2)):
                support_set_sam = input2[t]
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam / support_set_sam_norm
                listM[t] = query_sam @ support_set_sam
            for n in range(len(input2)):
                listS[n] = []
                listS[n], _ = listM[n].max(dim=1)
                listS[n] = torch.reshape(listS[n], [1, 441])
            matrix = torch.cat(([listS[p] for p in range(len(input2))]), 0)
            for j in range(len(input2)):
                weight_vector = self.weight_generators[j](query_sam, support_pro[j])
                for k in range(shot_num):
                    shotList[k] = (listM[j])[:, k * 441:(k + 1) * 441]
                    shotList[k], _ = shotList[k].max(dim=1)
                    shotList[k] = torch.reshape(shotList[k], [1, 441])
                topk_value, topk_index = torch.topk(listM[j], self.neighbor_k, 1)
                sim = torch.sum(topk_value, dim=1)
                data = sim * weight_vector
                inner_sim[0, j] = torch.cosine_similarity(query_Prototypical, support_Prototypical[j], dim=0, eps=1e-08)
                inner_sim_local[0,j]=torch.sum(data)
            Similarity_list.append(inner_sim)
            Similarity_list_local.append(inner_sim_local)
        Similarity_list = torch.cat(Similarity_list, 0)
        Similarity_list_local = torch.cat(Similarity_list_local, 0)

        return Similarity_list,Similarity_list_local

    def forward(self, x1, x2):
        Similarity_list, Similarity_list_local = self.cal_cosinesimilarity(x1, x2)
        return Similarity_list,Similarity_list_local




##############################################################################
# Classes: ResNetLike
##############################################################################

# Model: ResNetLike
# Refer to: https://github.com/gidariss/FewShotWithoutForgetting
# Input: One query image and a support set
# Base_model: 4 ResBlock layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->96->128->256

class ResBlock(nn.Module):
    def __init__(self, nFin, nFout):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential()
        self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
        self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL1', nn.Conv2d(nFin, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL2', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL3', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

        self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)

    def forward(self, x):
        return self.skip_layer(x) + self.conv_block(x)


class ResNetLike(nn.Module):
    def __init__(self, opt, neighbor_k=3, num=None, parameter_t=None,parameter_h=None):
        super(ResNetLike, self).__init__()

        self.in_planes = opt['in_planes']
        self.out_planes = [64, 96, 128, 256]
        # self.num_stages = 4
        self.num_stages = 4

        if type(opt['norm_layer']) == functools.partial:
            use_bias = opt['norm_layer'].func == nn.InstanceNorm2d
        else:
            use_bias = opt['norm_layer'] == nn.InstanceNorm2d

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]

        assert (type(self.out_planes) == list)
        assert (len(self.out_planes) == self.num_stages)
        num_planes = [self.out_planes[0], ] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else False
        dropout = opt['dropout'] if ('dropout' in opt) else 0

        self.feat_extractor = nn.Sequential()
        self.feat_extractor.add_module('ConvL0', nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1))

        for i in range(self.num_stages):
            self.feat_extractor.add_module('ResBlock' + str(i), ResBlock(num_planes[i], num_planes[i + 1]))
            if i < self.num_stages - 2:
                self.feat_extractor.add_module('MaxPool' + str(i), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.feat_extractor.add_module('ReluF1', nn.LeakyReLU(0.2, True))  # get Batch*256*21*21

        self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k, num=num,
                                            parameter_t=parameter_t,parameter_h=parameter_h)  # Batch*num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input1, input2):

        # extract features of input1--query image
        q = self.feat_extractor(input1)

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            support_set_sam = self.feat_extractor(input2[i])
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            S.append(support_set_sam)

        x = self.imgtoclass(q, S)  # get Batch*num_classes

        return x
