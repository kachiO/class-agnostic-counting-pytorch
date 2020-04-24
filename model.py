import torch, numpy as np, torch.nn as nn
from pathlib import Path
from collections import OrderedDict




class Bottleneck(nn.Module):

    def __init__(self, inplanes, chans, kernel_size=3, stride=1,  expansion=4, downsample=False):
        super(Bottleneck, self).__init__()
        planes = [chans, chans, chans * expansion]
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.adapt = nn.Conv2d(planes[0], planes[1], kernel_size=1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.conv3 = nn.Conv2d(planes[1], planes[2], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes[2])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.expansion = expansion

        # orthogonal initialization for adapt module
        for m in self.adapt.modules():
            if isinstance(m, nn.Conv2d):
                #print(m.weight.shape, m.weight.ndimension())
                nn.init.orthogonal_(m.weight, gain=0.1)

        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes[-1], kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes[-1]))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        adapt = self.adapt(out)
        out = self.conv2(out)

        out += adapt
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)

        return out



class ResNet50_Half(nn.Module):
    def __init__(self, in_chans=3, layers=[3, 4], chans=[64, 128], strides=[1, 2], expansion=4):
        super(ResNet50_Half, self).__init__()
        self.block = Bottleneck
        self.conv1 = nn.Conv2d(in_chans, chans[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(chans[0])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.inplanes = chans[0]
        self.expansion = expansion

        self.layer1 = self._make_layer(layers[0], chans[0], chans[0], stride=strides[0])
        self.layer2 = self._make_layer(layers[1], self.inplanes, chans[1], stride=strides[1])

    def _make_layer(self, nblocks, in_planes, planes, stride=1, prefix=''):
        layers = [self.block(in_planes, planes, stride=stride, downsample=True)]
        self.inplanes = planes * self.expansion
        layers += [self.block(self.inplanes, planes) for b in range(nblocks - 1)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x


def conv_block(nfin, nfout, ks, stride=1, padding=0, bias=False, bn=True, act_fn=None, convT=False, **kwargs):
    """
    Convolutional block with optional batch normalization and relu
    """
    _conv_block_list = [nn.Conv2d(nfin, nfout, ks, stride, padding=padding, bias=bias)]
    if convT:
        _conv_block_list = [nn.ConvTranspose2d(nfin, nfout, ks, stride, padding=padding, bias=bias, **kwargs)]

    if bn:
        _conv_block_list += [nn.BatchNorm2d(nfout)]

    if act_fn == 'relu':
        _conv_block_list += [nn.ReLU(inplace=True)]

    return nn.Sequential(*_conv_block_list)

class Relation_Module(nn.Module):
    """
    Relation module.
    """
    def __init__(self, in_planes, out_planes=256):
        super(Relation_Module, self).__init__()

        self.conv1 = conv_block(in_planes, out_planes, ks=3, padding=1, act_fn='relu')
        self.convT = conv_block(out_planes, out_planes, ks=3, stride=2, padding=1, convT=True, output_padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.convT(x)

        return x

class L2_Normalization(nn.Module):
    """
    L2 normalization layer with learnable parameter.
    """
    def __init__(self, scale=True, eps=1e-6):
        super(L2_Normalization, self).__init__()
        self.eps = eps
        self.scale = scale
        self.alpha = 1

        if self.scale:
            self.alpha = nn.Parameter(torch.ones(1))
            nn.init.uniform_(self.alpha, 10., 20.)

    def __repr__(self):
        return self.__class__.__name__ + f'(eps={self.eps}, alpha={self.alpha.data.tolist()[0]:.04f})'

    def forward(self, x):

        l2_norm = x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)
        l2_norm = self.alpha * l2_norm

        return l2_norm

config = OrderedDict(
            encoder=OrderedDict(
                            in_chans=3,
                            layers=[3, 4],
                            chans=[64, 128],
                            strides=[1, 2],
                            expansion=4),
            relation=dict(planes=256),
            l2norm=dict(scale=True))

class Generic_Matching_Net(nn.Module):
    """
    Generic Matching Network from Lu et al 2018
    Clas Agnostic Counting.
    """
    def __init__(self, config, pretrained=True):
        super(Generic_Matching_Net, self).__init__()
        self.encoder_patch = ResNet50_Half()
        self.encoder_image = ResNet50_Half()

        if pretrained:
            print('Loading imagenet weights.')
            from torchvision.models import resnet50
            res50 = resnet50(pretrained=True)
            self.encoder_patch.load_state_dict(res50.state_dict(), strict=False)
            self.encoder_image.load_state_dict(res50.state_dict(), strict=False)

        self.encoder_patch = nn.Sequential(self.encoder_patch, nn.AdaptiveAvgPool2d(1))
        self.l2_norm1 = L2_Normalization(config['l2norm']['scale'])
        self.l2_norm2 = L2_Normalization(config['l2norm']['scale'])
        in_planes = config['encoder']['chans'][-1] * config['encoder']['expansion'] * 2
        self.matching = Relation_Module(in_planes, config['relation']['planes'])
        self.prediction = conv_block(config['relation']['planes'], 1, ks=3, padding=1, bn=False, act_fn='relu')

    def forward(self, x):
        image, exemplar = x 
        F_image = self.l2_norm1(self.encoder_image(image))
        F_exemplar = self.l2_norm2(self.encoder_patch(exemplar))
        F_exemplar = F_exemplar.expand_as(F_image).clone()
        F = torch.cat((F_image, F_exemplar), dim=1)

        out = self.matching(F)
        out = self.prediction(out)

        return {'logits': out}

