# coding=utf-8

import torch
from thop import profile

from models import *

model = SegNetVGG16(num_class=8)
# model = TransUNetR50ViTB16(num_class=8, img_size=512)

# recommand
input = torch.randn(1, 3, 512, 512)
macs, params = profile(model, inputs=(input, ))
print('{:<30}  {:<8}'.format('Computational complexity: ', macs / (1000 * 1000 * 1000)))
print('{:<30}  {:<8}'.format('Number of parameters: ', params / (1000 * 1000)))

# # convcrf
# from models.graph_based.convcrf import convcrf
# from torch.autograd import Variable

# config = convcrf.default_conf
# config['filter_size'] = 13
# config['pyinn'] = False
# config['col_feats']['schan'] = 0.1  # normalization is used
# use_gpu = torch.cuda.is_available()

# gausscrf = convcrf.GaussCRF(conf=config, shape=(512, 512), nclasses=8)
# img_var = Variable(torch.randn(1, 3, 512, 512))
# unary_var = Variable(torch.softmax(torch.randn(1, 8, 512, 512), dim=1))
# macs, params = profile(gausscrf, inputs=(unary_var, img_var, ))
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs / (1000 * 1000 * 1000)))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params / (1000 * 1000)))