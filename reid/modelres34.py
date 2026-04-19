import torch, torchvision
import torch.nn as nn
from torch.nn import functional as F
# import dgl
from .pooling import GeneralizedMeanPoolingP

def remove_module_key(state_dict):
    for key in list(state_dict.keys()):
        if 'module' in key:
            state_dict[key.replace('module.','')] = state_dict.pop(key)
    return state_dict

class MyNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet_layer = torchvision.models.resnet34(pretrained=False)
        
        self.resnet_layer.load_state_dict(torch.load('E:/torch_cache/hub/checkpoints/resnet34-b627a593.pth'))

        self.classifier = nn.Linear(512, num_classes, bias=False)
        self.pool_bn = nn.BatchNorm1d(512)
        # self.globalpooling = nn.AdaptiveAvgPool2d(1)
        self.globalpooling = GeneralizedMeanPoolingP()

        nn.init.constant_(self.pool_bn.weight, 1)
        nn.init.constant_(self.pool_bn.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)

        self.resnet_layer = nn.Sequential(*list(self.resnet_layer.children())[:-2])
        # self.resnet_layer[-1][0].downsample[0].stride = (1,1)
        # self.resnet_layer[-1][0].conv2.stride = (1,1)


    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)

        bnx = self.pool_bn(x)
        if self.training is False:
            return F.normalize(bnx)
        y = self.classifier(bnx)
        return x,y,F.normalize(bnx)

def create_model(args):
    model = MyNet(num_classes=args.num_clusters).cuda()
    # model_ema = MyNet(num_classes=args.num_clusters).cuda()
    # model_ema.load_state_dict(model.state_dict())

    gat = None #GAT(args).cuda()
    if args.init_path:
        initial_weights = torch.load(args.init_path)
        weights = initial_weights['state_dict']
        weights.pop('classifier.weight')
        # weights.pop('classifier.bias')
        model.load_state_dict(weights, strict = False)
        # model_ema.load_state_dict(weights, strict = False)
        print(f'mAP of initial model {initial_weights["best_mAP"]}')
        args.best_map = initial_weights['best_mAP']
        args.start_epoch = initial_weights['epoch']
        args.best_epoch = args.start_epoch
        # gat_weights = initial_weights['gat_state_dict']
        # gat.load_state_dict(gat_weights)

    # for param in model_ema.parameters():
    #     param.detach_()

    return model, 1, gat