import os
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import (
    BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
)

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class PIDNet(nn.Module):
    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment

        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P Branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            no_relu = (i == (blocks - 1))
            layers.append(block(inplanes, planes, stride=1, no_relu=no_relu))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        layer = block(inplanes, planes, stride, downsample, no_relu=True)
        return layer

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(self.diff3(x),
                                  size=[height_output, width_output],
                                  mode='bilinear', align_corners=algc)
        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(self.diff4(x),
                                  size=[height_output, width_output],
                                  mode='bilinear', align_corners=algc)
        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(self.spp(self.layer5(x)),
                          size=[height_output, width_output],
                          mode='bilinear', align_corners=algc)

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_


def _load_pretrained_if_available(model, cfg, imgnet_pretrained=False):
    """Safe pretrained loading: skip if path empty or not found."""
    pth = getattr(cfg.MODEL, 'PRETRAINED', '') or ''
    if not pth or not os.path.isfile(pth):
        logging.info('=> no pretrained specified (or not found). Training from scratch.')
        return model

    try:
        if imgnet_pretrained:
            logging.info(f'=> loading ImageNet pretrained from: {pth}')
            pretrained_state = torch.load(pth, map_location='cpu')
            if isinstance(pretrained_state, dict) and 'state_dict' in pretrained_state:
                pretrained_state = pretrained_state['state_dict']
            model_dict = model.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items()
                                if (k in model_dict and v.shape == model_dict[k].shape)}
            logging.info(f'Loaded {len(pretrained_state)} parameters!')
            model_dict.update(pretrained_state)
            model.load_state_dict(model_dict, strict=False)
        else:
            logging.info(f'=> loading pretrained from: {pth}')
            pretrained_dict = torch.load(pth, map_location='cpu')
            if isinstance(pretrained_dict, dict) and 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            model_dict = model.state_dict()
            # some repos save keys with "model." prefix; strip if present
            fixed = {}
            for k, v in pretrained_dict.items():
                kk = k[6:] if k.startswith('model.') else k
                if kk in model_dict and v.shape == model_dict[kk].shape:
                    fixed[kk] = v
            logging.info(f'Loaded {len(fixed)} parameters!')
            model_dict.update(fixed)
            model.load_state_dict(model_dict, strict=False)
    except Exception as e:
        logging.warning(f'=> failed to load pretrained ({e}). Training from scratch.')

    return model


def get_seg_model(cfg, imgnet_pretrained):
    # pick variant by name
    name = str(cfg.MODEL.NAME).lower()
    if 's' in name:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES,
                       planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES,
                       planes=64, ppm_planes=96, head_planes=128, augment=True)
    else:
        model = PIDNet(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES,
                       planes=64, ppm_planes=112, head_planes=256, augment=True)

    model = _load_pretrained_if_available(model, cfg, imgnet_pretrained=imgnet_pretrained)
    return model


def get_pred_model(name, num_classes):
    name = str(name).lower()
    if 's' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)
    return model


if __name__ == '__main__':
    # Speed test (optional)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_pred_model(name='pidnet_s', num_classes=19).to(device).eval()
    input = torch.randn(1, 3, 1024, 2048).to(device)

    with torch.no_grad():
        for _ in range(10):
            model(input)

        iterations = None
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize() if device.type == 'cuda' else None
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
        FPS = 1000 / latency
        print(FPS)
