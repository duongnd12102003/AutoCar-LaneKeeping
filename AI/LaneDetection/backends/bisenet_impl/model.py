import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, k, s, p, dilation=d, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(
            ConvBNReLU(3, 64, k=3, s=2, p=1),
            ConvBNReLU(64, 64, k=3, s=1, p=1),
        )
        self.s2 = nn.Sequential(
            ConvBNReLU(64, 64, k=3, s=2, p=1),
            ConvBNReLU(64, 64, k=3, s=1, p=1),
            ConvBNReLU(64, 64, k=3, s=1, p=1),
        )
        self.s3 = nn.Sequential(
            ConvBNReLU(64, 128, k=3, s=2, p=1),
            ConvBNReLU(128, 128, k=3, s=1, p=1),
            ConvBNReLU(128, 128, k=3, s=1, p=1),
        )

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        return x

class StemBlock(nn.Module):
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, k=3, s=2, p=1)
        self.left = nn.Sequential(
            ConvBNReLU(out_ch, out_ch // 2, k=1, s=1, p=0),
            ConvBNReLU(out_ch // 2, out_ch, k=3, s=2, p=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_ch * 2, out_ch, k=3, s=1, p=1)

    def forward(self, x):
        x = self.conv(x)
        xl = self.left(x)
        xr = self.right(x)
        x = torch.cat([xl, xr], dim=1)
        x = self.fuse(x)
        return x  # (B,16,H/4,W/4)


class GELayerS1(nn.Module):
    def __init__(self, in_ch, out_ch, exp_ratio=6):
        super().__init__()
        mid_ch = in_ch * exp_ratio
        self.conv1 = ConvBNReLU(in_ch, mid_ch, k=3, s=1, p=1)
        self.dwconv = ConvBNReLU(
            mid_ch, mid_ch, k=3, s=1, p=1, groups=mid_ch
        )
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.short = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        shortcut = self.short(x)

        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.bn2(self.conv2(x))

        x = self.relu(x + shortcut)
        return x


class GELayerS2(nn.Module):
    def __init__(self, in_ch, out_ch, exp_ratio=6):
        super().__init__()
        mid_ch = in_ch * exp_ratio
        self.conv1 = ConvBNReLU(in_ch, mid_ch, k=3, s=1, p=1)
        self.dwconv1 = ConvBNReLU(
            mid_ch, mid_ch, k=3, s=2, p=1, groups=mid_ch
        )
        self.dwconv2 = ConvBNReLU(
            mid_ch, mid_ch, k=3, s=1, p=1, groups=mid_ch
        )
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.short = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 2, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        shortcut = self.short(x)

        x = self.conv1(x)
        x = self.dwconv1(x)
        x = self.dwconv2(x)
        x = self.bn2(self.conv2(x))

        x = self.relu(x + shortcut)
        return x


class CEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, in_ch, 1, 1, 0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.relu(self.conv1(feat))
        feat = feat + x
        feat = self.conv2(feat)
        return feat


class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = StemBlock(3, 16)

        self.s3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.s4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.s5_4 = GELayerS2(64, 128)
        self.s5_5 = GELayerS1(128, 128)

        self.ce = CEBlock(128, 128)

    def forward(self, x):
        x = self.stem(x)   # H/4
        x = self.s3(x)     # H/8
        x = self.s4(x)     # H/16
        x = self.s5_4(x)   # H/32
        x = self.s5_5(x)
        x = self.ce(x)
        return x  # (B,128,H/32,W/32)


class BGALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv_out = ConvBNReLU(128, 128, k=3, s=1, p=1)

    def forward(self, feat_d, feat_s):
        # feat_d: (B,128,H/8,W/8)
        # feat_s: (B,128,H/32,W/32)
        feat_s_up = F.interpolate(
            feat_s, size=feat_d.shape[2:], mode="bilinear", align_corners=False
        )
        left = self.left(feat_d)
        right = self.right(feat_s_up)
        out = self.conv_out(left + right)
        return out


class BiSeNetV2Lane(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
        self.bga = BGALayer()

        self.head = nn.Sequential(
            ConvBNReLU(128, 128, k=3, s=1, p=1),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feat_d = self.detail(x)
        feat_s = self.semantic(x)
        feat_fuse = self.bga(feat_d, feat_s)
        logits = self.head(feat_fuse)
        logits = F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        return logits


if __name__ == "__main__":
    m = BiSeNetV2Lane(num_classes=2)
    print("Params (M):", sum(p.numel() for p in m.parameters()) / 1e6)
