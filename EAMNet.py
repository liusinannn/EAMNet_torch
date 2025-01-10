import torch
import torch.nn as nn
import torch.nn.functional as F



class FEM_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FEM_1, self).__init__()
        self.conv_d1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU()
                                     )
        self.conv_d2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU()
                                     )
        self.conv_d3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=3, padding=3),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU()
                                     )
        self.conv_d4 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=4, padding=4),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU()
                                     )
        self.conv_fusion = nn.Sequential(nn.Conv2d(in_channels=4 * out_channels, out_channels=out_channels, kernel_size=1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU()
                                         )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        out_d1 = self.conv_d1(x)
        out_d2 = self.conv_d2(x)
        out_d3 = self.conv_d3(x)
        out_d4 = self.conv_d4(x)

        out = torch.cat((out_d1, out_d2, out_d3, out_d4), dim=1)
        out = self.conv_fusion(out)

        ca = self.channel_attention(out)
        out = out * ca

        out += identity
        return out


def crop(data1, h, w, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert (h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
    return data

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class sidebranch(nn.Module):
    def __init__(self,in_channels,factor):
        super(sidebranch, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=21, kernel_size=1, stride=1, padding=0)
        kernel_size = (2 * factor, 2 * factor)
        self.upsample = nn.ConvTranspose2d(in_channels=21, out_channels=1, kernel_size=kernel_size, stride=factor, padding=(factor//2),bias=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class UP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UP, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, d,e):
        d = self.up(d)
        cat = torch.cat([e, d], dim=1)
        out = self.block(cat)
        return out


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class EAMNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1):
        super(EAMNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.activ = nn.ReLU(inplace=True)

        self.inc = DoubleConv(n_channels, 64)
        self.inc2 = DoubleConv(64, 128)
        self.inc3 = DoubleConv(128,256)
        self.inc4 = DoubleConv(256, 512)
        self.inc5 = DoubleConv(512, 512)

        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv3_1_edge = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_1_edge = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_edge = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_edge = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_edge = nn.Conv2d(512, 512, 3, padding=1)

        self.up1 = UP(1024, 256,256)
        self.up2 = UP(512, 128,128)
        self.up3 = UP(256, 64,64)
        self.up4 = UP(128, 64,64)

        self.edge_up1 = sidebranch(64,1)
        self.edge_up2 = sidebranch(128,2)
        self.edge_up3 = sidebranch(256,4)
        self.edge_up4 = sidebranch(512,8)
        self.edge_up5 = sidebranch(512,16)

        self.fem_mask = FEM_1(64, 64)
        self.fem_dist = FEM_1(64, 64)

        self.outc_dist = OutConv(64, n_classes)
        self.outc_mask = OutConv(64, n_classes)
        self.outc_hed = OutConv(5, 1)

        self.coord_att_bridge = CoordAtt(512, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x1_down = self.maxpool1(x1)
        x2 = self.inc2(x1_down)
        x2_down = self.maxpool2(x2)
        x3 = self.inc3(x2_down)
        x3_down = self.maxpool3(x3)
        x4 = self.inc4(x3_down)
        x4_down = self.maxpool4(x4)
        x5 = self.inc5(x4_down)

        x5 = self.coord_att_bridge(x5)

        x1_edge = x1
        x2_edge = x2
        x3_edge = self.activ(self.conv3_1_edge(x3))
        x4_edge = self.activ(self.conv4_1_edge(x4))
        x5_edge_3 = self.activ(self.conv5_3_edge(x5))

        edge_1 = self.edge_up1(x1_edge)
        edge_1 = crop(edge_1, 512, 512, 1, 1)
        edge_2 = self.edge_up2(x2_edge)
        edge_3 = self.edge_up3(x3_edge)
        edge_4 = self.edge_up4(x4_edge)
        edge_5 = self.edge_up5(x5_edge_3)
        fuse = torch.cat((edge_1,edge_2,edge_3,edge_4,edge_5),dim=1)

        edges = self.outc_hed(fuse)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_mask = self.fem_mask(x)

        mask = self.outc_mask(x_mask)
        dist = self.outc_dist(x_mask)

        return mask,dist,edges
        # return x5

if __name__ == '__main__':
    batch_size = 1
    img_height = 512
    img_width = 512

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda:0"
    input = torch.rand(batch_size, 3, img_height, img_width).to(device)

    print(f"input shape: {input.shape}")
    model = EAMNet().to(device)
    output = model(input)
    print(f"output shapes: {[t.shape for t in output]}")
