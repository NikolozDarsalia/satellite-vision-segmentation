import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Utils of the Modules in UNet

# 1. Basic Convolution Module
class UNetConvolutionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.BatchNorm1 = nn.BatchNorm2d(output_channels)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.BatchNorm2 = nn.BatchNorm2d(output_channels)
        self.ReLU2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = self.ReLU1(x)
        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU2(x)
        return x


# 2. Down sampling & Convolution Module
class UNetDownSamplingModule(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = UNetConvolutionModule(input_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


# 3. Up sampling & Convolution Module
class UNetUpSamplingModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = UNetConvolutionModule(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 4. Final output convolution module
class UNetOutputConvolutionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class UNet_conc(nn.Module):
    def __init__(self, input_channels, output_classes):
        super().__init__()
        self.n_channels = input_channels
        self.n_classes = output_classes
        mid_channels = [16, 32, 64, 128, 256]
        self.inc = (UNetConvolutionModule(input_channels, mid_channels[0]))
        self.down1 = (UNetDownSamplingModule(mid_channels[0], mid_channels[1]))
        self.down2 = (UNetDownSamplingModule(mid_channels[1], mid_channels[2]))
        self.down3 = (UNetDownSamplingModule(mid_channels[2], mid_channels[3]))
        self.down4 = (UNetDownSamplingModule(mid_channels[3], mid_channels[4]))

        self.up1 = (UNetUpSamplingModule(mid_channels[4]*2+mid_channels[3]*2, mid_channels[3]))
        self.up2 = (UNetUpSamplingModule(mid_channels[3]+mid_channels[2]*2, mid_channels[2]))
        self.up3 = (UNetUpSamplingModule(mid_channels[2]+mid_channels[1]*2, mid_channels[1]))
        self.up4 = (UNetUpSamplingModule(mid_channels[1]+mid_channels[0]*2, mid_channels[0]))
        self.outc = (UNetOutputConvolutionModule(mid_channels[0], output_classes))

    def forward(self, x1, x2):
        # with the siamese structure, the encoder will encode the pre-disaster image & post-disaster image separately.
        # encode pre-disaster image

        x11 = self.inc(x1)
        x12 = self.down1(x11)
        x13 = self.down2(x12)
        x14 = self.down3(x13)
        x15 = self.down4(x14)

        # encode post-disaster image
        x21 = self.inc(x2)
        x22 = self.down1(x21)
        x23 = self.down2(x22)
        x24 = self.down3(x23)
        x25 = self.down4(x24)

        # concatenate the same level features in the channel dimension
        x1 = torch.cat((x11, x21), dim=1)
        x2 = torch.cat((x12, x22), dim=1)
        x3 = torch.cat((x13, x23), dim=1)
        x4 = torch.cat((x14, x24), dim=1)
        x5 = torch.cat((x15, x25), dim=1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

# Let's rewrite the UNet_conc to use the resnet34 retrained by ImageNet as encoder
class UNet_conc_pretrained_resnet34(nn.Module):
    def __init__(self, input_channels, output_classes, pretrained=True):
        super().__init__()
        self.n_channels = input_channels
        self.n_classes = output_classes
        mid_channels = [64, 128, 256, 512] # change some structures to adjust to resnet34

        # here we use the pretrained resnet34 to replace the original encoder modules
        resnet = models.resnet34(pretrained=pretrained)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.center = UNetConvolutionModule(mid_channels[3], mid_channels[3])

        self.up1 = UNetUpSamplingModule(mid_channels[3]*2+mid_channels[3]*2, mid_channels[3])
        self.up2 = UNetUpSamplingModule(mid_channels[3]+mid_channels[2]*2, mid_channels[2])
        self.up3 = UNetUpSamplingModule(mid_channels[2]+mid_channels[1]*2, mid_channels[1])
        self.up4 = UNetUpSamplingModule(mid_channels[1]+mid_channels[0]*2, mid_channels[0])
        self.conv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                      UNetConvolutionModule(mid_channels[0], mid_channels[0]//2),
                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.outc = UNetOutputConvolutionModule(mid_channels[0]//2, output_classes)

    def forward(self, x1, x2):

        # with the siamese structure, the encoder will encode the pre-disaster image & post-disaster image separately.
        # encode pre-disaster image
        x1 = self.conv1(x1)
        x1 = self.maxpool(x1)
        x11 = self.layer1(x1)
        x12 = self.layer2(x11)
        x13 = self.layer3(x12)
        x14 = self.layer4(x13)
        x15 = self.center(x14)

        # encode post-disaster image
        x2 = self.conv1(x2)
        x2 = self.maxpool(x2)
        x21 = self.layer1(x2)
        x22 = self.layer2(x21)
        x23 = self.layer3(x22)
        x24 = self.layer4(x23)
        x25 = self.center(x24)

        # concatenate the same level features in the channel dimension
        x1 = torch.cat((x11, x21), dim=1)
        x2 = torch.cat((x12, x22), dim=1)
        x3 = torch.cat((x13, x23), dim=1)
        x4 = torch.cat((x14, x24), dim=1)
        x5 = torch.cat((x15, x25), dim=1)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.conv2(x)

        logits = self.outc(x)
        return logits

def get_model_UNET():
  # you can use the below model to train from the scratch
  # return UNet_conc(3, 5).to(device)
  # or use the pretrained weight
  return UNet_conc_pretrained_resnet34(3, 5)