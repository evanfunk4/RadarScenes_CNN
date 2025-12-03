import torch.nn as nn
import torch




class ConvBlock(nn.Module):
   def __init__(self, in_ch: int, out_ch: int) -> None:
       super().__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
           nn.BatchNorm2d(out_ch),
           nn.ReLU(inplace=True),


           nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
           nn.BatchNorm2d(out_ch),
           nn.ReLU(inplace=True),
       )


   def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.conv(x)




class UNetBEV(nn.Module):
   """
   Simple UNet-style encoder-decoder for BEV semantic segmentation.
   Input:  (B, C_in, H, W)
   Output: (B, num_classes, H, W)
   """


   def __init__(self, in_channels: int, num_classes: int) -> None:
       super().__init__()


       # Encoder
       self.enc1 = ConvBlock(in_channels, 32)
       self.pool1 = nn.MaxPool2d(2)
       self.enc2 = ConvBlock(32, 64)
       self.pool2 = nn.MaxPool2d(2)
       self.enc3 = ConvBlock(64, 128)
       self.pool3 = nn.MaxPool2d(2)
       self.enc4 = ConvBlock(128, 256)


       # Decoder
       self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
       self.dec3 = ConvBlock(256, 128)
       self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
       self.dec2 = ConvBlock(128, 64)
       self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
       self.dec1 = ConvBlock(64, 32)


       self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)


   def forward(self, x: torch.Tensor) -> torch.Tensor:
       # x: (B, C_in, H, W)
       e1 = self.enc1(x)          # (B, 32, H, W)
       p1 = self.pool1(e1)        # (B, 32, H/2, W/2)


       e2 = self.enc2(p1)         # (B, 64, H/2, W/2)
       p2 = self.pool2(e2)        # (B, 64, H/4, W/4)


       e3 = self.enc3(p2)         # (B, 128, H/4, W/4)
       p3 = self.pool3(e3)        # (B, 128, H/8, W/8)


       bottleneck = self.enc4(p3) # (B, 256, H/8, W/8)


       u3 = self.up3(bottleneck)  # (B, 128, H/4, W/4)
       d3 = self.dec3(torch.cat([u3, e3], dim=1))


       u2 = self.up2(d3)          # (B, 64, H/2, W/2)
       d2 = self.dec2(torch.cat([u2, e2], dim=1))


       u1 = self.up1(d2)          # (B, 32, H, W)
       d1 = self.dec1(torch.cat([u1, e1], dim=1))


       out = self.out_conv(d1)    # (B, num_classes, H, W)
       return out
