import torch
from geosynth_unet import GeoSynthUNet

model = GeoSynthUNet(in_channels=16, out_channels=13)

x = torch.randn(2, 16, 256, 256)
y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)