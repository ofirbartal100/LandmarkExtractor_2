import torch
from torch import nn
import dsntnn
import collections
import torchvision.models


class FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(3, 16, kernel_size=3, padding=1)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(16)),
            ('conv2', nn.Conv2d(16, 16, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(16)),
            ('conv3', nn.Conv2d(16, 16, kernel_size=3, padding=1)),
        ])
        )

    def forward(self, x):
        return self.layers(x)


class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        # self.fcn = FCN()
        self.fcn = torchvision.models.segmentation.fcn_resnet50(num_classes=n_locations, pretrained_backbone=False)
        state_dict = torch.load(
            "/home/ofirbartal/Projects/LandmarksExtractor_2/checkpoints/torch/resnet50-19c8e357.pth")
        self.load_my_state_dict(self.fcn.backbone, state_dict)
        self.hm_conv = nn.Conv2d(n_locations, n_locations, kernel_size=1, bias=False)

    def load_my_state_dict(self, model, new_state_dict):
        own_state = model.state_dict()
        for name, param in new_state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def forward(self, images):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(images)['out']
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps


class Autoencoder(nn.Module):
    def __init__(self, n_locations=21):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1 * n_locations, out_channels=32 * n_locations, kernel_size=3, stride=2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=32 * n_locations, out_channels=64 * n_locations, stride=2, kernel_size=3),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=64 * n_locations, out_channels=64 * n_locations, stride=2, kernel_size=3),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=64 * n_locations, out_channels=64 * n_locations, stride=2, kernel_size=3)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64 * n_locations, out_channels=64 * n_locations, kernel_size=3,
                                     stride=2),

            torch.nn.ConvTranspose2d(in_channels=64 * n_locations, out_channels=64 * n_locations, kernel_size=3,
                                     stride=2),

            torch.nn.ConvTranspose2d(in_channels=64 * n_locations, out_channels=32 * n_locations, kernel_size=3,
                                     stride=2),

            torch.nn.ConvTranspose2d(in_channels=32 * n_locations, out_channels=1 * n_locations, kernel_size=4,
                                     stride=2)
        )

    def forward(self, input):
        latent = self.encoder(input)
        return self.decoder(latent)
