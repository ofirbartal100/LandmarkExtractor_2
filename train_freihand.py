import os
from torch.utils.tensorboard import SummaryWriter
import torchvision

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
from dataset import FreiHandDataset
from transforms import *

# dataset load
transform = ComposeKeyPoints(
    [To3ChannelsRGBKeyPoints(), ResizeKeypoints(224), RandomMirrorKeyPoints(), RandomAffineKeyPoints((-60, 60)),
     CropToKeyPoints(3), ResizeKeypoints(224), ToTensorKeyPoints(),
     ])
# NormalizeKeyPoints((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

train_set = FreiHandDataset(
    "/disk1/ofirbartal/Projects/Dataset/FreiHAND",
    transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

images, labels = next(iter(train_loader))
writer = SummaryWriter()
# writer.add_graph(model, images.cuda())
for i in range(len(images)):
    kps = KeypointsOnImage([Keypoint(*kp) for kp in labels[i]], shape=images[i].shape)
    p = kps.draw_on_image(images[i].permute(1, 2, 0))
    images[i] = torch.from_numpy(p).permute(2, 0, 1)
writer.add_image('images', torchvision.utils.make_grid(images))

writer.close()
