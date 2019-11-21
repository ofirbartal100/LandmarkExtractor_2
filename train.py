import os

import dsntnn

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.optim as optim
from dataset import GaneratedHandsDataset
from models.fully_conv_dsnt import CoordRegressionNetwork
from transforms import *
import time
from datetime import datetime

# dataset load
transform = ComposeKeyPoints(
    [To3ChannelsIRKeyPoints(), ResizeKeypoints(224), RandomMirrorKeyPoints(), RandomAffineKeyPoints((-60, 60)),
     ToTensorKeyPoints(), NormalizeKeyPoints((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_set = GaneratedHandsDataset(
    "/disk1/ofirbartal/Projects/Dataset/GANeratedHands_Release/dataset_csv/train_dataset.csv",
    transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

val_set = GaneratedHandsDataset(
    "/disk1/ofirbartal/Projects/Dataset/GANeratedHands_Release/dataset_csv/val_dataset.csv",
    transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=True)

checkpoints_path = "/disk1/ofirbartal/Projects/LandmarksExtraction/Checkpoints/FCN_checkpoints/{}/".format(
    datetime.now().strftime("%d_%m_%y__%H_%M_%S"))

if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()
torch.cuda.empty_cache()
# Visualize keypoints
# example for visualization
# images, labels = next(iter(train_loader))
# #unnormalize keypoints
# labels = ((labels+1) * images.shape[-1])  / 2
# for i in range(len(images)):
#     kps = KeypointsOnImage([Keypoint(*kp) for kp in labels[i]], shape=images[i].shape)
#     p = kps.draw_on_image(images[i].permute(1, 2, 0))
#     images[i] = torch.from_numpy(p).permute(2, 0, 1)
# writer.add_image('images', torchvision.utils.make_grid(images))
# writer.add_graph(model, images)

# # model and visualization
model = CoordRegressionNetwork(n_locations=21).cuda()

model = torch.nn.DataParallel(model, [0, 1], 0)
# model = torch.nn.DataParallel(model,[0],0)
# Visualize keypoints
# example for visualization
images, labels = next(iter(train_loader))

# writer.add_graph(model, images.cuda())
for i in range(len(images)):
    kps = KeypointsOnImage([Keypoint(*kp) for kp in labels[i]], shape=images[i].shape)
    p = kps.draw_on_image(images[i].permute(1, 2, 0))
    images[i] = torch.from_numpy(p).permute(2, 0, 1)
writer.add_image('images', torchvision.utils.make_grid(images))

param_list = list(model.module.fcn.classifier.parameters()) + list(model.module.hm_conv.parameters())
optimizer = optim.RMSprop(model.module.parameters(), lr=2.5e-4)

# model.load_state_dict(
#     torch.load("/disk1/ofirbartal/Projects/LandmarksExtraction/Checkpoints/FCN_checkpoints/06_11_19__16_39_37/epoch_32.pth"))

for epoch in range(100):

    total_loss = 0
    total_correct = 0
    model.train()
    start = time.time()

    for images, labels in train_loader:  # Get Batch
        cuda_images, cuda_labels = images.cuda(), labels.cuda()
        # Forward pass
        coords, heatmaps = model(cuda_images)

        # Per-location euclidean losses
        euc_losses = dsntnn.euclidean_losses(coords, cuda_labels)
        # Per-location regularization losses
        reg_losses = dsntnn.js_reg_losses(heatmaps, cuda_labels, sigma_t=1.0)
        # Combine losses into an overall loss
        loss = dsntnn.average_loss(euc_losses + reg_losses)

        total_loss += loss.item()

        # Calculate gradients
        optimizer.zero_grad()
        loss.backward()

        # Update model parameters with RMSprop
        optimizer.step()


    end = time.time()

    torch.save(model.state_dict(), checkpoints_path + "epoch_{}.pth".format(epoch))

    writer.add_scalar('Training Loss', total_loss / len(train_loader), epoch)
    # writer.add_scalar('Accuracy', total_correct / len(train_set), epoch)
    writer.add_scalar('Training Time', end - start, epoch)

    writer.add_histogram('model.fcn.classifier[0].weight', model.module.fcn.classifier[0].weight, epoch)
    writer.add_histogram('model.fcn.classifier[4].weight', model.module.fcn.classifier[4].weight, epoch)
    writer.add_histogram('model.fcn.classifier[4].bias', model.module.fcn.classifier[4].bias, epoch)
    writer.add_histogram('hm_conv.weight', model.module.hm_conv.weight, epoch)

    torch.cuda.empty_cache()
    total_loss = 0
    model.eval()
    start = time.time()
    for images, labels in val_loader:  # Get Batch
        cuda_images, cuda_labels = images.cuda(), labels.cuda()
        # Forward pass
        coords, heatmaps = model(cuda_images)

        # Per-location euclidean losses
        euc_losses = dsntnn.euclidean_losses(coords, cuda_labels)
        # Per-location regularization losses
        reg_losses = dsntnn.js_reg_losses(heatmaps, cuda_labels, sigma_t=1.0)
        # Combine losses into an overall loss
        loss = dsntnn.average_loss(euc_losses + reg_losses)

        total_loss += loss.item()

    end = time.time()

    writer.add_scalar('Validation Loss', total_loss / len(val_loader), epoch)
    # writer.add_scalar('Accuracy', total_correct / len(train_set), epoch)
    writer.add_scalar('Validation Time', end - start, epoch)

writer.close()
