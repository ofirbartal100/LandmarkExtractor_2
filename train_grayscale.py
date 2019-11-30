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


def accuracy_measure(threshes):
    def calc_dist(preds, labels):
        def dist_square(p1, p2):
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

        dist = []
        for p, l in zip(preds, labels):
            dist += [dist_square(p1, p2) for p1, p2 in zip(p, l)]

        return np.array(dist)

    threshes = np.array(threshes)

    def _accuracy_measure(raw_preds, raw_labels):
        image_space_coords = ((raw_preds + 1) * 224) / 2
        image_space_labels = ((raw_labels + 1) * 224) / 2
        dist = calc_dist(image_space_coords, image_space_labels)
        misses = np.zeros((3,), dtype=int)
        for i, thresh in enumerate(threshes):
            misses[i] = np.sum(dist > thresh)

        return misses

    return _accuracy_measure


threshes = [5, 10, 15]
accuracy_func = accuracy_measure(threshes)

# dataset load

transform = ComposeKeyPoints(
    [To3ChannelsGrayscaleKeyPoints(), ResizeKeypoints(224), RandomMirrorKeyPoints(), RandomAffineKeyPoints((-60, 60)),
     CropToKeyPoints(3), ResizeKeypoints(224), ToTensorKeyPoints(),
     NormalizeKeyPoints((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     ])

train_set = GaneratedHandsDataset(
    "/disk1/ofirbartal/Projects/Dataset/GANeratedHands_Release/dataset_csv/train_dataset.csv",
    transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

val_set = GaneratedHandsDataset(
    "/disk1/ofirbartal/Projects/Dataset/GANeratedHands_Release/dataset_csv/val_dataset.csv",
    transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=True)

checkpoints_path = "/disk1/ofirbartal/Projects/LandmarksExtraction/Checkpoints/FCN_checkpoints/gray_{}/".format(
    datetime.now().strftime("%d_%m_%y__%H_%M_%S"))

if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(comment='_grayscale_')
torch.cuda.empty_cache()

# # model and visualization
model = CoordRegressionNetwork(n_locations=21).cuda()
model = torch.nn.DataParallel(model, [0, 1], 0)

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

for epoch in range(100):

    total_loss = 0
    total_correct = 0
    total_acc_misses = np.zeros((len(threshes)), dtype=int)
    model.train()
    start = time.time()

    for images, labels in train_loader:  # Get Batch
        cuda_images, cuda_labels = images.cuda(), labels.cuda()
        # Forward pass
        coords, heatmaps = model(cuda_images)

        total_acc_misses = total_acc_misses + accuracy_func(coords, cuda_labels)

        # Loss
        euc_losses = dsntnn.euclidean_losses(coords, cuda_labels)
        reg_losses = dsntnn.js_reg_losses(heatmaps, cuda_labels, sigma_t=1.0)
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

    for i, thresh in enumerate(threshes):
        writer.add_scalar('Training Acc Within {}px'.format(thresh),
                          ((1 - (float(total_acc_misses[i]) / (len(train_set) * 21))) * 100), epoch)

    torch.cuda.empty_cache()
    total_loss = 0
    total_acc_hits = np.zeros((len(threshes)), dtype=int)
    model.eval()
    start = time.time()
    for images, labels in val_loader:  # Get Batch
        cuda_images, cuda_labels = images.cuda(), labels.cuda()
        # Forward pass
        coords, heatmaps = model(cuda_images)
        total_acc_hits += accuracy_func(coords, cuda_labels)

        # Per-location euclidean losses
        euc_losses = dsntnn.euclidean_losses(coords, cuda_labels)
        reg_losses = dsntnn.js_reg_losses(heatmaps, cuda_labels, sigma_t=1.0)
        loss = dsntnn.average_loss(euc_losses + reg_losses)

        total_loss += loss.item()

    end = time.time()

    writer.add_scalar('Validation Loss', total_loss / len(val_loader), epoch)
    # writer.add_scalar('Accuracy', total_correct / len(train_set), epoch)
    writer.add_scalar('Validation Time', end - start, epoch)

    for i, thresh in enumerate(threshes):
        writer.add_scalar('Validation Acc Within {}px'.format(thresh),
                          (1 - (float(total_acc_hits[i]) / (len(val_set) * 21))) * 100, epoch)

writer.close()
