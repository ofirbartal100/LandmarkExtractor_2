import os

import dsntnn

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from torch.utils.tensorboard import SummaryWriter
import torchvision
from dataset import GaneratedHandsDataset
from models.fully_conv_dsnt import CoordRegressionNetwork
from transforms import *
import time
from datetime import datetime
from menpo.shape import PointCloud
from menpofit.modelinstance import OrthoPDM
from menpo.io import export_pickle, import_pickle
from pathlib import Path


def calc_misses(preds, labels, thresh):
    def dist_square(p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    misses = 0

    thresh_square = thresh * thresh
    for p, l in zip(preds, labels):
        misses += np.sum(np.array([dist_square(p1, p2) for p1, p2 in zip(p, l)]) > thresh_square)

    return misses

def pdm_reconstruction(raw_coords,pdm_model):
    point_clouds_coordinates = [PointCloud(img_labels) for img_labels in raw_coords.detach().cpu().numpy()]
    reconstructed_point_clouds = []
    for pcc in point_clouds_coordinates:
        pdm_model.set_target(pcc)  # project the target
        reconstructed_point_clouds.append(np.array(pdm_model.target.as_vector()).reshape(-1,2)) # the projected target

    return np.array(reconstructed_point_clouds)

# dataset load
transform = ComposeKeyPoints(
    [To3ChannelsIRKeyPoints(), ResizeKeypoints(224), RandomMirrorKeyPoints(), RandomAffineKeyPoints((-60, 60)),
     ToTensorKeyPoints(), NormalizeKeyPoints((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_set = GaneratedHandsDataset(
    "/disk1/ofirbartal/Projects/Dataset/GANeratedHands_Release/dataset_csv_small/test_dataset.csv",
    transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True)

pixel_error_threshhold = 20

model = CoordRegressionNetwork(n_locations=21).cuda()
model = torch.nn.DataParallel(model, [0, 1], 0)

model.load_state_dict(
    torch.load(
        "/disk1/ofirbartal/Projects/LandmarksExtraction/Checkpoints/FCN_checkpoints/07_11_19__17_35_31//epoch_94.pth"))

torch.cuda.empty_cache()
total_loss = 0
total_misses = 0
total_pdm_misses = 0
model.eval()

shape_model = import_pickle(Path('big_train_pdm_weights.pkl'))
print(shape_model)

start = time.time()
for images, labels in test_loader:
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

    image_space_coords = ((coords + 1) * 224) / 2
    pdm_reconstruction_coords = pdm_reconstruction(image_space_coords,shape_model)

    image_space_labels = ((cuda_labels + 1) * 224) / 2
    misses = calc_misses(image_space_coords, image_space_labels, pixel_error_threshhold)
    pdm_misses = calc_misses(pdm_reconstruction_coords, image_space_labels, pixel_error_threshhold)
    total_misses += misses
    total_pdm_misses += pdm_misses

end = time.time()
print('Test Time {}s'.format(end - start))
print('Test Loss {}'.format(total_loss / len(test_set)))
acc = (1 - (float(total_misses) / (len(test_set) * 42))) * 100
print('Accuracy {}%'.format(acc))
pdm_acc = (1 - (float(total_pdm_misses) / (len(test_set) * 42))) * 100
print('PDM Reconstruction Accuracy {}%'.format(pdm_acc))
with open('testResults.txt', 'w') as f:
    f.write('Test Time {}s\n'.format(end - start))
    f.write('Test Loss {}'.format(total_loss))
    f.write('Accuracy {}%'.format(acc))
    f.write('PDM Reconstruction Accuracy {}%'.format(pdm_acc))
