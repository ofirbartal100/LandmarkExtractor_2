import os

import dsntnn

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from torch.utils.tensorboard import SummaryWriter
import torchvision
from dataset import GaneratedHandsDataset
from models.fully_conv_dsnt import CoordRegressionNetwork
from transforms import *
import time


def calc_dist(preds, labels):
    def dist_square(p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    dist = []
    for p, l in zip(preds, labels):
        dist += [dist_square(p1, p2) for p1, p2 in zip(p, l)]

    return np.array(dist)


# dataset load
transform = ComposeKeyPoints(
    [To3ChannelsIRKeyPoints(), ResizeKeypoints(224), RandomMirrorKeyPoints(), RandomAffineKeyPoints((-60, 60)),
     ToTensorKeyPoints(), NormalizeKeyPoints((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_set = GaneratedHandsDataset(
    "/disk1/ofirbartal/Projects/Dataset/GANeratedHands_Release/dataset_csv/test_dataset.csv",
    transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

pixel_error_threshhold_close = 5
pixel_error_threshhold_mid = 10
pixel_error_threshhold_far = 15

# # model and visualization
trained_model = CoordRegressionNetwork(n_locations=21).cuda()
trained_model = torch.nn.DataParallel(trained_model,[0,1],0)
trained_model.eval()

a = torch.load(
    "/disk1/ofirbartal/Projects/LandmarksExtraction/Checkpoints/FCN_checkpoints/07_11_19__17_35_31//epoch_94.pth")
# keys = list(a.keys())
# for key in keys:
#     a[key.replace('module.', '')] = a.pop(key)

trained_model.load_state_dict(a)

total_loss = 0
total_misses_close = 0
total_misses_mid = 0
total_misses_far = 0

start = time.time()
for images, labels in test_loader:
    cuda_images, cuda_labels = images.cuda(), labels.cuda()
    # Forward pass

    coords, heatmaps = trained_model(cuda_images)

    image_space_coords = ((coords + 1) * 224) / 2

    image_space_labels = ((cuda_labels + 1) * 224) / 2
    dist = calc_dist(image_space_coords, image_space_labels)

    total_misses_close += np.sum(dist > pixel_error_threshhold_close)
    total_misses_mid += np.sum(dist > pixel_error_threshhold_mid)
    total_misses_far += np.sum(dist > pixel_error_threshhold_far)

end = time.time()
print('Test Time {}s'.format(end - start))
acc_close = (1 - (float(total_misses_close) / (len(test_set) * 21))) * 100
acc_mid = (1 - (float(total_misses_mid) / (len(test_set) * 21))) * 100
acc_far = (1 - (float(total_misses_far) / (len(test_set) * 21))) * 100
print('Accuracy Close {}%'.format(acc_close))
print('Accuracy Mid {}%'.format(acc_mid))
print('Accuracy Far {}%'.format(acc_far))
with open('test2Results.txt', 'w') as f:
    f.write('Test Time {}s\n'.format(end - start))
    f.write('Accuracy Close {}%'.format(acc_close))
    f.write('Accuracy Mid {}%'.format(acc_mid))
    f.write('Accuracy Far {}%'.format(acc_far))
