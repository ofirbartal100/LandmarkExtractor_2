import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models.fully_conv_dsnt import CoordRegressionNetwork
from transforms import *
import argparse
from PIL import Image


def preproccess_image(path, channels):
    if channels == 'RGB':
        img = Image.open(path)
        img.convert('RGB')
    elif channels == 'GrayScale':
        img = Image.open(path)
        img.convert('RGB')
    else:
        return None

    np_img = np.array(img)
    seq = iaa.Sequential([iaa.Resize(224)])
    image_aug = seq(image=np_img)
    img_tensor = torch.from_numpy(image_aug.astype(np.float32)).permute(2, 0, 1)

    # move to range of 0-1
    max_v = torch.max(img_tensor)
    img_tensor /= max_v

    return TTF.normalize(img_tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def load_model(channels, parallel=False):
    model = CoordRegressionNetwork(n_locations=21).cuda()

    checkpoint_path = "/disk1/ofirbartal/Projects/LandmarksExtraction/Checkpoints/FCN_checkpoints/"

    if channels == 'RGB':
        checkpoint_path = checkpoint_path + 'RGB_26_11_19__23_46_16/epoch_38.pth'
    elif channels == 'GrayScale':
        checkpoint_path = checkpoint_path + 'gray_26_11_19__23_46_25/epoch_38.pth'
    else:
        return None

    if parallel:
        model = torch.nn.DataParallel(model, [0, 1], 0)
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        dict = torch.load(checkpoint_path)
        keys = list(dict.keys())
        for key in keys:
            dict[key.replace('module.', '')] = dict.pop(key)
        model.load_state_dict(dict)
    model.eval()
    return model


def draw_landmarks(img, landmarks):
    fingers = [1, 5, 9, 13, 17]
    for f in fingers:
        cv2.line(img, tuple(landmarks[0].astype(int)), tuple(landmarks[f]), (255, 0, 0))
        cv2.line(img, tuple(landmarks[f]), tuple(landmarks[f + 1]), (255, 0, 0))
        cv2.line(img, tuple(landmarks[f + 1]), tuple(landmarks[f + 2]), (255, 0, 0))
        cv2.line(img, tuple(landmarks[f + 2]), tuple(landmarks[f + 3]), (255, 0, 0))

    for l in landmarks:
        cv2.circle(img, tuple(l), 2, (0, 0, 0), -1)


def main(args):
    model = load_model(args.channels)

    # load image
    img = preproccess_image(args.image_path, args.channels)

    # Forward pass
    coords, heatmaps = model(img.unsqueeze(0).cuda())


    cv_img = cv2.imread(args.image_path)

    image_space_coords = ((coords.squeeze().detach().cpu().numpy() + 1) * (cv_img.shape[1],cv_img.shape[0])) / 2


    draw_landmarks(cv_img, image_space_coords.astype(int))
    cv2.imwrite("test_landmarks.jpg", cv_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Landmark Extraction On An Image.')
    parser.add_argument('--image_path', help='path to the image')
    parser.add_argument('--channels', default='RGB', help='type of image: RGB,GrayScale')

    args = parser.parse_args()
    main(args)
