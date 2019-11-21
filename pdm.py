import json
from pathlib import Path

import numpy as np
import pandas as pd
import re
from random import randint
import cv2
from menpo.shape import PointCloud
from menpofit.modelinstance import OrthoPDM
from menpo.io import export_pickle, import_pickle
import time

def generate_color():
    '''
    Generates a random combination
    of red, green and blue channels
    Returns:
        (r,g,b), a generated tuple
    '''
    col = []
    for i in range(3):
        col.append(randint(0, 255))

    return tuple(col)

def show_image(img):
    '''
    Displays an image
    Args:
        img(a NumPy array of type uint 8) an image to be
        dsplayed
    '''

    # cv2.imshow('', img)
    # cv2.waitKey(5000)
    cv2.imwrite('res.jpg',img)

def draw_shapes(canvas, shapes):
    '''
    Draws shapes on canvas
    Args:
        canvas(a NumPy matrix), a background on which
        shapes are drawn
        shapes(list), shapes to be drawn
    '''

    def draw_hand(pts,color):
        cv2.polylines(canvas, [pts[:4]], False, color, 2)
        cv2.polylines(canvas, [pts[5:8]], False, color, 2)
        cv2.polylines(canvas, [pts[9:12]], False, color, 2)
        cv2.polylines(canvas, [pts[13:16]], False, color, 2)
        cv2.polylines(canvas, [pts[17:20]], False, color, 2)
        cv2.polylines(canvas, [np.array([pts[0],pts[5]])], False, color, 2)
        cv2.polylines(canvas, [np.array([pts[0],pts[9]])], False, color, 2)
        cv2.polylines(canvas, [np.array([pts[0],pts[13]])], False, color, 2)
        cv2.polylines(canvas, [np.array([pts[0],pts[17]])], False, color, 2)

    for sh in shapes:
        pts = sh.reshape((-1, 1, 2)).astype('int32')
        color = generate_color()
        draw_hand(pts,color)

    show_image(canvas)


def process_csv_input(path):

    dataset = pd.read_csv(path)

    labels = dataset['label']

    def string_array_to_np_array(str_arr):
        landmarks = re.findall("\d\d*\.?\d*", str_arr)
        landmarks = np.array(landmarks, dtype='float')
        return landmarks.reshape(-1, 2)

    landmarks_dataset = np.zeros((len(labels),21,2),dtype='float')

    for i in range(len(labels)):
        landmarks_dataset[i] = string_array_to_np_array(labels.iloc[i])

    return landmarks_dataset

# input
dataset_csv_path = "/disk1/ofirbartal/Projects/Dataset/GANeratedHands_Release/dataset_csv/test_dataset.csv"

landmarks = process_csv_input(dataset_csv_path)

start = time.time()
# PDM
training_shapes = [PointCloud(l) for l in landmarks]
shape_model = OrthoPDM(training_shapes, max_n_components=None)
shape_model.n_active_components  = 0.95
end = time.time()
# shape_model = import_pickle(Path('pdm_weights.pkl'))
print(shape_model)
print("PDM Training Time: {}s".format(end-start))
export_pickle(shape_model,Path('test_pdm_weights.pkl'),overwrite=True)

# shape_model.set_target(result_landmarks) # project the target
# shape_model.target # the projected target


# draw
#create canvas on which the triangles will be visualized
# canvas = np.full([400,400], 255).astype('uint8')

#convert to 3 channel RGB for fun colors!
# canvas = cv2.cvtColor(canvas,cv2.COLOR_GRAY2RGB)
# draw_shapes(canvas,np.add([mean],100))