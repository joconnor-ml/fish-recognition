import numpy as np
import pandas as pd
import glob
import cv2

def get_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (21, 21), 0)
    return img

i = 0
for foreground in glob.glob("../input/*/*/*.jpg"):
    deltas = {}
    image = get_image(f1)
    for f2 in glob.glob("../input/train/NoF/*.jpg"):
        background = get_image(f2)
        if image.shape == background.shape:
            deltas[f2] = cv2.absdiff(image, background).mean()
