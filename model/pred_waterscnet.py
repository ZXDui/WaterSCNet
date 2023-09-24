# Using trained WaterSCNet for river segmentation and connectivity reconstruction.

import os
import cv2
from spectral import *
from glob import *
from wv.model.my_waterscnet import *


def pred_s(model_path, count_path, save_path):
    HEIGHT = 256
    WIDTH = 256
    IMAGE_SHAPE = [256, 256, 12]

    model = clone_atteunet4pred(IMAGE_SHAPE)

    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    file_list = glob(os.path.join(count_path, '*.hdr'))
    filelen = len(file_list) + 1
    for i in range(1, filelen):
        hdr_path = count_path + '/' + str(i) + '.hdr'
        img = open_image(hdr_path)
        a_img = img.load()
        n_img = np.array(a_img)

        img = n_img.reshape(-1, HEIGHT, WIDTH, 12)
        pr = model.predict(img)[0]

        pr = pr.reshape(HEIGHT, WIDTH)
        pr = np.where(pr > 0.5, np.ones_like(pr), np.zeros_like(pr))

        seg_img = np.zeros((HEIGHT, WIDTH))
        seg_img[:, :] = pr[:, :] * 255

        cv2.imwrite(save_path + str(i) + '.jpg', seg_img)

    print("pred done!")


def pred_v(model_path, count_path, save_path):
    HEIGHT = 256
    WIDTH = 256
    IMAGE_SHAPE = [256, 256, 12]

    model = v2c_sANDv(input_shape=IMAGE_SHAPE)

    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    file_list = glob(os.path.join(count_path, '*.hdr'))
    filelen = len(file_list) + 1
    for i in range(1, filelen):
        hdr_path = count_path + '/' + str(i) + '.hdr'
        img = open_image(hdr_path)
        a_img = img.load()
        n_img = np.array(a_img)

        img = n_img.reshape(-1, HEIGHT, WIDTH, 12)
        pr = model.predict(img)[0]

        pr = pr.reshape(HEIGHT, WIDTH)
        pr = np.where(pr > 0.5, np.ones_like(pr), np.zeros_like(pr))

        seg_img = np.zeros((HEIGHT, WIDTH))
        seg_img[:, :] = pr[:, :] * 255

        cv2.imwrite(save_path + str(i) + '.jpg', seg_img)

    print("pred done!")


if __name__ == "__main__":
    model_dir1 = 'Pleace your model log path'
    count_dir1 = 'Pleace your test dataset path'
    save_dir1 = 'Pleace your save path for model output'
    pred_s(model_dir1, count_dir1, save_dir1)

    model_dir2 = 'Pleace your model log path'
    count_dir2 = 'Pleace your test dataset path'
    save_dir2 = 'Pleace your save path for model output'
    pred_v(model_dir2, count_dir2, save_dir2)
