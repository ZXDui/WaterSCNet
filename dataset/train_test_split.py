# Randomly divide the experimental dataset into training, testing, and validation sets.

import cv2
import os
from spectral import *
import numpy as np


def dataset_split(h_path, h_train_path, h_test_path, hrgb_path, hrgb_train_path, hrgb_test_path,
                  o_path, o126_train_path, o126_test_path, orgb_train_path, orgb_test_path,
                  sl_path, sl126_train_path, sl126_test_path, slrgb_train_path, slrgb_test_path,
                  vl_path, vl126_train_path, vl126_test_path, vlrgb_train_path, vlrgb_test_path):

    save_path1 = 'Pleace your save trainval_list path'
    save_path2 = 'Pleace your save test_list path'
    ss_idx = np.load(save_path1)
    s_out = np.load(save_path2)

    # train
    num = 1
    for i in ss_idx:
        hdr = open_image(os.path.join(h_path, str(i)+'.hdr'))
        a_hdr = hdr.load()
        envi.save_image(h_train_path + "/%d.hdr" % num, a_hdr)

        hdrrgb = open_image(os.path.join(hrgb_path, str(i) + '.hdr'))
        a_hdrrgb = hdrrgb.load()
        envi.save_image(hrgb_train_path + "/%d.hdr" % num, a_hdrrgb)

        orin = cv2.imread(os.path.join(o_path, str(i)+'.jpg'), 1)
        cv2.imwrite(o126_train_path + "/%d.jpg" % num, orin)
        cv2.imwrite(orgb_train_path + "/%d.jpg" % num, orin)

        slabel = cv2.imread(os.path.join(sl_path, str(i)+'.jpg'), 1)
        cv2.imwrite(sl126_train_path + "/%d.jpg" % num, slabel)
        cv2.imwrite(slrgb_train_path + "/%d.jpg" % num, slabel)

        vlabel = cv2.imread(os.path.join(vl_path, str(i) + '.jpg'), 1)
        cv2.imwrite(vl126_train_path + "/%d.jpg" % num, vlabel)
        cv2.imwrite(vlrgb_train_path + "/%d.jpg" % num, vlabel)

        num += 1
    print('train set done')
    print("train_sample_num: ")
    print(num)

    # test
    num2 = 1
    for i in s_out:
        hdr = open_image(os.path.join(h_path, str(i)+'.hdr'))
        a_hdr = hdr.load()
        envi.save_image(h_test_path + "/%d.hdr" % num2, a_hdr)

        hdrrgb = open_image(os.path.join(hrgb_path, str(i) + '.hdr'))
        argb_hdr = hdrrgb.load()
        envi.save_image(hrgb_test_path + "/%d.hdr" % num2, argb_hdr)

        orin = cv2.imread(os.path.join(o_path, str(i)+'.jpg'), 1)
        cv2.imwrite(o126_test_path + "/%d.jpg" % num2, orin)
        cv2.imwrite(orgb_test_path + "/%d.jpg" % num2, orin)

        slabel = cv2.imread(os.path.join(sl_path, str(i)+'.jpg'), 1)
        cv2.imwrite(sl126_test_path + "/%d.jpg" % num2, slabel)
        cv2.imwrite(slrgb_test_path + "/%d.jpg" % num2, slabel)

        vlabel = cv2.imread(os.path.join(vl_path, str(i) + '.jpg'), 1)
        cv2.imwrite(vl126_test_path + "/%d.jpg" % num2, vlabel)
        cv2.imwrite(vlrgb_test_path + "/%d.jpg" % num2, vlabel)

        num2 += 1
    print('test set done')
    print("test_sample_num: ")
    print(num2)


if __name__ == '__main__':
    root_dir = 'Pleace your save crop path'
    save_dir = 'Pleace your exp path'

    hdr126_path = root_dir + '/hdr126_crop'
    hdr126_train_path = save_dir + '/b126/train/hdr'
    hdr126_test_path = save_dir + '/b126/test/hdr'

    hdrrgb_path = root_dir + '/hdrrgb_crop'
    hdrrgb_train_path = save_dir + '/brgb/train/hdr'
    hdrrgb_test_path = save_dir + '/brgb/test/hdr'

    img_path = root_dir + '/img_crop'
    img126_train_path = save_dir + '/b126/train/o'
    img126_test_path = save_dir + '/b126/test/o'
    imgrgb_train_path = save_dir + '/brgb/train/o'
    imgrgb_test_path = save_dir + '/brgb/test/o'

    slabel_path = root_dir + '/slabel_crop'
    slabel126_train_path = save_dir + '/b126/train/slabel'
    slabel126_test_path = save_dir + '/b126/test/slabel'
    slabelrgb_train_path = save_dir + '/brgb/train/slabel'
    slabelrgb_test_path = save_dir + '/brgb/test/slabel'

    vlabel_path = root_dir + '/vlabel_crop'
    vlabel126_train_path = save_dir + '/b126/train/vlabel'
    vlabel126_test_path = save_dir + '/b126/test/vlabel'
    vlabelrgb_train_path = save_dir + '/brgb/train/vlabel'
    vlabelrgb_test_path = save_dir + '/brgb/test/vlabel'

    dataset_split(hdr126_path, hdr126_train_path, hdr126_test_path, hdrrgb_path, hdrrgb_train_path, hdrrgb_test_path,
                  img_path, img126_train_path, img126_test_path, imgrgb_train_path, imgrgb_test_path,
                  slabel_path, slabel126_train_path, slabel126_test_path, slabelrgb_train_path, slabelrgb_test_path,
                  vlabel_path, vlabel126_train_path, vlabel126_test_path, vlabelrgb_train_path, vlabelrgb_test_path)

