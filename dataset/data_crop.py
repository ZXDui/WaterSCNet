# 1. Used to cut the original Sentinel-2 remote sensing data and corresponding segmentation labels and connectivity labels.
# 2. Filter the cut sample images to remove samples with poor representativeness.

import os
import numpy as np
from spectral import *
from glob import *
import cv2


def filter_pixel(p):
    cv2.imwrite("Pleace your save path", p)
    patch = cv2.imread("Pleace your img path", 0)
    retVal, patch = cv2.threshold(patch, 30, 1, cv2.THRESH_BINARY)
    flag = 0
    n = 40 * 40

    all_pixel = 256 * 256
    cnt_array = np.where(patch, 0, 1)
    non_water_pixel = np.sum(cnt_array)
    water_pixel = all_pixel - non_water_pixel
    if (water_pixel >= n) and (water_pixel <= all_pixel - n) and (water_pixel - non_water_pixel <= n):
        flag = 1
    return flag


def read_hdr(hdr_path):
    img = open_image(hdr_path)
    if img == None:
        print(hdr_path + "File cannot be opened!")
    return img


def hdr_orin_label_crop(hm_path, hm_spath, hrgb_path, hrgb_spath, o_path, o_spath,
                        sl_path, sl_spath, vl_path, vl_spath, crop_size, repetition_rate):

    slab = cv2.imread(sl_path)
    slab_img = np.array(slab)

    vlab = cv2.imread(vl_path)
    vlab_img = np.array(vlab)

    orin = cv2.imread(o_path)
    orin_img = np.array(orin)

    hm = read_hdr(hm_path)
    hm_img = hm.load()

    hrgb = read_hdr(hrgb_path)
    hrgb_img = hrgb.load()

    raw = slab_img.shape[0]
    col = slab_img.shape[1]

    lab_count_path = sl_spath
    lab_filelist = glob(os.path.join(lab_count_path, '*.jpg'))
    new_name = len(lab_filelist) + 1
    print("new_name", new_name)
    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
            cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
            cropped_start_col = int(j * crop_size * (1 - repetition_rate))

            slab_cropped = slab_img[cropped_start_raw: cropped_start_raw + crop_size,
                           cropped_start_col: cropped_start_col + crop_size, :]

            flag = filter_pixel(slab_cropped)
            if flag == 1:
                vlab_cropped = vlab_img[cropped_start_raw: cropped_start_raw + crop_size,
                               cropped_start_col: cropped_start_col + crop_size, :]
                orin_cropped = orin_img[cropped_start_raw: cropped_start_raw + crop_size,
                               cropped_start_col: cropped_start_col + crop_size, :]
                hm_cropped = hm_img[cropped_start_raw: cropped_start_raw + crop_size,
                             cropped_start_col: cropped_start_col + crop_size, :]
                hrgb_cropped = hrgb_img[cropped_start_raw: cropped_start_raw + crop_size,
                               cropped_start_col: cropped_start_col + crop_size, :]

                cv2.imwrite(sl_spath + "/%d.jpg" % new_name, slab_cropped)
                cv2.imwrite(vl_spath + "/%d.jpg" % new_name, vlab_cropped)
                cv2.imwrite(o_spath + "/%d.jpg" % new_name, orin_cropped)
                envi.save_image(hm_spath + "/%d.hdr" % new_name, hm_cropped)
                envi.save_image(hrgb_spath + "/%d.hdr" % new_name, hrgb_cropped)

                new_name = new_name + 1

    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
        slab_cropped = slab_img[cropped_start_raw: cropped_start_raw + crop_size, (col - crop_size): col, :]
        flag = filter_pixel(slab_cropped)
        if flag == 1:
            vlab_cropped = vlab_img[cropped_start_raw: cropped_start_raw + crop_size, (col - crop_size): col, :]
            orin_cropped = orin_img[cropped_start_raw: cropped_start_raw + crop_size, (col - crop_size): col, :]
            hm_cropped = hm_img[cropped_start_raw: cropped_start_raw + crop_size, (col - crop_size): col, :]
            hrgb_cropped = hrgb_img[cropped_start_raw: cropped_start_raw + crop_size, (col - crop_size): col, :]

            cv2.imwrite(sl_spath + "/%d.jpg" % new_name, slab_cropped)
            cv2.imwrite(vl_spath + "/%d.jpg" % new_name, vlab_cropped)
            cv2.imwrite(o_spath + "/%d.jpg" % new_name, orin_cropped)
            envi.save_image(hm_spath + "/%d.hdr" % new_name, hm_cropped)
            envi.save_image(hrgb_spath + "/%d.hdr" % new_name, hrgb_cropped)

            new_name = new_name + 1

    for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_col = int(j * crop_size * (1 - repetition_rate))
        slab_cropped = slab_img[(raw - crop_size): raw, cropped_start_col: cropped_start_col + crop_size, :]
        flag = filter_pixel(slab_cropped)
        if flag == 1:
            vlab_cropped = vlab_img[(raw - crop_size): raw, cropped_start_col: cropped_start_col + crop_size, :]
            orin_cropped = orin_img[(raw - crop_size): raw, cropped_start_col: cropped_start_col + crop_size, :]
            hm_cropped = hm_img[(raw - crop_size): raw, cropped_start_col: cropped_start_col + crop_size, :]
            hrgb_cropped = hrgb_img[(raw - crop_size): raw, cropped_start_col: cropped_start_col + crop_size, :]

            cv2.imwrite(sl_spath + "/%d.jpg" % new_name, slab_cropped)
            cv2.imwrite(vl_spath + "/%d.jpg" % new_name, vlab_cropped)
            cv2.imwrite(o_spath + "/%d.jpg" % new_name, orin_cropped)
            envi.save_image(hm_spath + "/%d.hdr" % new_name, hm_cropped)
            envi.save_image(hrgb_spath + "/%d.hdr" % new_name, hrgb_cropped)

            new_name = new_name + 1

    slab_cropped = slab_img[(raw - crop_size): raw, (col - crop_size): col, :]
    flag = filter_pixel(slab_cropped)
    if flag == 1:
        vlab_cropped = vlab_img[(raw - crop_size): raw, (col - crop_size): col, :]
        orin_cropped = orin_img[(raw - crop_size): raw, (col - crop_size): col, :]
        hm_cropped = hm_img[(raw - crop_size): raw, (col - crop_size): col, :]
        hrgb_cropped = hrgb_img[(raw - crop_size): raw, (col - crop_size): col, :]

        cv2.imwrite(sl_spath + "/%d.jpg" % new_name, slab_cropped)
        cv2.imwrite(vl_spath + "/%d.jpg" % new_name, vlab_cropped)
        cv2.imwrite(o_spath + "/%d.jpg" % new_name, orin_cropped)
        envi.save_image(hm_spath + "/%d.hdr" % new_name, hm_cropped)
        envi.save_image(hrgb_spath + "/%d.hdr" % new_name, hrgb_cropped)

        new_name = new_name + 1


def hdr126_crop(hdr_path, save_path, crop_size, repetition_rate):
    img = read_hdr(hdr_path)
    a_img = img.load()
    raw = a_img.shape[0]
    col = a_img.shape[1]
    channels = a_img.shape[2]

    count_path = save_path
    filelist = glob(os.path.join(count_path, '*.hdr'))
    new_name = len(filelist) + 1

    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
            cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
            cropped_start_col = int(j * crop_size * (1 - repetition_rate))
            cropped = a_img[cropped_start_raw: cropped_start_raw + crop_size,
                      cropped_start_col: cropped_start_col + crop_size, :]

            envi.save_image(save_path + "/%d.hdr" % new_name, cropped)

            new_name = new_name + 1

    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
        cropped = img[cropped_start_raw: cropped_start_raw + crop_size, (col - crop_size): col, :]

        envi.save_image(save_path + "/%d.hdr" % new_name, cropped)
        new_name = new_name + 1

    for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_col = int(j * crop_size * (1 - repetition_rate))
        cropped = img[(raw - crop_size): raw, cropped_start_col: cropped_start_col + crop_size, :]

        envi.save_image(save_path + "/%d.hdr" % new_name, cropped)
        new_name = new_name + 1

    cropped = img[(raw - crop_size): raw, (col - crop_size): col, :]

    envi.save_image(save_path + "/%d.hdr" % new_name, cropped)
    new_name = new_name + 1


def hdrrgb_crop(hdr_path, save_path, crop_size, repetition_rate):
    img = read_hdr(hdr_path)
    a_img = img.load()
    raw = a_img.shape[0]
    col = a_img.shape[1]
    channels = a_img.shape[2]

    count_path = save_path
    filelist = glob(os.path.join(count_path, '*.hdr'))
    new_name = len(filelist) + 1
    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
            cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
            cropped_start_col = int(j * crop_size * (1 - repetition_rate))
            cropped = a_img[cropped_start_raw: cropped_start_raw + crop_size,
                      cropped_start_col: cropped_start_col + crop_size, :]

            envi.save_image(save_path + "/%d.hdr" % new_name, cropped)

            new_name = new_name + 1

    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
        cropped = img[cropped_start_raw: cropped_start_raw + crop_size, (col - crop_size): col, :]

        envi.save_image(save_path + "/%d.hdr" % new_name, cropped)
        new_name = new_name + 1

    for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_col = int(j * crop_size * (1 - repetition_rate))
        cropped = img[(raw - crop_size): raw, cropped_start_col: cropped_start_col + crop_size, :]

        envi.save_image(save_path + "/%d.hdr" % new_name, cropped)
        new_name = new_name + 1

    cropped = img[(raw - crop_size): raw, (col - crop_size): col, :]

    envi.save_image(save_path + "/%d.hdr" % new_name, cropped)
    new_name = new_name + 1


def orin_crop(orin_label_path, save_path, crop_size, repetition_rate):
    img = cv2.imread(orin_label_path)
    n_img = np.array(img)
    raw = n_img.shape[0]
    col = n_img.shape[1]
    channels = n_img.shape[2]

    new_name = len(os.listdir(save_path)) + 1
    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):

            cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
            cropped_start_col = int(j * crop_size * (1 - repetition_rate))
            cropped = n_img[cropped_start_raw: cropped_start_raw + crop_size,
                      cropped_start_col: cropped_start_col + crop_size, :]

            cv2.imwrite(save_path + "/%d.jpg" % new_name, cropped)

            new_name = new_name + 1

    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
        cropped = n_img[cropped_start_raw: cropped_start_raw + crop_size, (col - crop_size): col, :]

        cv2.imwrite(save_path + "/%d.jpg" % new_name, cropped)
        new_name = new_name + 1

    for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_col = int(j * crop_size * (1 - repetition_rate))
        cropped = n_img[(raw - crop_size): raw, cropped_start_col: cropped_start_col + crop_size, :]

        cv2.imwrite(save_path + "/%d.jpg" % new_name, cropped)
        new_name = new_name + 1

    cropped = n_img[(raw - crop_size): raw, (col - crop_size): col, :]

    cv2.imwrite(save_path + "/%d.jpg" % new_name, cropped)
    new_name = new_name + 1


def label_crop(orin_label_path, save_path, crop_size, repetition_rate):
    img = cv2.imread(orin_label_path)
    n_img = np.array(img)
    raw = n_img.shape[0]
    col = n_img.shape[1]
    channels = n_img.shape[2]

    new_name = len(os.listdir(save_path)) + 1
    print(new_name)
    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
            cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
            cropped_start_col = int(j * crop_size * (1 - repetition_rate))
            cropped = n_img[cropped_start_raw: cropped_start_raw + crop_size,
                      cropped_start_col: cropped_start_col + crop_size, :]

            cv2.imwrite(save_path + "/%d.jpg" % new_name, cropped)

            new_name = new_name + 1

    for i in range(int((raw - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_raw = int(i * crop_size * (1 - repetition_rate))
        cropped = n_img[cropped_start_raw: cropped_start_raw + crop_size, (col - crop_size): col, :]

        cv2.imwrite(save_path + "/%d.jpg" % new_name, cropped)
        new_name = new_name + 1

    for j in range(int((col - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        cropped_start_col = int(j * crop_size * (1 - repetition_rate))
        cropped = n_img[(raw - crop_size): raw, cropped_start_col: cropped_start_col + crop_size, :]

        cv2.imwrite(save_path + "/%d.jpg" % new_name, cropped)
        new_name = new_name + 1

    cropped = n_img[(raw - crop_size): raw, (col - crop_size): col, :]

    cv2.imwrite(save_path + "/%d.jpg" % new_name, cropped)
    new_name = new_name + 1


if __name__ == '__main__':
    root_dir = 'Pleace your root path'
    save_dir = 'Pleace your save path'

    path_list = glob(os.path.join(root_dir, 'hdr_multi', '*.hdr'))
    for i in range(len(path_list)):
        hdr126_path = root_dir + '/hdr_multi/' + str(i + 1) + '.hdr'
        save_hdr126_path = save_dir + '/hdr126_crop'

        hdrrgb_path = root_dir + '/hdr_rgb/' + str(i + 1) + '.hdr'
        save_hdrrgb_path = save_dir + '/hdrrgb_crop'

        img_path = root_dir + '/img/' + str(i + 1) + '.jpg'
        save_img_path = save_dir + '/img_crop'

        slabel_path = root_dir + '/label/seg/' + str(i + 1) + '.jpg'
        save_slabel_path = save_dir + '/slabel_crop'

        vlabel_path = root_dir + '/label/vec/' + str(i + 1) + '.jpg'
        save_vlabel_path = save_dir + '/vlabel_crop'

        rep_rate = 0.1
        hdr_orin_label_crop(hdr126_path, save_hdr126_path, hdrrgb_path, save_hdrrgb_path, img_path, save_img_path,
                            slabel_path, save_slabel_path, vlabel_path, save_vlabel_path, 256, rep_rate)

    lab_count_path = save_dir + '/slabel_crop'
    lab_filelist = glob(os.path.join(lab_count_path, '*.jpg'))
    print("all_sample_num: ")
    print(len(lab_filelist))
