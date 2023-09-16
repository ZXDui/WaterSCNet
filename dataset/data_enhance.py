import os
import cv2
import gdal
import numpy as np
from glob import *
from spectral import *


def read_hdr(hdr_path):
    img = open_image(hdr_path)

    if img == None:
        print(hdr_path + "File cannot be opened!")
    return img


def hdr126_enhance(hdr_path, save_hdr_path):
    count_path = save_hdr_path
    filelist = glob(os.path.join(count_path, '*.hdr'))
    new_name = len(filelist) + 1

    img = read_hdr(hdr_path)
    a_img = img.load()
    n_img = np.array(a_img)

    envi.save_image(save_hdr_path + "/%d.hdr" % new_name, n_img)
    new_name += 1

    im_data_hor = np.flip(n_img, axis=0)
    envi.save_image(save_hdr_path + "/%d.hdr" % new_name, im_data_hor)
    new_name += 1
    # im_data_hor2 = np.flip(im_data_hor, axis=0)
    # envi.save_image(save_hdr_path + "/%d.hdr" % new_name, im_data_hor2)
    # new_name += 1

    im_data_vec = np.flip(n_img, axis=1)
    envi.save_image(save_hdr_path + "/%d.hdr" % new_name, im_data_vec)


def hdrrgb_enhance(hdr_path, save_hdr_path):
    count_path = save_hdr_path
    filelist = glob(os.path.join(count_path, '*.hdr'))
    new_name = len(filelist) + 1

    img = read_hdr(hdr_path)
    a_img = img.load()
    n_img = np.array(a_img)

    envi.save_image(save_hdr_path + "/%d.hdr" % new_name, n_img)
    new_name += 1

    im_data_hor = np.flip(n_img, axis=0)
    envi.save_image(save_hdr_path + "/%d.hdr" % new_name, im_data_hor)
    new_name += 1
    # im_data_hor2 = np.flip(im_data_hor, axis=0)
    # envi.save_image(save_hdr_path + "/%d.hdr" % new_name, im_data_hor2)
    # new_name += 1

    im_data_vec = np.flip(n_img, axis=1)
    envi.save_image(save_hdr_path + "/%d.hdr" % new_name, im_data_vec)


def orin_enhance(o_path, save_o_path):
    count_path = save_o_path
    filelist = glob(os.path.join(count_path, '*.jpg'))
    new_name = len(filelist) + 1

    label = cv2.imread(o_path)

    cv2.imwrite(save_o_path + "/%d.jpg" % new_name, label)
    new_name += 1

    im_data_hor = cv2.flip(label, 0)
    cv2.imwrite(save_o_path + "/%d.jpg" % new_name, im_data_hor)
    new_name += 1
    # im_data_hor2 = cv2.flip(im_data_hor, 0)
    # cv2.imwrite(save_o_path + "/%d.jpg" % new_name, im_data_hor2)
    # new_name += 1

    im_data_vec = cv2.flip(label, 1)
    cv2.imwrite(save_o_path + "/%d.jpg" % new_name, im_data_vec)


def slabel_enhance(o_path, save_o_path):
    count_path = save_o_path
    filelist = glob(os.path.join(count_path, '*.jpg'))
    new_name = len(filelist) + 1

    label = cv2.imread(o_path)

    cv2.imwrite(save_o_path + "/%d.jpg" % new_name, label)
    new_name += 1

    im_data_hor = cv2.flip(label, 0)
    cv2.imwrite(save_o_path + "/%d.jpg" % new_name, im_data_hor)
    new_name += 1
    # im_data_hor2 = cv2.flip(im_data_hor, 0)
    # cv2.imwrite(save_o_path + "/%d.jpg" % new_name, im_data_hor2)
    # new_name += 1

    im_data_vec = cv2.flip(label, 1)
    cv2.imwrite(save_o_path + "/%d.jpg" % new_name, im_data_vec)
    new_name += 1


def vlabel_enhance(o_path, save_o_path):
    count_path = save_o_path
    filelist = glob(os.path.join(count_path, '*.jpg'))
    new_name = len(filelist) + 1

    label = cv2.imread(o_path)

    cv2.imwrite(save_o_path + "/%d.jpg" % new_name, label)
    new_name += 1

    im_data_hor = cv2.flip(label, 0)
    cv2.imwrite(save_o_path + "/%d.jpg" % new_name, im_data_hor)
    new_name += 1
    # im_data_hor2 = cv2.flip(im_data_hor, 0)
    # cv2.imwrite(save_o_path + "/%d.jpg" % new_name, im_data_hor2)
    # new_name += 1

    im_data_vec = cv2.flip(label, 1)
    cv2.imwrite(save_o_path + "/%d.jpg" % new_name, im_data_vec)
    new_name += 1


if __name__ == '__main__':
    root_dir = 'Pleace your root path'
    save_dir = 'Pleace your save path'

    path_list = glob(os.path.join('Pleace your crop path', '*.hdr'))
    for i in range(1, len(path_list) + 1):
        hdr126_path = root_dir + '/hdr126_crop/' + str(i) + '.hdr'
        save_hdr126_path = save_dir + '/hdr126_en'

        hdrrgb_path = root_dir + '/hdrrgb_crop/' + str(i) + '.hdr'
        save_hdrrgb_path = save_dir + '/hdrrgb_en'

        o_path = root_dir + '/img_crop/' + str(i) + '.jpg'
        save_o_path = save_dir + '/img_en'

        slabel_path = root_dir + '/slabel_crop/' + str(i) + '.jpg'
        save_slabel_path = save_dir + '/slabel_en'

        vlabel_path = root_dir + '/vlabel_crop/' + str(i) + '.jpg'
        save_vlabel_path = save_dir + '/vlabel_en'

        hdr126_enhance(hdr126_path, save_hdr126_path)
        hdrrgb_enhance(hdrrgb_path, save_hdrrgb_path)
        orin_enhance(o_path, save_o_path)
        slabel_enhance(slabel_path, save_slabel_path)
        vlabel_enhance(vlabel_path, save_vlabel_path)
