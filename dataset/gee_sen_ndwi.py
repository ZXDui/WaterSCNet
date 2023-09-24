# Used to format convert Sentinel-2 images downloaded from Google Earth Engine (from Tiff to HDR).

import csv
import math
from sklearn import preprocessing
import gdal
import numpy as np
import os
import cv2
import PIL
from PIL import Image
from glob import *
from osgeo import gdal
from spectral import *


def check_tif(tif_path, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(tif_path)
    if dataset == None:
        print(tif_path + "文件无法打开")

    width = dataset.RasterXSize  # 栅格矩阵的列数
    height = dataset.RasterYSize  # 栅格矩阵的行数
    bands = dataset.RasterCount  # 波段数

    #  获取数据
    if data_width == 0 and data_height == 0:
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)

    geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    proj = dataset.GetProjection()  # 获取投影信息

    print(width)
    print(height)
    print(bands)
    print(geotrans)
    print(proj)
    # return width, height, bands, data, geotrans, proj


def normalization(data):
    nomal_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return nomal_data


def read_tif(tif_path, index):
    dataset = gdal.Open(tif_path)
    if dataset == None:
        print(tif_path + "File cannot be opened!")

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    sen_row = dataset.ReadAsArray(0, 0, width, height)
    data2 = np.array(dataset.GetRasterBand(1).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data3 = np.array(dataset.GetRasterBand(2).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data4 = np.array(dataset.GetRasterBand(3).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data8 = np.array(dataset.GetRasterBand(4).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data5 = np.array(dataset.GetRasterBand(5).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data6 = np.array(dataset.GetRasterBand(6).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data7 = np.array(dataset.GetRasterBand(7).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data8a = np.array(dataset.GetRasterBand(8).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data11 = np.array(dataset.GetRasterBand(9).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data12 = np.array(dataset.GetRasterBand(10).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data1 = np.array(dataset.GetRasterBand(11).ReadAsArray(0, 0, width, height).astype(np.uint16))
    data9 = np.array(dataset.GetRasterBand(12).ReadAsArray(0, 0, width, height).astype(np.uint16))
    # data_ndwi = np.array(dataset.GetRasterBand(13).ReadAsArray(0, 0, width, height).astype(np.uint16))

    sen = np.zeros((height, width, 12))
    sen[:, :, 0] = data2
    sen[:, :, 1] = data3
    sen[:, :, 2] = data4
    sen[:, :, 3] = data8
    sen[:, :, 4] = data5
    sen[:, :, 5] = data6
    sen[:, :, 6] = data7
    sen[:, :, 7] = data8a
    sen[:, :, 8] = data11
    sen[:, :, 9] = data12
    sen[:, :, 10] = data1
    sen[:, :, 11] = data9

    sen_rgb = np.zeros((height, width, 3))
    sen_rgb[:, :, 0] = data4
    sen_rgb[:, :, 1] = data3
    sen_rgb[:, :, 2] = data2

    img = np.zeros((height, width, 3))
    img[:, :, 0] = normalization(data4) * 255
    img[:, :, 1] = normalization(data3) * 255
    img[:, :, 2] = normalization(data2) * 255

    save_hdr_multi = 'Pleace your multi hdr path'
    save_hdr_rgb = 'Pleace your save hdr path'
    save_img = 'Pleace your save img path'

    envi.save_image(save_hdr_multi + str(index) + ".hdr", sen)
    envi.save_image(save_hdr_rgb + str(index) + ".hdr", sen_rgb)
    cv2.imwrite(save_img + str(index) + ".jpg", img)
    print("%d done" % index)


if __name__ == "__main__":
    path = "Pleace your row data path"
    path_list = [
        path + "20220204T022909_20220204T023312_T51RUQ.tif",
        path + "20220224T022659_20220224T023358_T51RUQ.tif",
        path + "20220227T023639_20220227T024821_T51RUQ.tif",
        path + "20220329T023549_20220329T024728_T51RUQ.tif",
        path + "20220420T022551_20220420T023827_T51RUQ.tif",
        path + "20221002T022529_20221002T023813_T51RUQ.tif",
        path + "20221206T023101_20221206T023750_T51RUQ.tif",
        path + "20221214T024119_20221214T024841_T51RUQ.tif",
        path + "20221221T023119_20221221T023818_T51RUQ.tif"
    ]

    index = [40, 41, 42, 43, 44, 45, 46, 47, 48]

    for i in range(len(path_list)):
        read_tif(path_list[i], index[i])

