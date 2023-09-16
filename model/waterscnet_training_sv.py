import os
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf
from keras import backend as K
from spectral import *
from PIL import Image
from random import shuffle


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class Generator(object):
    def __init__(self, batch_size, train_lines, image_size, dataset_path):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = 1
        self.dataset_path = dataset_path

    def s_generate(self):
        i = 0
        length = len(self.train_lines)
        inputs = []
        targets = []
        while True:
            if i == 0:
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            name = annotation_line.split()[0]

            hdr = open_image(os.path.join(os.path.join(self.dataset_path, "hdr"), name + ".hdr"))
            a_img = hdr.load()
            slabel = cv2.imread(os.path.join(os.path.join(self.dataset_path, "slabel"), name + ".jpg"), 0)
            label = np.array(slabel)
            label = label / 255
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

            inputs.append(np.array(a_img))
            seg_labels = label.reshape((int(self.image_size[1]) * int(self.image_size[0]), self.num_classes))
            targets.append(seg_labels)
            i = (i + 1) % length
            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []
                targets = []
                yield tmp_inp, tmp_targets

    def v_generate(self):
        i = 0
        length = len(self.train_lines)
        inputs = []
        targets = []
        while True:
            if i == 0:
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            name = annotation_line.split()[0]

            hdr = open_image(os.path.join(os.path.join(self.dataset_path, "hdr"), name + ".hdr"))
            a_img = hdr.load()
            vlabel = cv2.imread(os.path.join(os.path.join(self.dataset_path, "vlabel"), name + ".jpg"), 0)
            label = np.array(vlabel)
            label = label / 255
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

            inputs.append(np.array(a_img))
            seg_labels = label.reshape((int(self.image_size[1]) * int(self.image_size[0]), self.num_classes))
            targets.append(seg_labels)
            i = (i + 1) % length
            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []
                targets = []
                yield tmp_inp, tmp_targets

    def sv_generate(self):
        i = 0
        length = len(self.train_lines)
        inputs = []
        # targets = []
        targets_seg = []
        targets_vec = []
        while True:
            if i == 0:
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            name = annotation_line.split()[0]

            hdr = open_image(os.path.join(os.path.join(self.dataset_path, "hdr"), name + ".hdr"))
            a_img = hdr.load()
            slabel = cv2.imread(os.path.join(os.path.join(self.dataset_path, "slabel"), name + ".jpg"), 0)
            vlabel = cv2.imread(os.path.join(os.path.join(self.dataset_path, "vlabel"), name + ".jpg"), 0)
            labels = np.array(slabel)
            labelv = np.array(vlabel)
            labels = labels / 255
            labelv = labelv / 255
            labels[labels > 0.5] = 1
            labels[labels <= 0.5] = 0
            labelv[labelv > 0.5] = 1
            labelv[labelv <= 0.5] = 0

            inputs.append(np.array(a_img))
            seg_labels = labels.reshape((int(self.image_size[1]) * int(self.image_size[0]), self.num_classes))
            vec_labels = labelv.reshape((int(self.image_size[1]) * int(self.image_size[0]), self.num_classes))
            targets_seg.append(seg_labels)
            targets_vec.append(vec_labels)

            i = (i + 1) % length
            if len(inputs) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets_seg = np.array(targets_seg)
                tmp_targets_vec = np.array(targets_vec)
                inputs = []
                targets_seg = []
                targets_vec = []
                yield tmp_inp, [tmp_targets_seg, tmp_targets_vec]  # [a_img], [seg_labels, vec_labels]


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses = []
        self.val_loss = []

        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_loss')))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='Train loss')
        plt.plot(iters, self.val_loss, 'blue', linewidth=2, label='Val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

        root_dir = "Pleace your root path for plotting"
        np.save(root_dir + "iters.npy", np.array(iters))
        np.save(root_dir + "train_loss.npy", np.array(self.losses))
        np.save(root_dir + "val_loss.npy", np.array(self.val_loss))

