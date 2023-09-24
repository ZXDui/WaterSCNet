# Train WaterSCNet, including WaterSCNet-s and WaterSCNet-v.

import os
import sys
import time
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.optimizers import Adam
from wv.model.waterscnet_training_sv import *
from wv.model.my_waterscnet import *
import numpy as np
import tensorflow as tf
import random as python_random
np.random.seed(42)
python_random.seed(42)
tf.set_random_seed(42)


def train_segANDvec(log_dir, dataset_dir, max_patience):
    input_channls = 12
    inputs_size = [256, 256, input_channls]

    model = v2c_sANDv(inputs_size)

    with open(os.path.join(dataset_dir, "train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(dataset_dir, "val.txt"), "r") as f:
        val_lines = f.readlines()

    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=max_patience, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)
    loss_history = LossHistory(log_dir)

    if True:
        lr = 1e-4
        Epoch = 30000
        Batch_size = 4

        my_loss = {'output_1': 'binary_crossentropy', 'output_2': 'binary_crossentropy'}
        my_loss_weights = {'output_1': 0.5, 'output_2': 0.5}
        my_metrics = {'output_1': ['binary_accuracy'], 'output_2': ['binary_accuracy']}

        model.compile(loss=my_loss, loss_weights=my_loss_weights, optimizer=Adam(lr=lr), metrics=my_metrics)

        gen = Generator(Batch_size, train_lines, inputs_size, dataset_dir).sv_generate()
        gen_val = Generator(Batch_size, val_lines, inputs_size, dataset_dir).sv_generate()

        epoch_size = len(train_lines) // Batch_size
        epoch_size_val = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("Please expand the dataset")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines),
                                                                                   len(val_lines), Batch_size))

        start_time = time.process_time()
        model.fit_generator(gen,
                            steps_per_epoch=epoch_size,
                            validation_data=gen_val,
                            validation_steps=epoch_size_val,
                            epochs=Epoch,
                            callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history])

        traintime = time.process_time() - start_time
        return traintime


if __name__ == "__main__":
    log_path = "Pleace your log path"
    dataset_path = "Pleace your dataset path"

    m_patience = 100
    time_segANDvec = train_segANDvec(log_path, dataset_path, m_patience)
    print("Time for segANDvec:", time_segANDvec)
