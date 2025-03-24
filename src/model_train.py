import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from glob import glob
import os
import argparse
from get_data import get_data, read_params
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import tensorflow as tf

def train_model(config_file):
    config = get_data(config_file)
    train = config['model']['trainable']
    if train == True:
        img_size = config['model']['image_size']
        train_set = config['model']['train_path']
        test_set = config['model']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range = config['img_augment']['zoom_range']
        horizontal_flip = config['img_augment']['horizontal_flip']
        vertifal_flip = config['img_augment']['vertical_flip']
        class_mode = config['img_augment']['class_mode']
        batch = config['img_augment']['batch_size']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        epochs = config['model']['epochs']
        model_path = config['model']['sav_dir']

        print(type(batch))

        resnet = VGG16(input_shape=img_size + [3], weights = 'imagenet', include_top = False)
        for p in resnet.layers:
            p.trainable = False
        
        op = Flatten()(resnet.output)
        prediction = Dense(num_cls, activation='softmax')(op)
        mod = Model(inputs = resnet.input, outputs = prediction)
        print(mod.summary())
        img_size = tuple(img_size)
        
        mod.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        train_gen = ImageDataGenerator(rescale = rescale, 
                                       shear_range = shear_range, 
                                       zoom_range = zoom_range, 
                                       horizontal_flip = horizontal_flip, 
                                       vertical_flip = vertifal_flip,
                                       rotation_range = 90)
        test_gen = ImageDataGenerator(rescale = rescale)

        train_set = train_gen.flow_from_directory(train_set,
                                                  target_size = img_size,
                                                  batch_size = batch,
                                                  class_mode = class_mode)
        test_set = test_gen.flow_from_directory(test_set, 
                                                target_size=img_size,
                                                batch_size = batch,
                                                class_mode = class_mode)

        history = mod.fit(train_set,
                          epochs = epochs,
                          validation_data = test_set,
                          steps_per_epoch = len(train_set),
                          validation_steps = len(test_set)) 

        plt.plot(history.history['loss'], label = 'train_loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.plot(history.history['accuracy'], label = 'train_acc')
        plt.plot(history.history['val_accuracy'], label = 'val_acc')
        plt.legend()
        plt.savefig('reports/model_performance.png')

        mod.save(model_path)
        print("Model Saved Successfully....!")

    else:
        print("Model is not trainable")


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    train_model(config_file=passed_args.config)