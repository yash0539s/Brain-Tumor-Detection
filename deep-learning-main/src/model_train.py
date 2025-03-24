import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import argparse
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from get_data import get_data  # Assuming `get_data` correctly reads `params.yaml`

def train_model(config_file):
    config = get_data(config_file)

    train = config['model'].get('trainable', False)
    if train:
        img_size = tuple(config['model']['image_size'])
        train_set_path = config['model']['train_path']
        test_set_path = config['model']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment'].get('shear_range', 0.0)
        zoom_range = config['img_augment'].get('zoom_range', 0.0)
        horizontal_flip = config['img_augment'].get('horizontal_flip', False)
        vertical_flip = config['img_augment'].get('vertical_flip', False)
        class_mode = config['img_augment'].get('class_mode', 'categorical')
        batch_size = int(config['img_augment'].get('batch_size', 32))
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        epochs = int(config['model']['epochs'])
        model_path = config['model'].get('save_dir', 'models/trained.h5')


        print(f"Batch size type: {type(batch_size)}, value: {batch_size}")

        # Load VGG16 model without top layers
        base_model = VGG16(input_shape=img_size + (3,), weights='imagenet', include_top=False)
        for layer in base_model.layers:
            layer.trainable = False  # Freeze pre-trained layers

        # Add new layers
        x = Flatten()(base_model.output)
        output_layer = Dense(num_cls, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output_layer)

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        print(model.summary())

        # Data augmentation
        train_gen = ImageDataGenerator(rescale=rescale, shear_range=shear_range, zoom_range=zoom_range,
                                       horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                       rotation_range=90)
        test_gen = ImageDataGenerator(rescale=rescale)

        # Load datasets
        train_set = train_gen.flow_from_directory(train_set_path, target_size=img_size,
                                                  batch_size=batch_size, class_mode=class_mode)
        test_set = test_gen.flow_from_directory(test_set_path, target_size=img_size,
                                                batch_size=batch_size, class_mode=class_mode)

        # Train model
        history = model.fit(train_set, epochs=epochs, validation_data=test_set,
                            steps_per_epoch=len(train_set), validation_steps=len(test_set))

        # Plot training history
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
        plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
        plt.legend()
        plt.savefig('reports/model_performance.png')
        plt.show()

        # Save model
        model.save(model_path)
        print("Model saved successfully!")

    else:
        print("Model is not trainable")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    args = parser.parse_args()
    train_model(config_file=args.config)
