from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import argparse
from get_data import get_data
from keras_preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def m_evaluate(config_file):
    config = get_data(config_file)
    batch = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    te_set = config['model']['test_path']
    model = load_model('models/trained.h5')

    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = test_gen.flow_from_directory(te_set,
                                            target_size=(255, 255),  # ✅ Fix: Ensure correct input size
                                            batch_size=batch,
                                            class_mode=class_mode
                                            )
    label_map = test_set.class_indices
    print(label_map)

    Y_pred = model.predict(test_set, steps=len(test_set))  # ✅ Ensure correct steps
    y_pred = np.argmax(Y_pred, axis=1)

    print("Confusion Matrix")
    sns.heatmap(confusion_matrix(test_set.classes, y_pred), annot=True)
    plt.xlabel('Actual values: 0-Bulbasaur, 1-Charmander, 2-Squirtle, 3-Tauros')
    plt.ylabel('Predicted values: 0-Bulbasaur, 1-Charmander, 2-Squirtle, 3-Tauros')
    plt.savefig('reports/Confusion_Matrix')

    print("Classification Report")
    target_names = ['Bulbasaur', 'Charmander', 'Squirtle', 'Tauros']
    df = pd.DataFrame(classification_report(test_set.classes, y_pred, target_names=target_names, output_dict=True)).T
    df['support'] = df.support.apply(int)
    df.to_csv('reports/classification_report')
    print('Classification Report and Confusion Matrix saved in reports folder.')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml')
    passed_args = args_parser.parse_args()
    m_evaluate(config_file=passed_args.config)
