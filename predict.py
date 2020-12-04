#pip install tensorflow==2.1.0 --user
#pip --no-cache-dir install tfds-nightly --user
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
# TODO: Make all other necessary imports.
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from PIL import Image
import argparse

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

tfds.disable_progress_bar()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset, dataset_info = tfds.load("oxford_flowers102", as_supervised=True, with_info=True, download=True)

#Options
parser = argparse.ArgumentParser(description = "Image Classfication")
parser.add_argument('imagefiles', default='./test_images/orange_dahlia.jpg', help='Path to image', type=str)
parser.add_argument('saved_model', help='Add name of the saved model', type=str)
parser.add_argument('--top_k', default=5, help='Show top k possible name of the classes for flowers', type=int)
parser.add_argument('--category_name', default='label_map.json', help='Mapping of classes to names',type=str)

args = parser.parse_args()

image_path = args.imagefiles
saved_keras_model_filepath = args.saved_model
k = args.top_k
ctgry_name = args.category_name

#Reloading saved model
reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer': hub.KerasLayer})
#print(reloaded_keras_model.summary())

with open(ctgry_name, 'r') as f:
    class_names = json.load(f)

# Create the process_image function
IMG_SIZE = 224

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))/255
    image = image.numpy()
    return (image)

# Create the predict function
def predict_usd(image_path, model, top_k):    
    im = Image.open(image_path)
    image_arr = np.asarray(im)
    image_arr = process_image(image_arr)
    predictions = model.predict(np.expand_dims(image_arr, axis=0))
    top_k_values, indices = tf.nn.top_k(predictions, k = top_k)
    top_value, index = tf.nn.top_k(predictions, k = 1)
    incre = [value_class+1 for value_class in indices.numpy()[0]]
    top_classes = [x.astype('str') for x in incre]
    return (top_k_values.numpy()[0], top_classes, index)


probs, classes, best_index = predict_usd(image_path, reloaded_keras_model, k)
cls_nme = [class_names[x] for x in classes]
best_index = best_index.numpy()[0].item(0)
best_name = class_names[str(best_index)]
print('Predicted category name: ',best_name,' ','with predicted probability of',np.around(np.amax(probs), 4))
print('\nThe list of all top k probabilities')
print(np.around(probs, 4))
print(classes)
print(cls_nme)