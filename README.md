# Image-Classification-with-TensorFlow
Project 2 of Introduction to Machine Learning with TensorFlow- Udacity
This is my second project for Udacity's "Introduction to Machine Learning with TensorFlow Nanodegree" Program to learn about Deep Learning using TensorFlow.

Tools used in this project:

- Python
- Numpy
- Panda
- Matplotlib
- Tensonflow
- Tensorflow Hub
- Keras
- Anaconda (Jypyter Notebook)
- GPU required

## Files description  
- 1604018417.h5 Keras saved model
- predict.py commmand line application
- Project_Image_Classifier_Project.ipynb jupyter notebook contains codes for entire project

## Installation
- TensorFlow- 2.1.0  
`{ !pip install tensorflow==2.1.0 --user }`  
- TensorFlow Dataset  
`{ %pip --no-cache-dir install tfds-nightly --user }`
- TensorFlow Hub  
`{ pip install -q -U tensorflow_hub }`

The dataset used is [Oxford 102 Category Flower Dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102).

## How to run predict.py (Command line application)
Basic usage:
`{ $ python predict.py ./test_images/orchid.jpg 1604018417.h5 }`
Options:  
- Return the top 3 most likely classes:  
`{ $ python predict.py ./test_images/orchid.jpg 1604018417.h5 --top_k 3 }`
- Use a label_map.json file to map labels to flower names:  
`{ $ python predict.py ./test_images/orchid.jpg 1604018417.h5 --category_names label_map.json }`

## License

This project belongs to Udacity Nanodegree, all the copyrights belog to Udacity.
