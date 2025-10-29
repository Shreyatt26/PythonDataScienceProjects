## Goal: Classify digits (0-9) from MNIST dataset
## Learn: TensorFlow basics, building a sequential model, compiling, training, and evaluating
## Extensions: add dropout, normalization, visualize intermediate layers with tf.keras.utils.plot_model

## load libraries
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib as plt

## load training data
train = pd.read_csv("C:/Users/Shreya Tanguturi/Desktop/PythonPersonalProjects/llm-classification-finetuning/train.csv")
