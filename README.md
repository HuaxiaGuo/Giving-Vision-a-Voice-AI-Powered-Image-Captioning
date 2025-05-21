Giving Vision a Voice- AI-Powered Image Captioning  

![image](https://github.com/user-attachments/assets/069e4734-9d23-47e9-8fee-94a0a11d6cec)

Description:

Image captioning is a task that combines computer vision and natural language processing, aiming to automatically generate descriptive sentences for images. 
It plays an important role in making visual content more accessible and searchable. It typically involves two main components: a visual feature extractor (CNNs) and a language model (sequence-based models) to generate captions.

Requirements:

import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

Objectives:

In this project, we focus on training image captioning models that use CNNs (EfficientNet-B0) to extract image features, 
and an LSTM-based decoder to generate captions. We evaluate model performance using BLEU-1 to BLEU-4 metrics and compare the different CNN encoders. 
Our goal is to analyze the quality of generated captions and understand the performance and limitations of each model.

Parameter Chosen in Sampling:

Image encoder: Using a pre trained EfficientNet-B0 model, the original classification layer is removed and replaced with a custom classifier that includes a linear layer, 
ReLU activation, and Dropout, with an output dimension of 512.
Text decoder:
Embedding layer: Map the token to a 512 dimensional vector.
Attention mechanism (Attention class): Calculate the attention weights of image features and decoder hidden states to generate context vectors.
LSTM: Receive the concatenation of embedding vectors and context vectors, and output the hidden state.
Fully connected layer: Maps the output of LSTM to a probability distribution of vocabulary size.
Dropout: Used to prevent overfitting.
Training process: Using cross entropy loss function and Adam optimizer, iteratively train data to optimize model parameters with 100 epochs.
Evaluation metric: Use BLEU scores (BLEU-1 to BLEU-4) to assess the accuracy of generated descriptions.

Comments:

Our codebase for the model and discrete diffusion builds on CNN, LSTM and attention.

Citation:

https://daniel.lasiman.com/post/image-captioning/
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
https://towardsdatascience.com/how-to-build-an-image-captioning-model-in-pytorch-29b9d8fe2f8c


