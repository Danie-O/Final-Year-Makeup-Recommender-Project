import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import PIL
from PIL import Image
from collections import OrderedDict

import torch 
from torch.autograd import Variable
import torchvision 
from torchvision import datasets, models, transforms 
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from keras import models
import tensorflow as tf
import io
import pathlib 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


""" HELPER FUNCTIONS FOR SKINTONE PREPROCESSING & PREDICTION."""

def process_image(image):
    ''' Scale, crop, and normalize a PIL image for a PyTorch model,
        return a Numpy array.
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # tensor.numpy().transpose(1, 2, 0)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

def predict_skintone(image_path, model, topk=3):
    ''' Predict the skintone class of an image using a trained deep learning model.'''

    img = Image.open(image_path)
    img = process_image(img)
    
    # Convert 2D image to 1D vectorcle
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)

    model.eval()
    inputs = Variable(img).to(device)
    # model.classifier[0] = nn.Linear(2208, 120, bias = True)

    logits = model.forward(inputs)

    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)

    return (e.data.numpy().squeeze().tolist() for e in topk)


# load the makeup dataset
rec_data = pd.read_csv('./content/makeup.csv')

def view_classify(img_path, prob, classes, cat_to_name, rec_data, k=3):
    ''' 
        Function for viewing an image, probability of it belonging to the classes.
        Also display recommended products based on class with highest probability
    '''
    image = Image.open(img_path)

    # Select products equivalent to skintone of uploaded image
    skintone = cat_to_name[str(classes[0])]
    rec = rec_data.loc[(rec_data['SkinTone'] == skintone) & (rec_data['SkinType'] == predict_skintype(image))][:k]

    # Display recommended products based on the model's predictions in tabular format
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(rec.columns),
               fill_color='paleturquoise',
               align='left'),
    cells=dict(values=[rec.Foundation, rec.HEXColor, rec.SkinTone,rec.Company, rec.ProductURL, rec.Price, rec.Image, rec.VideoTutorial],
              fill_color='lavender',
              align='left'))
      ])
    
    fig.update_layout(
        title={
            'text': "Product Recommendations",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    fig.show()

    def make_clickable(val):
        # target _blank to open new window
        return '<a href="{}" target="_blank">{}</a>'.format(val, val)

    rec['ProductURL'] = rec['ProductURL'].apply(make_clickable)
    return (rec)


""" HELPER FUNCTIONS FOR SKINTYPE PREPROCESSING AND PREDICTIONS."""
def load_and_prep_image(file, img_shape=224):
    """ Read in an image from an UploadedFile object, turns it into a tensor, and reshapes it into (224, 224, 3). """
    # Open the image using PIL
    img = Image.open(file)

    # Resize the image and convert the image to a numpy array
    img = img.resize((img_shape, img_shape))
    img = np.array(img)

    # Rescale the image (get all values between 0 and 1)
    img = img / 255.0

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)

    return img

# Setup paths to our data directories
train_dir = "Skintype/skintype-data/train/"
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print((class_names))

def predict_skintype(model, file, class_names=class_names):
    """Import an image located at filename, make a prediction with model and return the predicted class as the title."""
    # Import the target image, preprocess it and make prediction
    img = load_and_prep_image(file)
    pred = model.predict(img)

    if len(pred[0]) > 1:
      pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]
        # print('Prediction Probabilities : ', len(pred[0]))
    return pred_class
