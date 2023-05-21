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

def load_checkpoint(feature: str):
    if feature.lower() == "skintone":
        checkpoint = torch.load('final_checkpoint_ic_d161.pth', map_location=torch.device('cpu'))
        # checkpoint['input_size'] = 25088
        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.load_state_dict = (checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']

        for param in model.parameters():
            param.requires_grad = False

        return model, checkpoint['class_to_idx']
    elif feature.lower() == "skintype":
        model = models.load_model('models/Skintype-Model')
        return model
        

""" HELPER FUNCTIONS FOR SKINTONE CLASSES PREPROCESSING & PREDICTION."""

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array.
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

    # def tensor_to_image(tensor):
    #     """Convert the tensor output from 'process_image' to a numpy array."""
    #     tensor = tensor.cpu().detach().numpy()
        
    #     # Scale the values to [0, 255]
    #     tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    #     tensor = 255 * tensor
        
    #     # Convert the numpy array to an image array
    #     image_array = tensor.transpose((1, 2, 0)).astype(np.uint8)
        
    #     return image_array
    # pil_image = tensor_to_image(image)
    # return pil_image
    return image

def recommend_products(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''

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
rec = pd.read_csv('./content/makeup.csv')

def view_classify(img_path, prob, classes, cat_to_name, rec, k=3):
    ''' 
        Function for viewing an image, probability of it belonging to the classes.
        Also display recommended products based on class with highest probability
    '''
    image = Image.open(img_path)

    # Show the image being classified
    fig, (ax1, ax2) = plt.subplots(figsize=(13, 5), ncols=2)
    ax1.set_title('Uploaded Image')
    ax1.imshow(image)
    ax1.axis('off')

    classes_dict = {0: "Dark skin", 1: "Olive skin", 2: "Porcelain skin"}
    # Show the prediction probability outputted by model for the image
    y_pos = np.arange(len(prob))
    ax2.barh(y_pos, prob, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([classes_dict[i] for i in classes])
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_title('Skintone Class Probability')

    # Set spacing between subplots
    plt.subplots_adjust(wspace=0.5)
    plt.show()

    # Select products equivalent to skintone of uploaded image
    skintype = cat_to_name[str(classes[0])]
    rec1 = rec.loc[rec['SkinTone'] == skintype][:k]

    # Display recommended products based on the model's predictions in tabular format
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(rec1.columns),
               fill_color='paleturquoise',
               align='left'),
    cells=dict(values=[rec1.Foundation, rec1.HEXColor, rec1.SkinTone,rec1.Company, rec1.ProductURL, rec1.Price, rec1.Image, rec1.VideoTutorial],
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
    # print(rec1.to_dict())

    def make_clickable(val):
        # target _blank to open new window
        return '<a href="{}" target="_blank">{}</a>'.format(val, val)

    rec1['ProductURL'] = rec1['ProductURL'].apply(make_clickable)
    # rec1.style.format({'ProductURL': make_clickable})

    return (rec1)


""" HELPER FUNCTIONS FOR SKINTYPE CLASSES PREPROCESSING AND PREDICTIONS."""

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(file, img_shape=224):
    """ Reads in an image from filename, turns it into a tensor and reshapes into (224,224,3). """
   
    # Read in the image and decode it into a tensor
    # image_bytes = file.read()
    image = Image.open(file)
    image = image.resize((img_shape, img_shape))
    # Convert the PIL Image to a TensorFlow tensor
    image = tf.keras.preprocessing.image.img_to_array(image)
    # Rescale the image (get all values between 0 and 1)
    image = image / 255.0
    return image


# Setup paths to our data directories
train_dir = "Skintype/skintype-data/train/"
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))


def predict_skintype(model, file, class_names=class_names):
   """
   Imports an image located at filename, makes a prediction with model
   and plots the image with the predicted class as the title.
   """
   # Import the target image and preprocess it
   img = load_and_prep_image(file)

   # Make a prediction
   pred = model.predict(tf.expand_dims(img, axis=0))

   if len(pred[0]) > 1:
      pred_class = class_names[tf.argmax(pred[0])]
   else:
    pred_class = class_names[int(tf.round(pred[0]))]
    print('Prediction Probabilities : ', pred[0])

    print(pred_class)
    return pred_class

from PIL import Image

image = Image.open("content/valid/darkskin/download.jpg")
skintype_model = load_checkpoint("skintype")
print(predict_skintype(skintype_model, "content/valid/darkskin/download.jpg"))
