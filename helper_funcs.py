import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
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

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array.
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # tensor.numpy().transpose(1, 2, 0)
    preprocess = transforms.Compose([
        transforms.Resize(256),
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

import plotly.graph_objects as go
import pandas as pd

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
