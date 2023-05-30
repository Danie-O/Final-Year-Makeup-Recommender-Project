import streamlit as st
from PIL import Image

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json

from helper_funcs import predict_skintone, device, predict_skintype
from keras.models import load_model
import torch


def load_checkpoint(feature: str):
    if feature.lower() == "skintone":
        checkpoint = torch.load('final_checkpoint_ic_d161.pth', map_location=torch.device('cpu'))
        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.load_state_dict = (checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']
        for param in model.parameters():
            param.requires_grad = False
        return model
    elif feature.lower() == "skintype":
        model2 = load_model('models/Skintype-Model')
        return model2


#Caching the model for faster loading
@st.cache_resource 
def load_models():
    global skintone_model, skintype_model
    skintone_model = load_checkpoint("skintone")
    skintype_model = load_checkpoint("skintype")
    return skintone_model, skintype_model


# Load pre-trained models
skintone_model, skintype_model = load_models()

# Load json file containing category to class name mappings
with open('./content/skin.json', 'r') as f:
    cat_to_name = json.load(f)

# Load the makeup dataset
makeup = pd.read_csv('./content/makeup.csv')
skintype_classes = {"normal_skin": "normal skin", "oily_skin": "oily skin"}

#Caching the model for faster loading
@st.cache_resource
def predict_skintone_class(img):
    probs, classes = predict_skintone(img, skintone_model.to(device))
    probs_dict = dict(zip(classes, probs))
    class_dict = {0: "dark skin", 1: "olive skin", 2: "porcelain skin"}
    return {class_dict[i]: float(probs_dict[i]) for i in classes}


def view_classify(prob, classes, cat_to_name,  skintype, dataset, k=3):
    classes_dict = {0: "dark skin", 1: "olive skin", 2: "porcelain skin"}
    # Convert the probability and classes data to a Pandas DataFrame for easier manipulation
    data = {'Probability': prob, 'Class': [classes_dict[i] for i in classes]}
    df_prob = pd.DataFrame(data)

    # Display the probability bar chart
    fig = go.Figure(go.Bar(
        x=df_prob['Probability'],
        y=df_prob['Class'],
        orientation='h'
    ))
    fig.update_layout(
        title='Skintone Class Probability',
        yaxis=dict(title='Class'),
        xaxis=dict(title='Probability')
    )
    st.plotly_chart(fig)

    # Select products equivalent to skintone of uploaded image
    skintone = cat_to_name[str(classes[0])]
    rec1 = pd.DataFrame()
    rec1 = rec1.append(dataset[(dataset['SkinType'] == skintype) & (dataset['SkinTone'] == skintone)].head(k))
    # rec1 = dataset.loc[(dataset['SkinTone'] == skintone) & (dataset['SkinType'] == skintype)][:k]

    # Display recommended products based on the predicted skintone and skintype in tabular format
    st.subheader('Product Recommendations')
    rec1_html = rec1.to_html(index=False, escape=False)
    rec1_html = rec1_html.replace('<th>', '<th style="text-align: left;">')
    rec1_html = rec1_html.replace('<td>', '<td style="text-align: left;">')
    st.write(rec1_html, unsafe_allow_html=True)


def show_predict_page():
    st.title("Makeup Recommender System")
    st.write("Try this out to get product recommendations just for you!")
    file = st.file_uploader("Upload your image", type=["jpg", "jpeg"])

    st.write("""##### Select number of products to recommend""")
    k = st.slider("Slide to select number", 1, 10, 3)
    ok = st.button("""Get makeup recommendations!""")

    if file:
        st.write("Image uploaded successfully! Open sidebar to view uploaded image.")
        with st.sidebar:
            # Show the image being classified
            img = Image.open(file)

            # Resize the image while maintaining the aspect ratio
            width, height = img.size
            new_width = 400  # Specify the desired width
            new_height = int(height * (new_width / width))
            resized_image = img.resize((new_width, new_height), resample=Image.LANCZOS)

            # Display the resized image
            st.image(resized_image, caption='Uploaded Image', use_column_width=True)

        if ok:
            st.write("Generating predictions, sit tight!")
            # Generate prediction using loaded model
            skintone_prediction = predict_skintone_class(file)
            st.write("""##### Predicted Skintone: """, list(skintone_prediction.keys())[0])

            skintype = predict_skintype(skintype_model, file)
            skintype_class = skintype_classes[skintype]
            st.write("""##### Predicted Skintype: """, skintype_class)
            probs, classes = predict_skintone(file, skintone_model.to(device))

            result = view_classify(probs, classes, cat_to_name, skintype=skintype_class, dataset=makeup, k=k)
            # st.write(result)
    elif ok:
        st.write("WARNING: You need to upload an image to get recommendations!")

