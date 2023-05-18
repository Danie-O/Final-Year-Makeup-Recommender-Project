import streamlit as st
from PIL import Image

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json

from helper_funcs import load_checkpoint, recommend_products, device


def load_model():
    model, class_to_idx = load_checkpoint('checkpoint_ic_d161.pth')
    return [model, class_to_idx]

# Load pre-trained model
model = load_model()[0]

# Load json file containing category to class name mappings
with open('./content/skin.json', 'r') as f:
    cat_to_name = json.load(f)

# load the makeup dataset
rec = pd.read_csv('./content/makeup.csv')


def predict_class(img):
    probs, classes = recommend_products(img, model.to(device))
    probs_dict = dict(zip(classes, probs))
    class_dict = {0: "Dark skin", 1: "Olive skin", 2: "Porcelain skin"}
    return {class_dict[i]: float(probs_dict[i]) for i in classes}


def view_classify(prob, classes, cat_to_name, rec, k=3):
    classes_dict = {0: "Dark skin", 1: "Olive skin", 2: "Porcelain skin"}

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

    # # Display the probability bar chart
    # st.subheader('Skintone Class Probability')
    # st.bar_chart(df_prob.set_index('Class'))

    # Select products equivalent to skintone of uploaded image
    skintype = cat_to_name[str(classes[0])]
    rec1 = rec.loc[rec['SkinTone'] == skintype][:k]

    # Display recommended products based on the model's predictions in tabular format
    st.subheader('Product Recommendations')
    rec1_html = rec1.to_html(escape=False)
    rec1_html = rec1_html.replace('<th>', '<th style="text-align: left;">')
    rec1_html = rec1_html.replace('<td>', '<td style="text-align: left;">')
    st.write(rec1_html, unsafe_allow_html=True)


def show_predict_page():
    st.title("Makeup Recommender System")
    st.write("Try this out to get product recommendations just for you!")
    file = st.file_uploader("Upload your image")

    k = st.slider("Select number of products to recommend", 1, 20, 3)
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
            prediction = predict_class(file)
            st.write("""#### Predicted class: """, list(prediction.keys())[0])
            st.write(prediction)
            # st.write()

            probs, classes = recommend_products(file, model.to(device))
            result = view_classify(probs, classes, cat_to_name, rec, k=k)
            # st.write(result)

