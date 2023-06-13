Makeup-Product-Recommender-System
-------------------------------------

Final year project: A recommender system for content-based recommendation of makeup products based on facial features identified from users' images.

This recommender utilises deep learning models which have been trained using the Convolutional Neural Network(CNN) algorithm, to predict the skintone and skintype class of a user. The models are then used to generate makeup product(foundation) recommendations by matching the predicted classes to products have the same skintone and skintype classes.

A user can get recommendations through a simple web application that allows users to upload their image. The uploaded image is processed and the output returned to the user consists of:

- Makeup product recommendations
- Links to purchase recommended products
- A follow-up email with the recommended products (on request).


Libraries and frameworks used include:
---------------------------------------
- Pytorch
- Tensorflow
- Python(Pandas, numpy, matplotlib and seaborn)
- Python(Flask)
- Python(Streamlit)
- Bootstrap

Usage
----
To run the Flask app locally, navigate to the project directory and run the following command in the terminal:
```
flask run
```

After running the flask app, a Plotly window would pop up showing the classes and their predicted probabilities for the given image.
Close the popup window and access the landing page via: <http://127.0.0.1:5000/>

Note: the Flask app is still under construction so the endpoint above only returns a static landing page.

To run the Streamlit app locally:
- Navigate to the project directory
```
cd Final-Year-Makeup-Recommender-Project/RecommenderSystem
```

- Run the following command in the terminal
```
streamlit run streamlit_app.py
```

A link would be generated and displayed in the terminal, through which the streamlit application can be accessed.



