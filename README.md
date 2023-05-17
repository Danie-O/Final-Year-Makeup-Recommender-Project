Makeup-Product-Recommender-System
-------------------------------------

Final year project: A recommender system for content-based recommendation of makeup products based on facial features identified from users' images.

This recommender utilises a deep learning model, trained using Convolutional Neural Networks(CNNs), to predict the skintone class of a user.
The recommender is then utilised for product recommendation by matching user skintones to products based on the skintone class which they fall under.
A user can get recommendations through a simple web application that allows users to upload their image. The uploaded image is processed and the output returned to the user consists of:

- Skintone-based makeup product recommendations
- Links to purchase recommended products
- A follow-up email containing the recommended products is also sent to the user(on request).

Key libraries and frameworks used include:

- Pytorch
- Tensorflow
- Python(Pandas, matplotlib and seaborn)
- Python(Flask)
- Javascript (ReactJS)

To run the Flask app,
[ flask run ]

You can access the landing page via:
After running the flask app, a Plotly window would pop up showing the classes and their predicted probabilities for the given image.
Close the popup window to get redirected to a page showing product recommendations with embedded links to shop for the products.
If you are not redirected, access the recommendations via: <http://127.0.0.1:4216/>
