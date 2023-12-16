import streamlit as st
from PIL import Image
import numpy as np
from fastbook import *
from fastai.vision.widgets import *
# import your model here (e.g., from tensorflow.keras.models import load_model)
learn_inf = load_learner('export.pkl')

# Load your pre-trained model (replace with the path to your model)
# model = load_model('path_to_your_model.h5')

st.title('Bear classifier')

# if st.button('Click Me'):
#     st.write('Streamlit is working!')
# else:
#     st.write('Click the button to test.')


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Click button to classify the image")

    if st.button('Classify'):
        pred,pred_idx,probs = learn_inf.predict(image)
        st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
