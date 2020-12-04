"""Streamlit web app"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from typing import List

st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache
def cached_model(x_valid: List):
    """
    Load logreg model from disk 
    """
    model = pickle.load(open('model.txt', 'rb'))
    y_pred = model.predict(x_valid)
    # result = model.score(x_valid, y_valid)
    # print(result)
    return y_pred

#result = cached_model(x_valid)

st.title("Cardiotocography signals")

uploaded_file = st.file_uploader("Load patient's cardiotocography (CTG)...", type="png")
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="CTG", use_column_width=True)

st.write("")
st.write("EHR")

st.write("EGA week: 39")
st.write("Parity: 1")
st.write("Age: 30")
