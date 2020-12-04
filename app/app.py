"""Streamlit web app"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
#from retinaface.pre_trained_models import get_model
#from retinaface.utils import vis_annotations

st.set_option("deprecation.showfileUploaderEncoding", False)


#@st.cache
#def cached_model():
#    m = get_model("resnet50_2020-07-20", max_size=1048, device="cpu")
#    m.eval()
#    return m
#model = cached_model()

st.title("CTG signals")

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="CTG", use_column_width=True)

st.write("")
st.write("EHR")

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])
st.line_chart(chart_data)
