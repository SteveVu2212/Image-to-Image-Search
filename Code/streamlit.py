import sqlite3
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt


st.title('Project: Image-to-Image Search')

st.header('I. Main steps')

st.subheader('1. Load and read data')

st.write('Crawling data from the website of Shopee')

from PIL import Image

img = Image.open('/Image-to-Image-Search/images/trainingdata.png')
st.image(img, width =500, caption = 'number of classified data')

st.subheader('2. Indexing data')

st.write('Using pre-trained model VGG16 to make image features from the penultimate layer')
img1 = Image.open('/Image-to-Image-Search/images/VGG16.jpeg')
st.image(img1, width =1000, caption = 'Pre-trained model: VGG16')

st.write('Image features here')
img2 = Image.open('/Image-to-Image-Search/Image embeddings.png')
st.image(img2, width =1000, caption = 'Image embeddings')

st.write('Applying Annoy to make index features')

st.subheader('3. Using embeddings to search through images')

st.write('Input here')
img3 = Image.open('/Image-to-Image-Search/input_train.png')
st.image(img3, width =300, caption = 'Input image')

st.write('Output here')
img4 = Image.open('/Image-to-Image-Search/Output.png')
st.image(img4, width =1000, caption = 'Output images')

st.header('II. Demo')
st.subheader('1. Input here')
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
st.subheader('2. Output')




