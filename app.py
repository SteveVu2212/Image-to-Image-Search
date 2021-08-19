import streamlit as st

import sqlite3
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import json
import time

import h5py
import numpy as np

from annoy import AnnoyIndex
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
# from tensorflow.compat.v1.keras.losses import cosine_proximity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_features(features_filename, mapping_filename):

    print ("Loading features...")
    images_features = np.load('%s.npy' % features_filename)
    with open('%s.json' % mapping_filename) as f:
        index_str = json.load(f)
        file_index = {int(k): str(v) for k, v in index_str.items()}
    return images_features, file_index


# path1 = '/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Code'

path1 = '/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/Code'

images_features, file_index = load_features(path1, path1)


def get_class_weights_from_vgg(save_weights=False, filename='class_weights'):

    model_weights_path = os.path.join(os.environ.get('HOME'),
                                      '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    weights_file = h5py.File(model_weights_path, 'r')
    weights_file.get('predictions').get('predictions_W_1:0')
    final_weights = weights_file.get('predictions').get('predictions_W_1:0')

    class_weights = np.array(final_weights)[:]
    weights_file.close()
    if save_weights:
        np.save('%s.npy' % filename, class_weights)
    return class_weights



def get_weighted_features(class_index, images_features):

    class_weights = get_class_weights_from_vgg()
    target_class_weights = class_weights[:, class_index]
    weighted = images_features * target_class_weights
    return weighted



def index_features(features, n_trees=1000, dims=4096, is_dict=False):

    print ("Indexing features...")
    feature_index = AnnoyIndex(dims, metric='angular')
    for i, row in enumerate(features):
        vec = row
        if is_dict:
            vec = features[row]
        feature_index.add_item(i, vec)
    feature_index.build(n_trees)
    return feature_index


# path2 = '/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Report'

path2 = '/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259'

weighted_features, file_index = load_features(path2, path2)


weighted_index = index_features(weighted_features)


# input_train = '/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Data/TrainingData/class-điện thoại/465.jfif'

input_train = '/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/Data/TrainingData/class-cặp sách/43.jfif'

def get_index(input_image, file_mapping):
    for index, file in file_mapping.items():
        if file == input_image:
            return index
    raise ValueError("Image %s not indexed" % input_image)


search_key = get_index(input_train, file_index)


def search_index_by_key(key, feature_index, item_mapping, top_n=10):

    distances = feature_index.get_nns_by_item(key, top_n, include_distances=True)
    return [[a, item_mapping[a], distances[1][i]] for i, a in enumerate(distances[0])]


weighted_results = search_index_by_key(search_key, weighted_index, file_index)


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
for i in range(len(weighted_results)):
    im = weighted_results[i][1]
    img=mpimg.imread(im)
    if i==0:
        img_ = img.copy()
    else:
        img_ = np.concatenate((img_,img), axis=1)

plt.figure(figsize=(15,15))
plt.imshow(img_)
plt.show()



import os
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image


def load_paired_img_wrd(folder):

    class_names = [fold for fold in os.listdir(folder) if ".DS" not in fold]
    image_list = []
    labels_list = []
    paths_list = []
    for cl in class_names:
        splits = cl.split("_")
        
        subfiles = [f for f in os.listdir(folder + "/" + cl) if ".DS" not in f]

        for subf in subfiles:
            full_path = os.path.join(folder, cl, subf)
            img = image.load_img(full_path, target_size=(224, 224))
            x_raw = image.img_to_array(img)
            x_expand = np.expand_dims(x_raw, axis=0)
            x = preprocess_input(x_expand)
            image_list.append(x)
            paths_list.append(full_path)
    img_data = np.array(image_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]

    return img_data, paths_list



# val_path = '/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Data/ValidationData'

val_path = '/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/Data/ValidationData'

img, img_paths = load_paired_img_wrd(val_path)



from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model

def load_headless_pretrained_model():

    pretrained_vgg16 = vgg16.VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=pretrained_vgg16.input,
                  outputs=pretrained_vgg16.get_layer('fc2').output)
    return model


model = load_headless_pretrained_model()


def generate_features(image_paths, model):

    print ("Generating features...")
    start = time.time()
    images = np.zeros(shape=(len(image_paths), 224, 224, 3))
    file_mapping = {i: f for i, f in enumerate(image_paths)}

    # We load all our dataset in memory because it is relatively small
    for i, f in enumerate(image_paths):
        img = image.load_img(f, target_size=(224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        images[i, :, :, :] = x_expand

    logger.info("%s images loaded" % len(images))
    inputs = preprocess_input(images)
    logger.info("Images preprocessed")
    images_features = model.predict(inputs)
    end = time.time()
    logger.info("Inference done, %s Generation time" % (end - start))
    return images_features, file_mapping


img_features, file_index1 = generate_features(img_paths, model)


a = []
for i in range(len(file_index),len(file_index)+len(file_index1)+1):
    a.append(i)


b = file_index1.values()



dictionary = dict(zip(a,b))


file_index.update(dictionary)


weighted_features1 = get_weighted_features(284, img_features)

weighted_index1 = index_features(weighted_features1)


vec1 = []

for i in range(len(dictionary)):
    x = weighted_index1.get_item_vector(i)
    vec1.append(x)


for i in range(len(file_index)-len(file_index1),len(file_index)):
    weighted_index.add_item(i,vec1[i-(len(file_index)-len(file_index1))])


weighted_index.get_n_items()
############################################################
st.title('Đồ án: Truy vấn hình ảnh')


st.header('I. Các bước thực hiện')


st.subheader('1. Tải và đọc hình ảnh')


# st.text('Tải dữ liệu từ shopee')

st.write('Tải dữ liệu từ shopee')

from PIL import Image

# img = Image.open('/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Report/Trainingdata.png')

img = Image.open('/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/images/Trainingdata.png')

st.image(img, width =500, caption = 'Số lượng ảnh chia theo class')


st.subheader('2. Indexing hình ảnh')

st.write('Sử dụng pre-trained model VGG16 để tạo image features, lấy penultimate layer')

img1 = Image.open('/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Report/Images for report/VGG16.jpeg')

img1 = Image.open('/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/images/VGG16.jpeg')

st.image(img1, width =1000, caption = 'Pre-trained model: VGG16')


st.write('Hình ảnh image_features tại đây')

# img2 = Image.open('/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Report/Images for report/Image embeddings.png')

img2 = Image.open('/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/images/Image embeddings.png')

st.image(img2, width =1000, caption = 'Image embeddings')


st.write('Sử dụng Annoy để tạo index_features')


st.subheader('3. Sử dụng embeddings to search through images')

st.write('Input images tại đây')

# img3 = Image.open('/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Report/Images for report/input_train.png')

img3 = Image.open('/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/images/input_train.png')

st.image(img3, width =300, caption = 'Input image')

st.write('Output images tại đây')

# img4 = Image.open('/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Report/Images for report/Output.png')

img4 = Image.open('/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/images/Output.png')

st.image(img4, width =1000, caption = 'Output images')


st.header('II. Demo')

st.subheader('1. Input tại đây')
st.set_option('deprecation.showfileUploaderEncoding', False)
from PIL import Image

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    # image.save('/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Data/ValidationData/images/demo.jpg')
    image.save('/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/Data/ValidationData/demo1.jpg')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
#     st.write(uploaded_file)

st.subheader('2. Output')
if st.button('Run'):
    
    # input_path = '/Users/mac/Documents/Machine Learning/Machine Learning Courses/Courses at Vietnam National University/10. Capstone project/Final project/Data/ValidationData/images/demo.jpg'
    input_path = '/Users/mac/Documents/5. Machine Learning & Deep Learning/4. Machine Learning Courses/3. Courses at Vietnam National University/VuAnhTu_CapstoneProject_K259/Image-to-Image-Search/Data/ValidationData/demo1.jpg'
    search_key1 = get_index(input_path, file_index)
    results = search_index_by_key(search_key1, weighted_index, file_index)
    for i in range(len(results)):
        im = results[i][1]
        img=mpimg.imread(im)
        if i==0:
            img_ = img.copy()
        else:
            img_ = np.concatenate((img_,img), axis=1)
    st.image(img_, caption='Uploaded Image.', use_column_width=True)


