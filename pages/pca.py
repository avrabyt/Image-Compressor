import numpy as np
import cv2 #pip install opencv-python
import streamlit as st
from sklearn.decomposition import PCA
import os, random

st.set_page_config(layout="wide")

if 'random_file' not in st.session_state:
    st.session_state.random_file = "cat2.jpg"

st.title("Usage of Principal Component Analysis (PCA) in Image compression")
uploaded_file = st.sidebar.file_uploader("", type=['jpg','png','jpeg','tiff','bmp'])

if uploaded_file is not None:
    st.sidebar.info("File uploaded : " + uploaded_file.name)
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

else:
    # This button doesnot exist when file upload exists
    # Try new examples from Image directory 
    # Updates session state variable random file 
    if st.sidebar.button("Try New Example Image!"):
        random_file = random.choice(os.listdir("Image"))
        st.session_state.random_file = random_file
    
    # Opens file from the session state variable
    st.sidebar.info("Example file : " + st.session_state.random_file)
    image = cv2.imread("Image/{}".format(st.session_state.random_file))

# Split the Image into RGB arrays
blue,green,red = cv2.split(image)
st.sidebar.write(blue.shape)
# Compute PCA for Individual arrays 
pca_components = st.slider("No of PCA components",0, blue.shape[0],20)
pca = PCA(pca_components)
# pca.fit(image)
# Transform each channel and then Inverse it
redT = pca.fit_transform(red)
redI = pca.inverse_transform(redT)

greenT = pca.fit_transform(green)
greenI = pca.inverse_transform(greenT)

blueT = pca.fit_transform(blue)
blueI = pca.inverse_transform(blueT)

# Reconstruct the Image
re_image = (np.dstack((blueI, greenI, redI))).astype(np.uint8)

# Dump the image
col1, col2 = st.columns(2)
with col1:
    st.image(image,use_column_width="always")
with col2:
    st.image(re_image, use_column_width="always")

# with st.expander("Best number of components"):
#     # https://www.kaggle.com/code/mirzarahim/introduction-to-pca-image-compression-example/notebook
#     from matplotlib.image import imread
#     import matplotlib.pyplot as plt
#     image_raw = imread("Image/{}".format(st.session_state.random_file))
#     image_sum = image_raw.sum(axis=2)
#     image_bw = image_sum/image_sum.max()
#     pca = PCA()
#     pca.fit(image_bw)
#     # Getting the cumulative variance

#     var_cumu = np.cumsum(pca.explained_variance_ratio_)*100

#     # How many PCs explain 95% of the variance?
#     k = np.argmax(var_cumu>95)
#     st.write(k)
#     plt.figure(figsize=[10,5])
#     plt.title('Cumulative Explained Variance explained by the components')
#     plt.ylabel('Cumulative Explained variance')
#     plt.xlabel('Principal components')
#     plt.axvline(x=k, color="k", linestyle="--")
#     plt.axhline(y=95, color="r", linestyle="--")
#     ax = plt.plot(var_cumu)
#     st.pyplot(plt)