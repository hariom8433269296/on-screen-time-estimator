import streamlit as st
import os
from keras.utils.layer_utils import get_source_inputs
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
import cv2
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
detector = MTCNN()
model = VGGFace(model = 'resnet50',include_top = False,input_shape = (224,224,3),pooling = 'avg')
features_list = pickle.load(open('features.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))
st.title('which celebrity do you look like')
up_img = st.file_uploader('choose an image')
def save_uploaded_image(up_img):
    try:
        st.text('hello')
        path = os.path.join('sample',up_img.name)
        st.text(path)
        with open(path,'wb') as f:
            f.write(up_img.getbuffer())
            return True
    except:
        st.text('false')
        return False
def extract_features(img_path,model,detector):
    sample_img = cv2.imread(img_path)
    results = detector.detect_faces(sample_img)
    x, y, width, height = results[0]['box']
    face = sample_img[y:y + height, x:x + width]
     # now extract image features
    image = Image.fromarray(face)
    image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result
def recommend(features):
    similarity = []
    for i in range(len(features_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), features_list[i].reshape(1, -1)))
    index = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    similarity=sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
    st.text(similarity)
    return index
if up_img is not None:
    if(save_uploaded_image(up_img)):
        display_image = Image.open(up_img)
        features = extract_features(os.path.join('sample',up_img.name),model,detector)
        index = recommend(features)
        st.text(index)
        col1,col2 = st.columns(2)
        with col1:
            st.header('your uploaded_img')
            st.image(display_image,width = 200)
        with col2:
            predicted_actor = " ".join(filenames[index].split('\\')[1].split('_'))
            st.header('image seems like '+ predicted_actor)
            st.image(filenames[index],width=200)