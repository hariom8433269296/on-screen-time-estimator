import os
import pickle
from mtcnn import MTCNN
from tensorflow.keras.preprocessing import image
from keras.utils.layer_utils import get_source_inputs
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import tqdm
import cv2


# Load image paths
#this program is used to extract features of the images of our dataset and store them in array
def filen():
    actors = os.listdir('uploads_images')
    filenames = [os.path.join('uploads_images', actor) for actor in actors]
    pickle.dump(filenames, open('filenames.pkl', 'wb'))
    return filenames  # Return filenames


# Function for feature extraction
def feature_extraction(img_path, model, detector, retained_column_indices):
    img = image.load_img(img_path)

    # Face detection using MTCNN
    img = np.asarray(img)
    results = detector.detect_faces(img)
    if len(results) > 0:
        x1, y1, width1, height1 = results[0]['box']
        face = img[y1:y1 + height1, x1:x1 + width1]

        # Image preprocessing for VGGFace
        face = cv2.resize(face, (224, 224))
        img_array = image.img_to_array(face)
        expanded_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)

        # Feature extraction using VGGFace
        result = model.predict(preprocessed_img).flatten()

        # Retain specific columns as specified by retained_column_indices
        result_list = np.array(result)[retained_column_indices]
        return result_list
    else:
        return []


# Main function
def main():
    # Load retained_column_indices
    retained_column_indices = pickle.load(open('retained_column_indices.pkl', 'rb'))

    # Load VGGFace model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    detector = MTCNN()

    # Get filenames from filen() function
    filenames = filen()

    features = []
    for file in tqdm.tqdm(filenames, desc="Extracting Features"):
        extracted_feature = feature_extraction(file, model, detector, retained_column_indices)
        if len(extracted_feature) > 0:  # Check if any features were extracted
            features.append(extracted_feature)

    # Save features to a pickle file
    pickle.dump(features, open('features_abhishek.pkl', 'wb'))
    print(len(features))



if __name__ == "__main__":
    main()
