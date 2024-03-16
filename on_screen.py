#main program which reads a video frame by frame and apply algorithms in sequence. YOLO(detect persons)->MTCNN(face detection)->
#vggface(face recognition)->cosine similarity(matching with original image)
from keras.utils.layer_utils import get_source_inputs
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
import cv2
import os
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
# Load the COCO class labels (80 classes including 'person')
def func():
    labels_path = 'yolo/coco.names'
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()

    # Load the YOLO network configuration and weights
    config_path = 'yolo/yolov3.cfg'
    weights_path = 'yolo/yolov3.weights'
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # Set the preferable target and backend for the DNN module
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    retained_column_indices= pickle.load(open('retained_column_indices.pkl','rb'))
    # Load the image
    #image_path = 'yolo/hariom.jpeg'
    # image = cv2.imread(image_path)
    actors = os.listdir('uploads_video')
    filename = [os.path.join('uploads_video', actor) for actor in actors]
    print(filename)
    for file in filename:
        cap = cv2.VideoCapture(file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    model = VGGFace(model = 'resnet50',include_top = False,input_shape = (224,224,3),pooling = 'avg')
    detector = MTCNN()
    feature_matrix = pickle.load(open('features_abhishek.pkl','rb'))
    cnt=0
    f=0
    flag = False
    threshold = 100000000
    print(fps)
    # ret, prev_frame = cap.read()
    z = 0
    while(cap.isOpened()):
        z+=1
        print("sec ",end=" ")
        print(z)
        # Create a blob from the image and perform forward pass through the network
        for i in range(fps):
         ret, image = cap.read()
        # if not ret:
        #     break
        # frame_diff = cv2.absdiff(image, prev_frame)
        # diff_sum = np.sum(frame_diff)
        # if(diff_sum < threshold):
        #     print("skipped ")
        #     continue
        # prev_frame = image
        if(ret):
            height, width = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers = net.getUnconnectedOutLayersNames()
            layer_outputs = net.forward(output_layers)

            # Define the minimum confidence threshold to filter out weak detections
            confidence_threshold = 0.5

            # Loop over each output layer and detect humans
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # Check if the detected object is a person and passes the confidence threshold
                    if class_id == 0 and confidence > confidence_threshold:
                        # Scale the bounding box coordinates to match the original image size
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        bbox_width = int(detection[2] * width)
                        bbox_height = int(detection[3] * height)
                        x = int(center_x - bbox_width / 2)
                        y = int(center_y - bbox_height / 2)

                        # Adjust bounding box coordinates within image boundaries
                        x = max(0, x)
                        y = max(0, y)
                        x2 = min(width, x + bbox_width)
                        y2 = min(height, y + bbox_height)

                        # Crop detected face region
                        frame = image[y:y2, x:x2]

                        # Display the cropped face
                        # cv2.imshow('win', frame)
                        cnt += 1
                        results = detector.detect_faces(frame)
                        if(len(results) > 0):
                            x1, y1, width1, height1 = results[0]['box']
                            face = frame[y1:y1 + height1, x1:x1 + width1]
                            try:
                                face = cv2.resize(face,(224,224))
                            except:
                                break
                            cv2.imshow('win',face)
                            face_array = np.asarray(face)
                            face_array = face_array.astype('float32')
                            expanded_img = np.expand_dims(face_array, axis=0)
                            try:
                                preprocessed_img = preprocess_input(expanded_img)
                                result = model.predict(preprocessed_img).flatten()
                                result_list = np.array(result)[retained_column_indices]
                                similarity = []
                                for i in range(len(feature_matrix)):
                                    similarity.append(cosine_similarity(result_list.reshape(1, -1), feature_matrix[i].reshape(1, -1)))
                                max_sim = max(similarity)
                                if(max_sim > 0.5):
                                    f+=1
                                    print("identified ",end="")
                                    print(f)

                                    flag = True
                                    break
                            except:
                                break
                if(flag):
                    flag=False
                    break

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    print("given person appeared in the video for ",end=" ")
    print(f,end=" ")
    print("seconds")
    cv2.destroyAllWindows()
func()
