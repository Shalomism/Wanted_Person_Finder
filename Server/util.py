import joblib
import json
import numpy as np
import base64
import cv2
from wevelet import w2d
from mtcnn import MTCNN

detector = MTCNN()
__class_name_to_number = {}
__class_number_to_name = {}
__model = None
def classify_image(image_base64_data, file_path = None):
    imgs = get_cropped_image(file_path, image_base64_data)
    result = []
    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32*32*3, 1), scaled_img_har.reshape(32*32, 1)))
        len_image_array = 32* 32 * 3 + 32* 32

        final = combined_img.reshape(1, len_image_array).astype(float)

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })
    if len(result) > 0:
        top_score = max(result[0]['class_probability'])
        threshold = 50
        if top_score < threshold:
            return None
        else:
            return result

        

def load_saved_artifacts():
    print("Loading Started")
    global __class_name_to_number
    global __class_number_to_name
    with open("artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
    global __model
    if __model is None:
        with open("artifacts/saved_model.pkl", "rb") as f:
            __model = joblib.load(f)
    print("Artifacts Loaded")

       
def get_cv2_from_base64(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("Decoded image shape:", img.shape if img is not None else None)
    return img

def get_cropped_image(image_path, image_base64_data):

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_from_base64(image_base64_data)
    cropped_faces = []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print("Running detection...")
    results = detector.detect_faces(img_rgb)
    # print("Detection complete. Results:", results)
    # Get the first detected face only
    cropped_faces = []
    for result in results:
        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)
        cropped = img[y:y+h, x:x+w]
        cropped_faces.append(cropped)
    return cropped_faces
        
def get_base64():
    with open("b64.txt") as f:
        return f.read()

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

if __name__ == '__main__':
    load_saved_artifacts()
    # print(classify_image(get_base64(), None))
    # print(classify_image(None, "test_images/test1.jpeg"))
    # print(classify_image(None, "test_images/test2.jpeg"))
    # print(classify_image(None, "test_images/test3.jpg"))
    print(classify_image(None, "test_images/test4.jpeg"))