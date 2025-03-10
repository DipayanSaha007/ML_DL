import os
import sys
import joblib
import json
import numpy as np
import pickle
import cv2
import pywt
from my_package import Best_Model_and_Parameters     # My package to check for best model & parameters

##### ALL THESE IS DONE IN THE 'tempCodeRunnerFile.py'

# #### (1) Preprocessing: Detect face and eyes -->
#     ## When we look at any image, most of the time we identify a person using a face. 
#     ## An image might contain multiple faces, also the face can be obstructed and not clear. 
#     ## The first step in our pre-processing pipeline is to detect faces from an image. 
#     ## Once face is detected, we will detect eyes, if two eyes are detected then only we keep that image otherwise discard it.
# # Use OpenCV's Prebuilt Path
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# # Function for detecting face & eyes
# def get_cropped_image_(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         if len(eyes) >= 2:
#             return roi_color

# # Path to the directory
# path_to_data = './Project/Image-Project/images_dataset/'
# path_to_cr_data = "./Project/Image-Project/images_dataset/cropped/"

# # For loading all the directories path inside 'images_dataset'
# img_dir = []
# for entry in os.scandir(path_to_data):
#     if entry.is_dir():
#         img_dir.append(entry.path)

# cropped_img_dir = []
# celebrity_file_names_dict = {}
print("(1) Preprocessing: Detect face and eyes --> Done")

# #### (2) Preprocessing: Crop the facial region of the image -->
# # Creating the 'cropped' directory
# if os.path.exists(path_to_cr_data):
#     # shutil.rmtree(path_to_cr_data)
#     h=1
# else:
#     os.mkdir(path_to_cr_data)

# ## Once the directory is created i don't need it again
# # Making cropped images in the directory
# for img_dir in img_dir:
#     count = 1
#     celeb_name = img_dir.split('/')[-1]
#     celebrity_file_names_dict[celeb_name] = []
#     for entry in os.scandir(img_dir):
#         roi_color = get_cropped_image_(entry.path)
#         if roi_color is not None:
#             cropped_folder = path_to_cr_data + celeb_name
#             if not os.path.exists(cropped_folder):
#                 os.makedirs(cropped_folder)
#                 cropped_img_dir.append(cropped_folder)
#             cropped_file_name = celeb_name + str(count) + ".png"
#             cropped_file_path = cropped_folder + "/" + cropped_file_name

#             cv2.imwrite(cropped_file_path, roi_color)
#             celebrity_file_names_dict[celeb_name].append(cropped_file_path)
#             count += 1

##### NOW LOADING THOSE DATAS FROM THE OTHER FILE
# Path to the pickle file
pickle_file = r'C:\Users\User\OneDrive\Desktop\ML Projects\Project\Image-Project\processed_data.pkl'

# Read data from the pickle file
with open(pickle_file, 'rb') as f:
    processed_data = pickle.load(f)

# Accessing the data from the loaded dictionary
cropped_img_dir = processed_data['cropped_img_dir']
celebrity_file_names_dict = processed_data['celebrity_file_names_dict']
img_dir = processed_data['img_dir']

print("(2) Preprocessing: Crop the facial region of the image --> Done")
# print(celebrity_file_names_dict)

#### (3) Preprocessing: Use wavelet transform as a feature for traning our model -->
    ## To craete a wavelet transformed image that gives clues on facial features such as eyes, nose, lips etc.
    ##  This along with raw pixel image can be used as an input for our classifier
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0
    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)
    return imArray_H

# To get the celeb names as numbers
class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1

# Suppress all stderr (warnings, errors)
sys.stderr = open(os.devnull, 'w')
# Example function to safely load images, and handle missing files
def safe_imread(image_path):
    if not os.path.exists(image_path):
        return None  # Return None if the file doesn't exist
    return cv2.imread(image_path)

x = []
y = []
# Using the above function
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = safe_imread(training_image)  # Use safe_imread to load image
        if img is None:
            continue
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har= w2d(img, 'db1', 5)
        scalled_raw_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack([scalled_raw_img.reshape(32*32*3, 1), scalled_raw_img_har.reshape(32*32, 1)])
        x.append(combined_img)
        y.append(class_dict[celebrity_name])
x = np.array(x).reshape(len(x), 4096).astype(float)
print("(3) Preprocessing: Use wavelet transform and get 'x' and 'y' --> Done")

#### (4) Training the model & saving it in a 'save_model.pkl' file -->
    ## Data cleaning process is done. Now we are ready to train our model
df = Best_Model_and_Parameters.get_best(x, y)
# print(df)
best_clf = Best_Model_and_Parameters.best_estimator['svm']
# print(best_clf)

# Save the class dictionary
with open(r"C:\Users\User\OneDrive\Desktop\ML Projects\Project\Image-Project\class_dictionary.json", "w") as f:
    f.write(json.dumps(class_dict))

# Save the model as a pickle in a file
try:
    joblib.dump(best_clf, r'C:\Users\User\OneDrive\Desktop\ML Projects\Project\Image-Project\saved_model.pkl')
    print("(4) Training the model & saving it in a 'save_model.pkl' file --> Done")
except Exception as e:
    print(f"Error: {e}")
