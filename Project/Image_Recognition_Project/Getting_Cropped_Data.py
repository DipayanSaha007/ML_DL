# image_preprocessing.py
import os
import shutil
import cv2
import pickle

# Global variables to hold the processed data
cropped_img_dir = []
celebrity_file_names_dict = {}
img_dir = []

# Path to store serialized data
pickle_file = './Project/Image-Project/processed_data.pkl'

# Use OpenCV's Prebuilt Path
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function for detecting face & eyes
def get_cropped_image_(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

# Path to the directory
path_to_data = './Project/Image-Project/images_dataset/'
path_to_cr_data = "./Project/Image-Project/images_dataset/cropped/"

# Function to perform preprocessing (called only once)
def preprocess_images():
    global cropped_img_dir, celebrity_file_names_dict, img_dir
    
    # # Check if the data is already saved
    # if os.path.exists(pickle_file):
    #     with open(pickle_file, 'rb') as f:
    #         processed_data = pickle.load(f)
    #         cropped_img_dir = processed_data['cropped_img_dir']
    #         celebrity_file_names_dict = processed_data['celebrity_file_names_dict']
    #         img_dir = processed_data['img_dir']
    #         print("Data loaded from pickle file.")
    #         return

    # For loading all the directories path inside 'images_dataset'
    img_dir = []
    for entry in os.scandir(path_to_data):
        if entry.is_dir():
            img_dir.append(entry.path)

    cropped_img_dir = []
    celebrity_file_names_dict = {}

    # Creating the 'cropped' directory
    if os.path.exists(path_to_cr_data):
        shutil.rmtree(path_to_cr_data)
    os.mkdir(path_to_cr_data)

    # Making cropped images in the directory
    for img_dir_path in img_dir:
        count = 1
        celeb_name = img_dir_path.split('/')[-1]
        celebrity_file_names_dict[celeb_name] = []
        for entry in os.scandir(img_dir_path):
            roi_color = get_cropped_image_(entry.path)
            if roi_color is not None:
                cropped_folder = path_to_cr_data + celeb_name
                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)
                    cropped_img_dir.append(cropped_folder)
                cropped_file_name = celeb_name + str(count) + ".png"
                cropped_file_path = cropped_folder + "/" + cropped_file_name

                cv2.imwrite(cropped_file_path, roi_color)
                celebrity_file_names_dict[celeb_name].append(cropped_file_path)
                count += 1
    
    # Save the processed data to a file using pickle
    processed_data = {
        'cropped_img_dir': cropped_img_dir,
        'celebrity_file_names_dict': celebrity_file_names_dict,
        'img_dir': img_dir
    }
    with open(pickle_file, 'wb') as f:
        pickle.dump(processed_data, f)

    print("Preprocessing done and data saved to pickle.")

# Functions to return values
def get_cropped_img_dir():
    preprocess_images()  # Ensure preprocessing is done before returning data
    return cropped_img_dir

def get_celebrity_file_names_dict():
    preprocess_images()  # Ensure preprocessing is done before returning data
    return celebrity_file_names_dict

def get_img_dir():
    preprocess_images()  # Ensure preprocessing is done before returning data
    return img_dir

preprocess_images()