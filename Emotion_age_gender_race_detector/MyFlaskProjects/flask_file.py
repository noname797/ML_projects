# from flask import Flask
# app = Flask(__name__)

# @app.route("/")

# def hello():
#     return "Hello World!"

# if __name__ == "__main__":
#     app.run()

import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
from tensorflow import keras
import numpy as np
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

face_classifier = cv2.CascadeClassifier('/home/noname797/Work/Files/haarcascade_frontalface_default.xml')
def face_detector(img):
    # Convert image to grayscale
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #BGR2GRAY for gray images / self trained data
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return np.zeros((128,128), np.uint8)
    
    x,y,w,h = faces[0]
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = img[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (128, 128), interpolation = cv2.INTER_AREA)
    return roi_gray


UPLOAD_FOLDER = '/home/noname797/Work/ML_projects/MyFlaskProjects/Uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'scoobydoo'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])

def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image_path=os.path.dirname(os.path.realpath(__file__))+"/Uploads/"+filename
			image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/Uploads/"+filename)
			image=face_detector(image)
			result = prediction(image)
			redirect(url_for('upload_file',filename=filename))
			return """
			<!doctype html>
			<title>Results</title>
			<br>
			<img src='"""+image_path+"""' alt="photo">
			<h2>Image contains a - """+result+ """</h2>
			<form method=post enctype=multipart/form-data>
			  <input type=file name=file>
			  <input type=submit value=Upload>
			</form>
			"""
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form method=post enctype=multipart/form-data>
	  <input type=file name=file>
	  <input type=submit value=Upload>
	</form>
	'''
class_labels_gender={0: 'male', 1: 'female'}
class_labels_race={0:"White",1:"Black",2:"Asian",3:"Indian",4:"Others"}
class_labels_emotion={0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
class_labels_short_age={0: '11', 1: '14', 2: '17', 3: '2', 4: '20', 5: '23', 6: '26', 7: '29', 8: '32', 9: '35', 10: '38', 11: '41', 12: '44', 13: '47', 14: '5', 15: '50', 16: '53', 17: '56', 18: '59', 19: '8'}


classifier_gender=tf.keras.models.load_model("/home/noname797/Work/Files/gender_detect.hd5")
classifier_emotions=tf.keras.models.load_model("/home/noname797/Work/Files/emotion_effnet.hd5")
classifier_age=tf.keras.models.load_model("/home/noname797/Work/Files/short_age_detect.hd5") # Using short_age
classifier_race=tf.keras.models.load_model("/home/noname797/Work/Files/race_detect.hd5")

def prediction(image_in):
    img_emotion=cv2.resize(image_in,(48,48),interpolation=cv2.INTER_CUBIC)
    x_emotion=tf.keras.preprocessing.image.img_to_array(img_emotion)
    x_emotion=x_emotion*1./255
    x_emotion=np.expand_dims(x_emotion,axis=0)
    images_emotion=np.vstack([x_emotion])
    x=tf.keras.preprocessing.image.img_to_array(image_in)
    x=x*1./255
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    classes_emotion=classifier_emotions.predict_classes(images_emotion)
    classes_gender=classifier_gender.predict_classes(images)
    classes_age=classifier_age.predict_classes(images)
    classes_race=classifier_race.predict_classes(images)
    text=  "Prediction {} {} {} {}".format(class_labels_emotion[classes_emotion[0]],class_labels_gender[classes_gender[0]],class_labels_short_age[classes_age[0]],class_labels_race[classes_race[0]])
    return text


if __name__ == "__main__":
	app.run()