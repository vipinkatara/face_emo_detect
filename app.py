# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: Krish Naik
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import requests
from flask_cors import CORS

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
CORS(app)

# Model saved with Keras model.save()
MODEL_PATH ='emo_model_vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)





def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="angry"
    elif preds==1:
        preds="disgust"
    elif preds==2:
        preds="fear"
    elif preds==3:
        preds="happy"
    elif preds==4:
        preds="neutral"
    elif preds==5:
        preds="sad"
    else:
        preds="surprise"
    
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
       link = request.json
        file_name = str(random.randint(0, 100000))
        r=requests.get(link['url'], allow_redirects=True)
        open(file_name,'wb').write(r.content)
        #r = requests.get(link, allow_redirects=True)
        #open(str, 'wb').write(r.content)
        predicted_keyword = predictOut(file_name)

        # we don't need the audio file any more - let's delete it!
        os.remove(file_name)
       
       
       
        # # Get the file from post request
        # f = request.files['file']

        # # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
