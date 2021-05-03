import random
import os
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import librosa
import numpy as np
from keras.models import Sequential, Model, model_from_json
import keras 
import pickle
import os
import pandas as pd
import sys
import warnings
import librosa.display
import requests
#import IPython.display as ipd



# instantiate flask app
app = Flask(__name__)

# def predictOut(file_path):
       

#         json_file = open('./model/model_json.json', 'r')
#         loaded_model_json = json_file.read()
#         json_file.close()
#         loaded_model = model_from_json(loaded_model_json)

# # load weights into new model
#         loaded_model.load_weights("./model/Emotion_Model.h5")
#         print("Loaded model from disk")

# # the optimiser
#         opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
#         loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#         print("Loaded model from disk - 1")
#         #preprocessing
#         newdf = preprocess(file_path)
#         print("Loaded model from disk - preprocessing")
#         # Apply predictions
#         newdf= np.expand_dims(newdf, axis=2)
#         newpred = loaded_model.predict(newdf, 
#                          batch_size=16, 
#                          verbose=1)
#         print("Loaded model from disk - before labels")
#         filename = './labels'
#         infile = open(filename,'rb')
#         lb = pickle.load(infile)
#         infile.close()
#         print("Loaded model from disk - labels")
#         # Get the final predicted label
#         final = newpred.argmax(axis=1)
#         final = final.astype(int).flatten()
#         final = (lb.inverse_transform((final)))
#         print("Loaded model from disk - return", final)
#         return final
class livePredictions:
    """
    Main class of the application.
    """

    def __init__(self, path, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.path = path
        self.file = file

    def load_model(self):
        """
        Method to load the chosen model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        """
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def convertclasstoemotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

    def makepredictions(self):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        return(self.convertclasstoemotion(predictions)
    

    


# def preprocess(file_path):
       
#         # Lets transform the dataset so we can apply the predictions
#         X, sample_rate = librosa.load(file_path
#                               ,res_type='kaiser_fast'
#                               ,duration=2.5
#                               ,sr=44100
#                               ,offset=0.5
#                              )
#         print('222---------------')
#         sample_rate = np.array(sample_rate)
#         mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
#         print('333-----------')
#         newdf = pd.DataFrame(data=mfccs).T
#         return newdf



@app.route("/", methods=["GET", "POST"])
def home():
    return "hello, world"
@app.route("/predict", methods=["POST"])
def predict():
	

	# get file from POST request and save it

	audio_file = request.json
	file_name = str(random.randint(0, 100000))
	r = requests.get(audio_file[url],allow_redirects=True)
	r.save(file_name)
	audio_file.save(file_name)


    pred = livePredictions(path='SER_model.h5',file=file_name)

    pred.load_model()
    predict = pred.makepredictions()

	# instantiate keyword spotting service singleton and get prediction
	# kss = Keyword_Spotting_Service()
	#predicted_keyword = predictOut(file_name)

	# we don't need the audio file any more - let's delete it!
	os.remove(file_name)

	# send back result as a json file
	result = {"keyword": predict}
	return jsonify(result)


if __name__== "__main__":
    app.run(debug=False)
    #app.run(debug=False)
