from types import MethodDescriptorType
from flask import Flask, render_template, request, json, flash, url_for
from flask.globals import session
from flaskext.mysql import MySQL
from werkzeug.utils import redirect

from flaskext.mysql import MySQL
from flask import Flask, render_template, json, request, redirect, url_for, session
from flask import jsonify
from werkzeug.utils import secure_filename

import librosa
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import numpy as np
import tensorflow as tf

from playsound import playsound
from tensorflow.python.keras.models import model_from_json
from tensorflow import keras
import pandas as pd
mysql = MySQL()

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'speech_emotion_recognition_database'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)


@app.route("/")
def home():
    return render_template("Home.html")


@app.route("/login_validation", methods=['POST'])
def login_validation():
    try:
        conn = mysql.connect()
        cursor = conn.cursor()
        _email = request.form['email']
        _password = request.form['password']

        if _email and _password:
            cursor.execute(
                'SELECT * FROM usermaster WHERE MailId = % s AND password = % s', (_email, _password))
            data = cursor.fetchone()

            if data:
                session['loggedin'] = True
                session['id'] = data[0]
                session['username'] = data[2]
                conn.commit()
                return render_template('tmp.html')
            else:
                flash('Incorrect username or password !')
                return redirect(url_for('SignIn'))

        else:

            flash('Please Enter the required fields')
            return redirect(url_for('SignIn'))

    except Exception as e:
        return json.dumps({'error': str(e)})


@ app.route("/SignIn")
def SignIn():
    return render_template('sg.html')


@ app.route("/Registration")
def Registration():
    return render_template('reg.html')


@ app.route("/register_validation", methods=['POST'])
def register_validation():
    try:
        conn = mysql.connect()
        cursor = conn.cursor()
        _name = request.form['name']
        _email = request.form['email']
        _password = request.form['password']
        _mobile = request.form['MobileNo']

        if _name and _email and _password and _mobile:
            cursor.callproc('sp_createUser',
                            (_name, _email, _password, _mobile))
            data = cursor.fetchall()

            if len(data) == 0:
                conn.commit()
                return render_template('sg.html')
            else:
                flash(str(data[0]))
                return redirect(url_for('Registration'))

        else:
            flash('Please Enter the required fields')
            return redirect(url_for('Registration'))

    except Exception as e:
        flash(str(e))
        return redirect(url_for('Registration'))


@ app.route("/tmp")
def tmp():
    return render_template('tmp.html')


@ app.route("/Logout")
def Logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return render_template('sg.html')


@app.route("/Final", methods=["POST", "GET"])
def Final():
    return render_template('final.html')


@ app.route("/UploadAudioFile", methods=['POST', 'GET'])
def UploadAudioFile():
    f = request.files['file']
    f.save('F:\\All BE Things\\main\\SpeechEmotionRecognitionProject-main\\Speech Emotion Recognition\\uploaded_voice\\recording.wav')

    data, sampling_rate = librosa.load(
        'F:\\All BE Things\\main\\SpeechEmotionRecognitionProject-main\\Speech Emotion Recognition\\uploaded_voice\\recording.wav')

    json_file = open('model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/Emotion_Model.h5")
    print("Loaded model from disk")
    # the optimiser
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer=opt, metrics=['accuracy'])

    json_file = open('model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/Emotion_Model.h5")
    print("Loaded model from disk")
    # the optimiser
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer=opt, metrics=['accuracy'])

    X, sample_rate = librosa.load('F:\\All BE Things\\main\\SpeechEmotionRecognitionProject-main\\Speech Emotion Recognition\\uploaded_voice\\recording.wav',
                                  res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)

    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=13), axis=0)
    newdf = pd.DataFrame(data=mfccs).T
    print(newdf)

    newdf = np.expand_dims(newdf, axis=2)
    newpred = loaded_model.predict(newdf,
                                   batch_size=16,
                                   verbose=1)
    print(newpred)

    import pickle
    filename = 'labels'
    infile = open(filename, 'rb')
    lb = pickle.load(infile)
    infile.close()

    # Get the final predicted label
    final = newpred.argmax(axis=1)
    final = final.astype(int).flatten()
    final = (lb.inverse_transform((final)))
    print(final)
    flash('The emotion recognised is ' +
          str(final[0]).split('_')[0]+' '+str(final[0]).split('_')[1])
    return redirect(url_for('Final'))


if __name__ == "__main__":
    app.run(debug=True)
