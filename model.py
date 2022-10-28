import numpy as np
import pandas as pd
# NLP Modules
from textblob import TextBlob

# Feature Engineering
from sklearn.feature_extraction.text import CountVectorizer

# Splitting dataset for training and testing
from sklearn.model_selection import train_test_split

# Speech to Text
api_key = '82c8268643b842bab6fe3905b8109f71'
endpoint = 'https://centralindia.api.cognitive.microsoft.com/sts/v1.0/issuetoken'
import azure.cognitiveservices.speech as speechsdk

# To ignore Warnings
import warnings

warnings.simplefilter(action='ignore')

import pickle

data = pickle.load(open('data.pkl', 'rb'))

X = data['data']
y = data['class']

cv = CountVectorizer()
X_ = cv.fit_transform(X)

X_ = X_.toarray()

X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.4, random_state=True)

model = pickle.load(open('model.pkl', 'rb'))

y_pred = model.predict(X_test)


def get_sentiment(sample):
    blob = TextBlob(sample)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        result = "Positive"
    elif sentiment < 0:
        result = "Negative"
    else:
        result = "Neutral"
    return result


def detect_hate_speech(sample):
    vect = cv.transform(sample).toarray()
    pred_prob = np.max(model.predict_proba(vect))
    pred = model.predict(vect)

    return pred, pred_prob


def detect(sample):
    hatespeech = detect_hate_speech(sample)
    mp = "Try again"
    if hatespeech[0] == 0:
        mp = "Hate Speech Detected"
    elif hatespeech[0] == 1:
        mp = "Offensive Language Detected"
    elif hatespeech[0] == 2:
        mp = "NO Hate Speech Detected"

    return mp, hatespeech[1], hatespeech[0]


def recognize_from_microphone():
    speech_config = speechsdk.SpeechConfig(subscription=api_key, endpoint=endpoint)
    speech_config.speech_recognition_language = "en-US"

    ##########################################################
    speech_config.set_profanity(speechsdk.ProfanityOption.Raw)
    ##########################################################

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(speech_recognition_result.text)
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        return speech_recognition_result.no_match_details
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        return cancellation_details.reason


def voice_file(file):
    speech_config = speechsdk.SpeechConfig(subscription=api_key, endpoint=endpoint)
    speech_config.speech_recognition_language = "en-US"

    ##########################################################
    speech_config.set_profanity(speechsdk.ProfanityOption.Raw)
    ##########################################################

    audio_config = speechsdk.audio.AudioConfig(filename=file)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(speech_recognition_result.text)
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        return speech_recognition_result.no_match_details
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        return cancellation_details.reason


def get_input(choice):
    if choice == 1:
        return [input()]
    elif choice == 2:
        txt = recognize_from_microphone()
        return [txt]
