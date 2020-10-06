#! /usr/bin/env python
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import sys
import os
import logging
import re


sys.path.append('webservice')
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)


dir_name = os.path.dirname(__file__)
model_path = os.path.abspath(os.path.join(dir_name, 'static/news_verification_model.pkl'))
feature_extractor_path = os.path.abspath(os.path.join(dir_name, 'static/tfidf_feature_extractor.pkl'))


if 'news_model' not in globals():
    # load the model
    with open(model_path, 'rb') as stream:
        news_model = pickle.load(stream)
    # load the tf-idf feature extractor
    with open(feature_extractor_path, 'rb') as stream:
        tfidf_feature_extractor = pickle.load(stream)

app = Flask(__name__)
CORS(app)


def remove_special_characters(text):
    """ remove special characters from text """
    delete_chars = "[]()@''+&'"
    space_chars = "_.-"
    table = dict((ord(c), " ") for c in space_chars)
    table.update(dict((ord(c), None) for c in delete_chars))
    return text.translate(table)


def text_pre_processing(text, stop_words=True):
    """ remove special characters, hyperlinks, and lowercase everything """
    # remove hyperlinks
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text.lower(), flags=re.MULTILINE)
    text = re.sub('<[^>]*>', '', text)
    cleaned_text = remove_special_characters(text)
    return cleaned_text


def features_from_data(data):
    """"""
    text = data['text']
    corpus = [text_pre_processing(text)]
    return tfidf_feature_extractor.transform(corpus)


def get_predictions(data):
    """    predict from data applicant's data    """
    features = features_from_data(data)
    prediction = news_model.predict(features)
    probability = news_model.predict_proba(features)
    # prepare the response
    result = {}
    result['news-status'] = 'Fake' if prediction[0] == 0 else 'Real'
    result['probabilities'] = {'Fake': np.round(probability[0][0], 3),
                               'Real': np.round(probability[0][1], 3)}
    return result


def prepare_response(result):
    """    Create a response to send to the calling  application from predicted outcome    """
    response = jsonify(result)
    response.status_code = 200
    response = make_response(response)
    response.headers['Access-Control-Allow-Origin'] = "*"
    response.headers['content-type'] = "application/json"
    return response


@app.route("/api/v1/news-checker", methods=["POST"])
def get_credit_rating():
    if request.method == 'POST':
        # Get data posted as json
        app.logger.debug(request)
        data = request.get_json()
        result = get_predictions(data)
        response = prepare_response(result)
        return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)




