# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:05:11 2018

@author: 13383861
"""

import requests
def get_image_response(image_loc: str):
    headers = {'Prediction-Key': "fdc828690c3843fe8dc65e532d506d7e", "Content-type": "application/octet-stream", "Content-Length": "1000"}
    with open(image_loc,'rb') as f:
        response =requests.post('https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/287a5a82-272d-45f3-be6a-98bdeba3454c/image?iterationId=3d1fd99c-0b93-432f-b275-77260edc46d3', data=f, headers=headers)
    return response.json()


def get_highest_pred(image_json):
    max_pred = 0
    max_pred_details = ''
    for pred in image_json['predictions']:
        if pred['probability'] > max_pred:
            max_pred = pred['probability']
            max_pred_details = pred
    return max_pred, max_pred_details
        
sensor_reading = lambda image_loc: get_highest_pred(get_image_response(image_loc))

if __name__ == "__main__":
    assert get_highest_pred(get_image_response('C:/Users/13383861/Downloads/test_train.jpg'))[0] > 0.6
