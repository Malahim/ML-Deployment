# This is a sample Python script.
import json

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sklearn.preprocessing import LabelEncoder
import pickle
import dill
import pandas as pd
from flask_cors import CORS
import platform
import os

df = pd.read_csv('dataset.csv', encoding='latin1')

newLocation = LabelEncoder()
newSkill1 = LabelEncoder()
newSkill2 = LabelEncoder()
newSkill3 = LabelEncoder()

app = Flask(__name__)
#
CORS(app)

file_size = os.path.getsize("Model_predict.pkl")
print(file_size)
print(platform.python_version())
print(pickle.format_version)

try:
    with open('Model_predict.pkl', 'rb') as file:
        model = pickle.Unpickler(file).load()
except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
    print("An error occurred while unpickling the file:", e)


df['location'] = newLocation.fit(df['location'])
df['Skill_info_1'] = newSkill1.fit(df['Skill_info_1'])
df['Skill_info_2'] = newSkill2.fit(df['Skill_info_2'])
df['Skill_info_3'] = newSkill3.fit(df['Skill_info_3'])
# creating an API object
api = Api(app)

#prediction api call
class prediction(Resource):
    def post(self):

        data = request.get_json()

        Total_Reviews = data['Total_Reviews']
        Ratings = data['Ratings']
        Location = data['Location']
        Skill_1 = data['Skill_1']
        Skill_2 = data['Skill_2']
        Skill_3 = data['Skill_3']
        inputList = {'Ratings': [Ratings], 'Total_Reviews': [Total_Reviews],  'location': [Location], 'Skill_info_1': [Skill_1], 'Skill_info_2': [Skill_2], 'Skill_info_3': [Skill_3]}
        dfM = pd.DataFrame(inputList)
        dfM['location'] = newLocation.transform(dfM['location'])
        dfM['Skill_info_1'] = newSkill1.transform(dfM['Skill_info_1'])
        dfM['Skill_info_2'] = newSkill2.transform(dfM['Skill_info_2'])
        dfM['Skill_info_3'] = newSkill2.transform(dfM['Skill_info_3'])

        prediction = model.predict(dfM)
        #print("prediction",prediction)
        if(prediction[0]>0):
            response = prediction-15
            arr_list = response.tolist()
            print(arr_list)
        else:
            response = {"Details are not enough"}

        return arr_list[0]


api.add_resource(prediction, '/prediction')

if __name__ == '_main_':
    app.run(debug=True)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
