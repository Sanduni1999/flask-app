from flask import Flask, jsonify, request
import pickle
from flask_cors import CORS
import numpy as np
import pandas as pd


app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

# Load the scaler object
# scaler = pickle.load(open("C:/Users/sandu/OneDrive/Desktop/ReactJS/flask-app/scaler.pkl", "rb"))

scaler = None
model= None

@app.route('/hello', methods=['GET'])
def welcome():
    return jsonify({"name":"Hello World!"})


# @app.route('/predict', methods=['POST'])
# def predict():
#     try: 
#         data = request.get_json(force=True)
#         prediction_data = np.array([list(data.values())])
#         # prediction_data = np.asarray(prediction_data)
#         print(prediction_data.shape)
#         # data2d = prediction_data.reshape(prediction_data.shape[0], -1)
#         print(prediction_data.shape)
#         print(data.values())
#         # scaler = pickle.load(open("C:/Users/sandu/OneDrive/Desktop/ReactJS/flask-app/scaler.pkl",'rb'))
#         print(scaler)

#         # Check if data_2d has more than 2 dimensions
#         if len(data_2d.shape) > 2:
#             # Remove the last dimension
#             data_2d = np.squeeze(data_2d, axis=-1)
       
 
#         # scaled_data = scaler.fit_transform(prediction_data[:, np.newaxis])

#         scaled_data = scaler.transform(prediction_data.flatten().reshape(1, -1))

#         # scaled_data = scaler.transform(data2d)
#         print(scaled_data)
#         # scaled_data = d.values

#         # print(d)
#     except Exception as e:
#         return jsonify({"error":"No data found", "message":str(e)})
#     # return jsonify({"name":"Hello World!"})
#     return jsonify({"result":scaled_data.tolist()})

@app.route('/predict', methods=['POST'])
def predict():
    data_2d = None
    success_response = []
    try: 
        web_request = request.get_json(force=True)
        if(len(web_request["wp"]) > 0):
            wp = web_request["wp"]
            for i in range(len(wp)):
                data = wp[i]

                data_2d = np.array([list(data.values())]).reshape(1, -1)
                scaler = pickle.load(open("C:/Users/sandu/OneDrive/Desktop/ReactJS/flask-app/scaler.pkl",'rb'))
                scaled_data = scaler.transform(data_2d)
                print(scaled_data)
                model = pickle.load(open("C:/Users/sandu/OneDrive/Desktop/ReactJS/flask-app/votingModel.pkl", 'rb'))
                results = model.predict(scaled_data)
                print(results)

                success_response.append({"pred": (results.tolist())[0], "lat": data["Lat"], "lng": data["Long"]})

        return jsonify({"result":success_response, "status": 200})

    except Exception as e:
        return jsonify({"message":"No data found", "result":str(e), "status": 500})
    


# Route for seeing a data
@app.route('/data')
def get_time():
  
    # Returning an api for showing in  reactjs
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":"", 
        "programming":"python"
        }

if __name__ == '__main__':
    # app.run()
    scaler = pickle.load(open("C:/Users/sandu/OneDrive/Desktop/ReactJS/flask-app/scaler.pkl",'rb'))

    # app.run(host = "192.168.1.7", debug=True)
    app.run(debug = True, port = 5000)




# with open('votingModel.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Load the scaler object from the pickle file
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)





# @app.route('/data')
# def get_time():
#     # Load the input data and preprocess it as needed
#     input_data = ...

#     # Make predictions using the loaded model
#     predictions = model.predict(input_data)

#     # Return the predictions as a dictionary or JSON
#     return {'predictions': predictions.tolist()}



     