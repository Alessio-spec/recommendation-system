from flask import Flask, request, jsonify
import prediction,evaluation
import time
from flask_cors import CORS

EXPORT_PATH = '../exportNew/'

app = Flask(__name__)
CORS(app)

@app.route('/receive-data', methods=['POST'])
def receive_data():
    # Receive data from FlutterFlow app
    data = request.json
    print("Received data:", data)

    data_string = data['data']

    # Process data
    start_time = time.time()
    # processed_data = Project.main(data_string)
    predictions = prediction.predict(data_string)
    end_time = time.time()
    evaluations = evaluation.evaluate()

    total_time = (end_time - start_time)
    # processed_data = data_string
    print('Execution time with time function in seconds: ', round(total_time, 2))
    print('predicitons: ', predictions)
    print('evalutaions: ',evaluations)
    application_data = jsonify({'predictions': predictions, 'evaluations': evaluations})
    print('Response: ', application_data)

    return application_data

if __name__ == '__main__':
    app.run(debug=False)
    # app.run(debug=True, host='0.0.0.0')
