from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import matplotlib.pyplot as plt
import pandas
import numpy
import torch.nn as nn
import math
import plotly.express as px
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pytrends.request import TrendReq
from datetime import timedelta
from meteostat import Hourly, Daily, Stations

dateNow = datetime.datetime.now()

app = Flask(__name__)
CORS(app)

def load_mdata():
    column_names = ['date','Cases','Rainfall','Temperature','RH','searches1','searches2']
    mdata = pandas.read_csv('../dengue/public/websiteSmooth.csv', names=column_names, header=0)
    mdata['date'] = pandas.to_datetime(mdata['date'])
    mdata.set_index('date', inplace=True)
    return mdata

mdata = load_mdata()
history = mdata[['Cases', 'Rainfall', 'Temperature', 'RH', 'searches1', 'searches2']].values
print(history)

scaler = MinMaxScaler()
history_scaled = scaler.fit_transform(history)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return numpy.array(xs), numpy.array(ys)

seq_length = 50
target_index = mdata.columns.get_loc('Cases')
X, y = create_sequences(history_scaled, seq_length)

train_size = int(len(y) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

X_train = X_train.reshape(X_train.shape[0], seq_length, X_train.shape[2]) 
X_test = X_test.reshape(X_test.shape[0], seq_length, X_test.shape[2])

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = dropout

    def forward(self, x):
        if self.training:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = nn.Identity()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout_layer(out[:, -1, :])
        out = self.fc(out)
        return out

input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 6
dropout = 0.5
lstm_model = LSTM(input_size, hidden_size, num_layers, output_size, dropout)
learning_rate = 0.001
num_epochs = 180

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

lstm_model.load_state_dict(torch.load('../dengue/public/model2.pth', weights_only=True))
lstm_model.eval()
print("LSTM model loaded.")

with torch.no_grad():
    train_outputs = lstm_model(X_train).squeeze().numpy()
    test_outputs = lstm_model(X_test).squeeze().numpy()

with torch.no_grad():
    train_outputs = lstm_model(X_train).squeeze()
    test_outputs = lstm_model(X_test).squeeze()
    train_outputs_cases = train_outputs[:, 0].numpy()
    test_outputs_cases = test_outputs[:, 0].numpy()

test_outputs_cases_reshaped = test_outputs_cases.reshape(-1, 1)
lstm_predictions = scaler.inverse_transform(numpy.hstack([test_outputs_cases_reshaped, numpy.zeros((test_outputs_cases_reshaped.shape[0], 5))]))[:, 0]

def forecast():
    mdata = load_mdata()
    
    history = mdata[['Cases', 'Rainfall', 'Temperature', 'RH', 'searches1', 'searches2']].values
    scaler = MinMaxScaler()
    history_scaled = scaler.fit_transform(history)
    
    X_full, y_full = create_sequences(history_scaled, seq_length)
    
    train_size = int(len(y_full) * 0.7)
    
    X_test_local = X_full[train_size:]
    X_test_local = torch.from_numpy(X_test_local).float()
    X_test_local = X_test_local.reshape(X_test_local.shape[0], seq_length, X_test_local.shape[2])
    
    historical_data = X_test_local.squeeze().numpy()[-1]

    num_forecast_steps = 30
    forecasted_values = []
    
    with torch.no_grad():
        for _ in range(num_forecast_steps):
            historical_data_tensor = torch.as_tensor(historical_data).float().unsqueeze(0)
            predicted_value = lstm_model(historical_data_tensor).numpy()[0, 0]
            forecasted_values.append(predicted_value)
            historical_data = numpy.roll(historical_data, shift=-1)
            historical_data[-1] = predicted_value
    
    if not pandas.api.types.is_datetime64_any_dtype(mdata.index):
        mdata.index = pandas.to_datetime(mdata.index)
        
    last_date = mdata.index[-1]
    future_dates = pandas.date_range(start=last_date + timedelta(days=1), periods=num_forecast_steps)
    
    target_index = mdata.columns.get_loc('Cases')
    target_min = scaler.data_min_[target_index]
    target_max = scaler.data_max_[target_index]
    forecasted_cases = numpy.array(forecasted_values) * (target_max - target_min) + target_min
    
    output = [future_dates, forecasted_cases]
    return output
    
def get_realtime():
    final_realtime = [[], []] # first sublist for searches, second sublist for weather
    recent_date = pandas.to_datetime(mdata.index[-1])
    day_after = recent_date + timedelta(days=1)
    timeframe = f"{recent_date.strftime('%Y-%m-%d')} {day_after.strftime('%Y-%m-%d')}"

    # search trends =======================================================================
    # access data from pytrends

    pytrends = TrendReq(hl = 'en-US', tz = 480)
    kw_list = ['dengue', 'dengue symptoms']
    pytrends.build_payload(kw_list, cat = 0, timeframe = timeframe, geo="PH")

    df = pytrends.interest_over_time()
    df = df.drop(columns=['isPartial'])

    output = [[], []]
    for i in df['dengue']: output[0].append(i)
    for i in df['dengue symptoms']: output[1].append(i)
    
    # normalize data acc to historical record
    historical_s1 = mdata['searches1'].iloc[-1]
    historical_s2 = mdata['searches2'].iloc[-1]

    final_realtime[0].append(float(historical_s1 * output[1][0] / output[0][0]))
    final_realtime[0].append(float(historical_s2 * output[1][1] / output[0][1]))

    # weather ==============================================================================

    station_id = 98430
    hourly_data = Hourly(station_id, pandas.to_datetime(str(recent_date) + " 00:00:00"),
                          pandas.to_datetime(str(day_after) + " 00:00:00")).fetch()
    final_realtime[1].append(float(hourly_data['temp'].iloc[-1]))
    final_realtime[1].append(float(hourly_data['rhum'].iloc[-1]))
    final_realtime[1].append(float(hourly_data['prcp'].iloc[-1]))

    final_realtime.append(day_after)

    return final_realtime

#debugging
@app.route('/')
def home():
    return 'balls'

@app.route('/update')
def update():
    realtime = get_realtime()
    recent_data = pandas.read_csv('../dengue/public/recent_data.csv', names=['date', 'Cases'], header=0)
    global mdata
    if mdata.index.name == "date": mdata = mdata.reset_index()
    new_data = pandas.DataFrame({
        "date": [realtime[2]],
        "Cases": [recent_data['Cases'][0]],
        "Rainfall": [realtime[1][2]],
        "Temperature": [realtime[1][0]],
        "RH": [realtime[1][1]],
        "searches1": [realtime[0][0]],
         "searches2": [realtime[0][1]]
    })

    mdata_new = pandas.concat([mdata, new_data], ignore_index=True)
    try:
        mdata_new["date"] = mdata_new["date"].dt.date
    except AttributeError:
        mdata_new["date"] = pandas.to_datetime(mdata_new["date"]).dt.date
    mdata_new.to_csv("../dengue/public/websiteSmooth.csv", index=False)

    response = jsonify({"status": "Update successful"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
    return response

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    
    try:
        data = request.get_json()
        input_features = numpy.array(data['features'], dtype=numpy.float32).reshape(1, 30, 6)
        input_tensor = torch.from_numpy(input_features).float()
        input_tensor = input_tensor[:, :-1, :]
        
        with torch.no_grad():
            prediction = forecast()
            print(prediction)
            print("================")
            output1 = list(map(lambda ts: ts.strftime("%Y-%m-%d"), prediction[0].tolist()))
            output2 = list(map(int, prediction[1].tolist()))
            
            fresh_mdata = load_mdata()
            fresh_mdata.index = pandas.to_datetime(fresh_mdata.index, errors='coerce')
            
            df = pandas.DataFrame({
                "date": [(fresh_mdata.index[-1] + timedelta(days=1)).date()],
                "Cases": [output2[0]]
            })
            df.to_csv("../dengue/public/recent_data.csv", index=False)

            df_p = pandas.DataFrame({
                "date": [x for x in output1],
                "Cases": [x for x in output2]
            })
            df_p.to_csv("../dengue/public/predictions.csv", index=False)
        
        print(output1)
        print(output2)
        return jsonify({'prediction': [output1, output2]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400 

@app.route('/plot-data')
def plot_data():
    data = pandas.read_csv('../dengue/public/predictions.csv', names=['date', 'Cases'], header=0)
    return jsonify({'x': data['date'].values(), 'y': data['Cases'].values()})

if __name__ == '__main__':
    app.run(debug=True, port=8000)