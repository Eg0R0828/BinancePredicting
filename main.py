# -*- coding: utf-8 -*-


import pandas, tensorflow, numpy, requests
import matplotlib.pyplot as plt


# writing a csv-file with data set
def save_csv(data, filename='./data_set.csv'):
    data_frame = pandas.DataFrame(data.json())
    data_frame.to_csv(filename, index=False, header=False)


# getting a data set (example: LTC/USDT pair and 15 min interval) from an exchange web-site
def get_dataset():
    root_url = 'https://api.binance.com/api/v3/klines'
    parameters = {'symbol': 'LTCUSDT', 'interval': '15m'}
    res = None
    try:
        res = requests.get(root_url, params=parameters)
    except BaseException:
        print('ERROR-CODE [Requests]: ', res.status_code())
        return
    finally:
        save_csv(res)


# reading saved data set from the csv-file
def reading_csv(filename='./data_set.csv'):
    data_frame = None
    try:
        data_frame = numpy.array((pandas.read_csv(filename, header=None).values[:, 1:5]).astype(float))
    except BaseException:
        print('ERROR: A CSV-file (' + filename + ') was deleted or damaged!')
        return
    finally:
        return data_frame


# training and validation plots drawing
def create_metrics_plots(training_history, metrics=('loss', 'mean_absolute_error')):
    for metric in metrics:
        metric_data = training_history.history[metric]
        val_metric_data = training_history.history['val_' + metric]
        epochs = range(1, len(metric_data) + 1)

        plt.plot(epochs, metric_data, 'bo', label='Training ' + metric)
        plt.plot(epochs, val_metric_data, 'b', label='Validation ' + metric)

        plt.title('Training and validation ' + metric)
        plt.xlabel('epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.show()


# creating and training a new NN-model
def model_processing(memory_size=3, input_dim=4, model_name='./model.nn'):
    # data reading from the csv-file
    data = None
    try:
        data = reading_csv()
    except BaseException:
        return
    finally:
        # preparing training and validation data
        X = numpy.ndarray((len(data) - memory_size, memory_size, input_dim))
        Y = numpy.ndarray((len(data) - memory_size, input_dim))
        for i in range(len(data) - memory_size):
            X[i] = data[i:(i + memory_size), :] / 100
            Y[i] = data[i + memory_size, :] / 100

        # creating a model
        nn_model = tensorflow.keras.models.Sequential()
        nn_model.add(tensorflow.keras.layers.LSTM(input_dim, activation='tanh', return_sequences=True, input_shape=(memory_size, input_dim)))
        nn_model.add(tensorflow.keras.layers.LSTM(input_dim, activation='tanh', return_sequences=True))
        nn_model.add(tensorflow.keras.layers.LSTM(input_dim, activation='tanh', return_sequences=False))
        nn_model.add(tensorflow.keras.layers.Dense(input_dim, activation='tanh'))

        # compiling, training and saving a model
        nn_model.compile(optimizer='nadam', loss='mse', metrics=['mae'])
        training_history = nn_model.fit(X, Y, epochs=10, batch_size=10, validation_split=0.1)
        nn_model.save(model_name)

        # training and validation plots drawing
        create_metrics_plots(training_history)


# visualization of the difference between original and predicted data
def create_predictions_plots(original_data, predicted_data, predicted_candles):
    plt.figure()

    # original data: candlesticks
    for i in range(original_data.shape[0]):
        if original_data[i][3] >= original_data[i][0]:
            color = 'green'
            bottom = original_data[i][0]
        else:
            color = 'red'
            bottom = original_data[i][3]
        plt.bar(x=i + 1, height=abs(original_data[i][3] - original_data[i][0]), width=0.5, bottom=bottom, color=color)
        plt.bar(x=i + 1, height=abs(original_data[i][1] - original_data[i][2]), width=0.05, bottom=original_data[i][2], color=color)

    # example: close prices (original and predicted for LATEST HOUR)
    plt.plot(range(1, len(predicted_data) + 1 - predicted_candles), predicted_data[:-predicted_candles, 3], 'bo-', label='original close price')
    plt.plot(range(len(predicted_data) + 1 - predicted_candles, len(predicted_data) + 1), predicted_data[-predicted_candles:, 3], 'yo-', label='predicted close price')

    plt.title('Original and predicted data')
    plt.legend()
    plt.xlabel('timeline, *15 min')
    plt.ylabel('LTC/USDT')
    plt.show()


# using saved NN-model for predicting
def get_predictions(memory_size=3, filename='./model.nn'):
    # data reading
    try:
        data = reading_csv()
    except BaseException:
        return

    # saved model reading
    try:
        nn_model = tensorflow.keras.models.load_model(filename)
    except BaseException:
        print('ERROR: Saved NN-model file (' + filename + ') was deleted or damaged!')
        return

    # preparing an input data for predicting
    predicted_candles = 4
    original_data = data[-predicted_candles-memory_size:, :]
    predicted_data = numpy.ndarray((1, original_data.shape[0], original_data.shape[1]))
    predicted_data[0] = original_data

    # predicting
    for i in range(predicted_candles):
        predicted_data[0, -predicted_candles+i, :] = nn_model.predict(predicted_data[:, i:-predicted_candles+i, :] / 100) * 100

    # visualization of the difference between original and predicted data
    create_predictions_plots(original_data, predicted_data[0], predicted_candles)


if __name__ == '__main__':
    # a number of candles for predicting
    memory_size = 10
    # candle parameters count
    input_dim = 4

    if input('Do You want to create a new or update an existing model (Y/...)? ') == 'Y':
        if input('Do You want to download or update an existing data set (Y/...)? ') == 'Y':
            get_dataset()
        model_processing(memory_size=memory_size, input_dim=input_dim)
    if input('Do You want to test saved model (Y/...)? ') == 'Y':
        get_predictions(memory_size=memory_size)
