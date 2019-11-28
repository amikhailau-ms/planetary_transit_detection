import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Sequential

TRAIN_DATASET = "dataset/train.npz"
TEST_DATASET = "dataset/test.npz"
KEPLER_DATASET_SIZE = 2688


def load_dataset_from_file(filename):
    data = np.load(filename)["arr_0"]
    dataX = np.empty([KEPLER_DATASET_SIZE, 1002, 1])
    dataY = np.empty([KEPLER_DATASET_SIZE, 1])
    row_counter = 0
    for x_array in data:
        for i in range(1002):
            dataX[row_counter][i][0] = x_array[1005 + i]
        dataY[row_counter][0] = x_array[2007]
    return dataX, dataY


def load_full_dataset():
    trainX, trainY = load_dataset_from_file(TRAIN_DATASET)
    testX, testY = load_dataset_from_file(TEST_DATASET)
    return trainX, trainY, testX, testY


def train_and_test_one_model(model_func, params=0):
    trainX, trainY, testX, testY = load_full_dataset()
    scores = list()
    score = model_func(trainX, trainY, testX, testY, params) * 100
    print("#{0} run for model function {1} - score {2}".format(1, model_func.__name__, score))
    scores.append(score)
    summarize_results(model_func.__name__, scores, params)
    

def summarize_results(name, scores, params):
    mean, sq_deviation = np.mean(scores), np.std(scores)
    if params == 0:
        print("Model function %s results - %.3f (+/-%.3f)" % (name, mean, sq_deviation))
    else:
        print("Model function %s with kernel size %d results - %.3f (+/-%.3f)" % (name, params, mean, sq_deviation))
        

def basic_2conv_3maxpool_no_dff_cnn_model_with_variable_kernel_size(trainX, trainY, testX, testY, kernel_size):
    if kernel_size <= 0:
        print("Wrong kernel size")
        return None
    epochs, batch_size = 3, 32
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=kernel_size, activation='relu', input_shape=(1002, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=1, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=1, kernel_size=kernel_size, activation='relu'))
    model.add(Flatten())
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
    plot_model(model, show_shapes=True, to_file='model_graphs/2conv-1maxpool.png')
    _, accuracy = model.evaluate(testX, testY, batch_size=batch_size)
    return accuracy


def basic_3conv_2maxpool_cnn_model_with_variable_kernel_size(trainX, trainY, testX, testY, kernel_size):
    if kernel_size <= 0:
        print("Wrong kernel size")
        return None
    epochs, batch_size = 3, 32
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=kernel_size, activation='relu', input_shape=(1002, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=1, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=1, kernel_size=kernel_size, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
    plot_model(model, show_shapes=True, to_file='model_graphs/3conv-2maxpool.png')
    _, accuracy = model.evaluate(testX, testY, batch_size=batch_size)
    return accuracy


def basic_1conv_1maxpool_cnn_model_with_variable_kernel_size(trainX, trainY, testX, testY, kernel_size):
    if kernel_size <= 0:
        print("Wrong kernel size")
        return None
    epochs, batch_size = 10, 32
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=kernel_size, activation='relu', input_shape=(1002, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    plot_model(model, show_shapes=True, to_file='model_graphs/1conv-1maxpool.png')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(testX, testY, batch_size=batch_size)
    return accuracy


def three_headed_1conv_1maxpool_dff_cnn_model(trainX, trainY, testX, testY, params):
    epochs, batch_size = 3, 100
    inputs1 = Input(shape=(1002,1))
    conv1 = Conv1D(filters=1, kernel_size=3, activation='relu')(inputs1)
    pool1 = MaxPooling1D(pool_size=3)(conv1)
    flat1 = Flatten()(pool1)
                    
    inputs2 = Input(shape=(1002,1))
    conv2 = Conv1D(filters=1, kernel_size=10, activation='relu')(inputs2)
    pool2 = MaxPooling1D(pool_size=3)(conv2)
    flat2 = Flatten()(pool2)
                    
    inputs3 = Input(shape=(1002,1))
    conv3 = Conv1D(filters=1, kernel_size=25, activation='relu')(inputs3)
    pool3 = MaxPooling1D(pool_size=3)(conv3)
    flat3 = Flatten()(pool3)
                    
    merged = concatenate([flat1, flat2, flat3])
    dense1 = Dense(1000, activation='relu')(merged)
    outputs = Dense(1)(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    plot_model(model, show_shapes=True, to_file='model_graphs/3-head.png')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit([trainX, trainX, trainX], trainY, epochs=epochs, batch_size=batch_size)
    _, accuracy = model.evaluate([testX, testX, testX], testY, batch_size=batch_size)
    return accuracy


train_and_test_one_model(basic_3conv_2maxpool_cnn_model_with_variable_kernel_size, 20)
#new_set = [[np.float64(kepid)], [np.float64(1002.0)], [np.float64(time_window)], time, flux, [np.float64(first_transit)], [np.float64(transit_time)]]
