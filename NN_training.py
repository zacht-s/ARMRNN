import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


def build_and_train_nn(data, topology, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                       loss=tf.keras.losses.MAE, epochs=10,
                       metrics=[tf.keras.metrics.MAE, tf.keras.metrics.mean_squared_error], save=True, name='NN_model'):

    y_raw = data['TP1']
    x_raw = data.loc[:, data.columns != 'TP1']
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.2, random_state=30)

    input_size = topology.pop(0)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_size))

    for i in topology:
        model.add(tf.keras.layers.Dense(i))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train, y_train, epochs=epochs)
    print('STARTING EVAL')
    model.evaluate(x_test, y_test)

    print(model.summary())

    if save:
        model.save(name)
    else:
        pass


if __name__ == '__main__':
    raw_data = pd.read_csv('test.csv')
    topology = [4, 16, 4, 1]
    build_and_train_nn(raw_data, topology, name='ARMRNN41')



