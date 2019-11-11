from tensorflow import keras
import tensorflow as tf
import numpy as np

data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise

train_x, train_y = data_x[:900], data_y[:900]
test_x, test_y = data_x[900:], data_y[900:]

l1 = keras.layers.Dense(10, activation=keras.activations.relu, name="l1")
l2 = keras.layers.Dense(1, name="l2")


class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__(name="m")
        self.l1 = l1
        self.l2 = l2

    def call(self, x, training=None, mask=None):
        x = self.l1(x)
        x = self.l2(x)
        return x


model = Model()
model2 = Model()

model.build((None, 1))
model2.build((None, 1))

model.compile(
    optimizer=keras.optimizers.SGD(0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError()],
)
model.fit(train_x, train_y, batch_size=32, epochs=3, validation_split=0.2, shuffle=True)
print(np.all(model.get_weights()[0] == model2.get_weights()[0]))

