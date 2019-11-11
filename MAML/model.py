from tensorflow import keras


class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = keras.layers.Dense(64, activation=keras.activations.tanh, input_shape=(1, ))
        self.l2 = keras.layers.Dense(64, activation=keras.activations.tanh)
        self.out = keras.layers.Dense(1)

    def call(self, x, training=None, mask=None):
        x = self.l1(x)
        x = self.l2(x)
        y = self.out(x)
        return y
