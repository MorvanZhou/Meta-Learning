# omniglot data: https://github.com/brendenlake/omniglot/tree/master/python

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt

tf.random.set_seed(2)
np.random.seed(1)

EPOCH = 20
STEP = 1000
N_WAY = 60
N_SUPPORT = 5
N_QUERY = 5
N_EXAMPLE = 20
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL = 28, 28, 1

DATA_DIR = "../data/omniglot/images_background"


def get_train_data():
    # load train dataset
    data = []
    for root, dirs, files in os.walk(DATA_DIR, topdown=True):
        class_data = np.zeros([1, N_EXAMPLE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL], dtype=np.float32)
        for i, file in enumerate(files):
            if not file.endswith(".png"):
                continue
            if i >= N_EXAMPLE:
                break
            img_path = os.path.join(root, file)
            img = 1. - np.array(
                Image.open(img_path).resize((IMG_HEIGHT, IMG_WIDTH)),
                np.float32, copy=False)
            class_data[0, i, :, :, 0] = img
        data.append(class_data)
        if len(data) == 100:
            break
    return np.concatenate(data, axis=0)


class ConvBlock(keras.Model):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.c = keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.bn = keras.layers.BatchNormalization()
        self.p = keras.layers.MaxPool2D(2)

    def call(self, _x, training=None, mask=None):
        _x = self.c(_x)
        _x = self.bn(_x, training=training)
        _x = keras.activations.relu(_x)
        return self.p(_x)


c1 = ConvBlock(64, 3)
c2 = ConvBlock(64, 3)
c3 = ConvBlock(64, 3)
o = ConvBlock(64, 3)


class ProtoNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.o = o
        self.flat = keras.layers.Flatten()

    def call(self, _x, training=None, mask=None):
        _x = self.c1(_x, training=training)
        _x = self.c2(_x, training=training)
        _x = self.c3(_x, training=training)
        _x = self.o(_x, training=training)
        return self.flat(_x)


def euclidean_distance(qy, sy):
    # this calculation is inspired by:
    # https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/5e18a5e5b369903092f683d434efb12c7c40a83c/src/prototypical_loss.py
    # and https://github.com/abdulfatir/prototypical-networks-tensorflow/blob/master/ProtoNet-Omniglot.ipynb

    # qy [n, d]
    # sy [m, d]
    n, d = tf.shape(qy)[0], tf.shape(qy)[1]
    m = tf.shape(sy)[0]
    qy = tf.tile(tf.expand_dims(qy, axis=1), (1, m, 1))     # -> [n, m, d]
    sy = tf.tile(tf.expand_dims(sy, axis=0), (n, 1, 1))     # -> [n, m, d]
    return tf.reduce_mean(tf.square(qy - sy), axis=2)       # -> [n, m]


def train():
    @tf.function
    def train_step(sx, qx):
        with tf.GradientTape() as tape:
            # [way * support_shot, height, width, 1] support set, provides prototype
            support_y = pnet(sx)  # -> [way * support_shot, d]

            # [way * query_shot, height, width, 1] query set
            query_y = pnet_reuse(qx)  # -> [way * query_shot, d]

            # find c from support set -> [way, d]
            support_c = tf.reduce_mean(tf.reshape(support_y, [N_WAY, N_SUPPORT, -1]), axis=1)
            _loss, _acc = loss_func(query_y, support_c)

        grads = tape.gradient(_loss, pnet.trainable_variables)
        opt.apply_gradients(zip(grads, pnet.trainable_variables))
        return _loss, _acc

    def loss_func(qy, sy):
        dists = euclidean_distance(qy, sy)  # -> [way * query_shot, way]
        log_p_y = tf.reshape(
            tf.nn.log_softmax(-dists),
            [N_WAY, N_QUERY, -1]
        )  # -> [way, query_shot, way]
        cross_entropy = -tf.reduce_mean(
            tf.reshape(
                tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1),
                [-1]
            )
        )
        _acc = tf.reduce_mean(tf.dtypes.cast((tf.equal(tf.argmax(log_p_y, axis=-1), labels)), tf.float32))
        return cross_entropy, _acc

    train_data = get_train_data()   # [class, n_example, img_height, img_width, img_channel]

    pnet = ProtoNet()
    pnet_reuse = ProtoNet()
    opt = keras.optimizers.Adam(lr=0.001)

    fixed_range = np.arange(N_EXAMPLE)
    labels = np.tile(np.arange(N_WAY)[:, None], (1, N_QUERY)).astype(np.uint8)
    y_one_hot = tf.one_hot(labels, depth=N_WAY)

    for ep in range(EPOCH):
        for step in range(STEP):
            class_idx = np.random.randint(len(train_data), size=N_WAY)
            perm = np.random.permutation(fixed_range)
            support_idx = perm[:N_SUPPORT]
            query_idx = perm[N_SUPPORT: N_SUPPORT + N_QUERY]

            train_class = train_data[class_idx]
            support_x = train_class[:, support_idx]  # [way, support_shot, height, width, 1]
            query_x = train_class[:, query_idx]   # [way, query_shot, height, width, 1]
            support_x_reshape = support_x.reshape([-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
            query_x_reshape = query_x.reshape([-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])

            loss, acc = train_step(support_x_reshape, query_x_reshape)

            if step % 10 == 0:
                print("ep {} | step {} | loss {:.2f} | acc {:.2f}".format(ep, step, loss.numpy(), acc.numpy()))

    os.makedirs("./pnet", exist_ok=True)
    pnet.save_weights("./pnet/model.ckpt")


def eval_compare(src, tgts):
    src_data = np.zeros([1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL], dtype=np.float32)
    tgt_data = np.zeros([3, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL], dtype=np.float32)
    for i, file in enumerate(tgts):
        img = 1. - np.array(
            Image.open(file).resize((IMG_HEIGHT, IMG_WIDTH)),
            np.float32, copy=False)
        tgt_data[i, :, :, 0] = img
    src_data[0, :, :, 0] = 1. - np.array(
            Image.open(src).resize((IMG_HEIGHT, IMG_WIDTH)),
            np.float32, copy=False)

    latest_model = tf.train.latest_checkpoint("./pnet")
    model = ProtoNet()
    model.load_weights(latest_model)
    model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.MeanSquaredError())
    src_y = model.predict(src_data)
    tgt_y = model.predict(tgt_data)
    idx = np.mean(np.power(src_y - tgt_y, 2), axis=1).argsort()

    tgt_data = tgt_data[idx]
    tgts = [tgts[i] for i in idx]
    plt.subplot(221)
    plt.imshow(src_data[0, :, :, 0])
    plt.title(os.path.basename(src))
    plt.subplot(222)
    plt.imshow(tgt_data[0, :, :, 0])
    plt.title(os.path.basename(tgts[0]))
    plt.subplot(223)
    plt.imshow(tgt_data[1, :, :, 0])
    plt.title(os.path.basename(tgts[1]))
    plt.subplot(224)
    plt.imshow(tgt_data[2, :, :, 0])
    plt.title(os.path.basename(tgts[2]))
    plt.show()


eval_compare(
    "../data/omniglot/images_evaluation/Atemayar_Qelisayer/character07/0991_06.png",
    [
        "../data/omniglot/images_evaluation/Atemayar_Qelisayer/character01/0985_04.png",
        "../data/omniglot/images_evaluation/Atemayar_Qelisayer/character03/0987_10.png",
        "../data/omniglot/images_evaluation/Atemayar_Qelisayer/character07/0991_03.png",
    ]
)