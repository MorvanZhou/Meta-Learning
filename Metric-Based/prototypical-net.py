# omniglot data: https://github.com/brendenlake/omniglot/tree/master/python

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

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

pnet.save_weights("./pnet.ckpt")


# for step in range(STEP):
#     class_idx = np.random.randint(len(train_data), size=N_WAY)
#     perm = np.random.permutation(fixed_range)
#     support_idx = perm[:N_SUPPORT]
#     query_idx = perm[N_SUPPORT: N_SUPPORT + N_QUERY]
#
#     train_class = train_data[class_idx]
#     support_x = train_class[:, support_idx]  # [way, support_shot, height, width, 1]
#     query_x = train_class[:, query_idx]   # [way, query_shot, height, width, 1]
#     support_x_reshape = support_x.reshape([-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
#     query_x_reshape = query_x.reshape([-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
#
#     loss, acc = train_step(support_x_reshape, query_x_reshape)
#
#     if step % 10 == 0:
#         print("step {} | loss {:.2f} | acc {:.2f}".format(step, loss.numpy(), acc.numpy()))


# def conv_block(inputs, out_channels, name='conv'):
#     with tf.variable_scope(name):
#         conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
#         conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
#         conv = tf.nn.relu(conv)
#         conv = tf.contrib.layers.max_pool2d(conv, 2)
#         return conv
#
# def encoder(x, h_dim, z_dim, reuse=False):
#     with tf.variable_scope('encoder', reuse=reuse):
#         net = conv_block(x, h_dim, name='conv_1')
#         net = conv_block(net, h_dim, name='conv_2')
#         net = conv_block(net, h_dim, name='conv_3')
#         net = conv_block(net, z_dim, name='conv_4')
#         net = tf.contrib.layers.flatten(net)
#         return net

# def euclidean_distance(a, b):
#     # a.shape = N x D
#     # b.shape = M x D
#     N, D = tf.shape(a)[0], tf.shape(a)[1]
#     M = tf.shape(b)[0]
#     a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
#     b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
#     return tf.reduce_mean(tf.square(a - b), axis=2)
#
#
# # Load Train Dataset
# root_dir = './data/omniglot'
# train_split_path = os.path.join(root_dir, 'splits', 'train.txt')
# with open(train_split_path, 'r') as train_split:
#     train_classes = [line.rstrip() for line in train_split.readlines()]
# n_classes = len(train_classes)
# train_dataset = np.zeros([n_classes, n_examples, im_height, im_width], dtype=np.float32)
# for i, tc in enumerate(train_classes):
#     alphabet, character, rotation = tc.split('/')
#     rotation = float(rotation[3:])
#     im_dir = os.path.join(root_dir, 'data', alphabet, character)
#     im_files = sorted(glob.glob(os.path.join(im_dir, '*.png')))
#     for j, im_file in enumerate(im_files):
#         im = 1. - np.array(Image.open(im_file).rotate(rotation).resize((im_width, im_height)), np.float32, copy=False)
#         train_dataset[i, j] = im
# print(train_dataset.shape)
#
# x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
# q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
# x_shape = tf.shape(x)
# q_shape = tf.shape(q)
# num_classes, num_support = x_shape[0], x_shape[1]
# num_queries = q_shape[1]
# y = tf.placeholder(tf.int64, [None, None])
# y_one_hot = tf.one_hot(y, depth=num_classes)
# emb_x = encoder(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim, z_dim)
# emb_dim = tf.shape(emb_x)[-1]
# emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)
# emb_q = encoder(tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]), h_dim, z_dim, reuse=True)
# dists = euclidean_distance(emb_q, emb_x)
# log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
# ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
# acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))
#
# train_op = tf.train.AdamOptimizer().minimize(ce_loss)
#
# sess = tf.InteractiveSession()
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
#
# for ep in range(n_epochs):
#     for epi in range(n_episodes):
#         epi_classes = np.random.permutation(n_classes)[:n_way]
#         support = np.zeros([n_way, n_shot, im_height, im_width], dtype=np.float32)
#         query = np.zeros([n_way, n_query, im_height, im_width], dtype=np.float32)
#         for i, epi_cls in enumerate(epi_classes):
#             selected = np.random.permutation(n_examples)[:n_shot + n_query]
#             support[i] = train_dataset[epi_cls, selected[:n_shot]]
#             query[i] = train_dataset[epi_cls, selected[n_shot:]]
#         support = np.expand_dims(support, axis=-1)
#         query = np.expand_dims(query, axis=-1)
#         labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
#         _, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={x: support, q: query, y:labels})
#         if (epi+1) % 50 == 0:
#             print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep+1, n_epochs, epi+1, n_episodes, ls, ac))
#
# # Load Test Dataset
# root_dir = './data/omniglot'
# test_split_path = os.path.join(root_dir, 'splits', 'test.txt')
# with open(test_split_path, 'r') as test_split:
#     test_classes = [line.rstrip() for line in test_split.readlines()]
# n_test_classes = len(test_classes)
# test_dataset = np.zeros([n_test_classes, n_examples, im_height, im_width], dtype=np.float32)
# for i, tc in enumerate(test_classes):
#     alphabet, character, rotation = tc.split('/')
#     rotation = float(rotation[3:])
#     im_dir = os.path.join(root_dir, 'data', alphabet, character)
#     im_files = sorted(glob.glob(os.path.join(im_dir, '*.png')))
#     for j, im_file in enumerate(im_files):
#         im = 1. - np.array(Image.open(im_file).rotate(rotation).resize((im_width, im_height)), np.float32, copy=False)
#         test_dataset[i, j] = im
# print(test_dataset.shape)
#
#
# n_test_episodes = 1000
# n_test_way = 20
# n_test_shot = 5
# n_test_query = 15
#
# print('Testing...')
# avg_acc = 0.
# for epi in range(n_test_episodes):
#     epi_classes = np.random.permutation(n_test_classes)[:n_test_way]
#     support = np.zeros([n_test_way, n_test_shot, im_height, im_width], dtype=np.float32)
#     query = np.zeros([n_test_way, n_test_query, im_height, im_width], dtype=np.float32)
#     for i, epi_cls in enumerate(epi_classes):
#         selected = np.random.permutation(n_examples)[:n_test_shot + n_test_query]
#         support[i] = test_dataset[epi_cls, selected[:n_test_shot]]
#         query[i] = test_dataset[epi_cls, selected[n_test_shot:]]
#     support = np.expand_dims(support, axis=-1)
#     query = np.expand_dims(query, axis=-1)
#     labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
#     ls, ac = sess.run([ce_loss, acc], feed_dict={x: support, q: query, y:labels})
#     avg_acc += ac
#     if (epi+1) % 50 == 0:
#         print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_test_episodes, ls, ac))
# avg_acc /= n_test_episodes
# print('Average Test Accuracy: {:.5f}'.format(avg_acc))