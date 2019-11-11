from tensorflow import keras
import numpy as np
from MAML.sine_task_generator import SineData, sample_shot
from MAML.model import Model
from MAML.test import eval
import os

EPOCH_IN = 8        # not too small
EPOCH_OUT = 10000
TEST_EPOCH = 32
N_CURVE = 1000
N_POINT = 50
K_SHOT = 10
LR_INNER = 0.01
LR_META = 0.01

loss_func = keras.losses.MeanSquaredError()
data = SineData(N_CURVE, N_POINT)
test_tasks = data.get_test_tasks()
meta_path = "training/reptile.ckpt"
normal_path = "training/normal_reptile.ckpt"


def train():
    model = Model()
    model.build((None, 1))
    model.compile(
        optimizer=keras.optimizers.SGD(LR_INNER),
        loss=loss_func,
    )

    meta_w = model.get_weights()
    for ep in range(EPOCH_OUT):
        task = data.sample_tasks(1)[0]
        k_shot = sample_shot(task, K_SHOT)
        # inner update
        model.fit(k_shot.x, k_shot.y, batch_size=10, epochs=EPOCH_IN, verbose=0)

        # meta update
        for tw, mw in zip(model.get_weights(), meta_w):
            mw += LR_META * (tw - mw)

        model.set_weights(meta_w)

        # test loss
        if ep % 100 == 0:
            losses = []
            for task in test_tasks:
                y_ = model(task.x)
                losses.append(loss_func(task.y, y_))
            print("ep={} | test loss={:.4f}".format(ep, np.mean(losses)))

    os.makedirs("training", exist_ok=True)
    model.save_weights(meta_path)


def normal_train():
    model = Model()
    model.compile(optimizer=keras.optimizers.SGD(LR_INNER), loss=loss_func)
    for ep in range(EPOCH_OUT):
        task = data.sample_tasks(1)[0]
        k_shot = sample_shot(task, K_SHOT)
        model.fit(k_shot.x, k_shot.y, batch_size=10, verbose=0)
    model.save_weights(normal_path)


train()
normal_train()
eval(
    meta_path=meta_path,
    normal_path=normal_path,
    k_shot=K_SHOT, lr_meta=LR_META, test_tasks=test_tasks,
    test_epoch=TEST_EPOCH, loss_func=loss_func
)