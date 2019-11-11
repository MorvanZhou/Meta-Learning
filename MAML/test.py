from MAML.model import Model
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def eval(meta_path, normal_path, k_shot, lr_meta, test_tasks, test_epoch, loss_func):
    model = Model()
    model.build((None, 1))
    normal = Model()
    normal.build((None, 1))

    # new task
    for i in range(4):
        model.load_weights(meta_path)
        normal.load_weights(normal_path)
        new_task = test_tasks[i]
        index = np.argsort(new_task.x.ravel())
        initial_y = model(new_task.x).numpy().ravel()[index]
        normal_initial_y = normal(new_task.x).numpy().ravel()[index]

        new_task_x, new_task_y = new_task.x[:k_shot], new_task.y[:k_shot]
        # train new task
        model.compile(
            optimizer=keras.optimizers.SGD(lr_meta),
            loss=loss_func,
        )

        model.fit(new_task_x, new_task_y, epochs=test_epoch, verbose=0)

        # train from finetune
        normal.compile(optimizer=keras.optimizers.SGD(lr_meta), loss=loss_func)
        normal.fit(new_task_x, new_task_y, epochs=test_epoch, verbose=0)

        # train from scratch
        model_scratch = Model()
        model_scratch.compile(
            optimizer=keras.optimizers.SGD(lr_meta),
            loss=loss_func,
        )
        model_scratch.fit(new_task_x, new_task_y, epochs=test_epoch, verbose=0)

        x_, y_ = new_task.x.ravel()[index], model(new_task.x).numpy().ravel()[index]
        y_scratch = model_scratch(new_task.x).numpy().ravel()[index]
        y_normal = normal(new_task.x).numpy().ravel()[index]
        plt.subplot(2, 2, i + 1)
        plt.plot(x_, new_task.y.ravel()[index], label="target", c="k", alpha=0.3)
        plt.scatter(new_task_x.ravel(), new_task_y.ravel(), c="k", s=20, alpha=0.4)
        plt.plot(x_, y_, label="meta {} epoch".format(test_epoch), c="r", alpha=0.5)
        plt.plot(x_, initial_y, label="meta start point", ls="--", c="r", alpha=0.5)
        plt.plot(x_, y_scratch, label="train from scratch", c="y", alpha=0.5)
        plt.plot(x_, y_normal, label="normal {} epoch".format(test_epoch), c="b", alpha=0.5)
        plt.plot(x_, normal_initial_y, label="normal start point", c="b", ls="--", alpha=0.5)
        plt.ylim(-5.5, 5.5)
    plt.legend(prop={'size': 6})
    plt.show()
