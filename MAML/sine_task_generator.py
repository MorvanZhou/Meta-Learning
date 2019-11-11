import numpy as np
import matplotlib.pyplot as plt


class TaskData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class SineData:
    def __init__(self, n_curves, n_point, test_ratio=0.2):
        x = np.random.rand(n_curves, n_point) * 5 * 2 - 5
        amplitude, phase = np.random.rand(n_curves, 1)*5+0.1, np.random.rand(n_curves, 1) * np.pi * 2
        y = amplitude * np.sin(x + phase)

        p = int(x.shape[0]*test_ratio)
        self.test_x, self.train_x = x[:p], x[p:]
        self.test_y, self.train_y = y[:p], y[p:]

    def sample_tasks(self, n):
        assert n <= self.train_x.shape[0]
        tasks = []
        for i in np.random.randint(0, self.train_x.shape[0], n):
            tasks.append(TaskData(self.train_x[i][:, None], self.train_y[i][:, None]))
        return tasks

    def get_test_tasks(self):
        tasks = []
        for i in range(len(self.test_x)):
            tasks.append(TaskData(self.test_x[i][:, None], self.test_y[i][:, None]))
        return tasks


def sample_shot(task, k):
    assert isinstance(task, TaskData)
    indices = np.random.randint(0, task.x.shape[0], k)
    return TaskData(task.x[indices], task.y[indices])


if __name__ == "__main__":
    gen = SineData(20, 100)
    tasks = gen.sample_tasks(3)
    for t in tasks:
        index = np.argsort(t.x.ravel())
        x_, y_ = t.x.ravel()[index], t.y.ravel()[index]
        plt.plot(x_, y_)
        plt.scatter(x_, y_)
    plt.show()

