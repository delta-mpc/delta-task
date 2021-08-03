import numpy as np


if __name__ == '__main__':
    count = 2000

    ys = np.random.randint(10, size=(count, 1))
    xs = np.random.randint(256, size=(count, 784))

    data = np.concatenate([ys, xs], axis=1)
    np.savez("mnist.npz", data)
