import numpy as np
import matplotlib.pyplot as plt

def neurous(sqrt=False, num=1000, b=0):
    x = np.ones(num)
    if not sqrt:
        w = np.random.rand(num) - 0.5
    else:
        w = (np.random.rand(num) - 0.5) / np.sqrt(num)
    return np.dot(w, x) + b

if __name__ == "__main__":
    iteration = 10000
    zl = []
    for i in range(iteration):
      zl.append(neurous(sqrt=True))

    n, bins, patches = plt.hist(zl, 100, alpha=0.8)
    plt.show()

    sigmod = lambda x:1. /(1. + np.exp(-x))
    al = sigmod(np.array(zl))
    n, bins, patches = plt.hist(al, 100, alpha=0.8)
    plt.show()

    np.set_printoptions(precision=2, threshold=20000)
    print(zl)
    print(al)


