import itertools
import numpy as np
import matplotlib.pyplot as plt
from utillc import *
print_everything()

ee=1.e-2 * 3
R, H, N = 1, 1, 50
ea = np.linspace(0, 2 * np.pi, N)
eh = np.linspace(0, 1, N)

y = np.asarray(list(itertools.product(ea, eh)))


f1 = lambda a, h : (R * h * np.cos(a), R * h * np.sin(a), h)
f = lambda ah : f1(ah[0], ah[1])
X = np.asarray(list(map(f, y)))
EKON(y.shape, X.shape)

X += np.random.uniform(-ee,ee, size=(N*N, 3))
y += np.random.uniform(-ee,ee, size=(N*N, 2))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('X2')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=".")

plt.show()
