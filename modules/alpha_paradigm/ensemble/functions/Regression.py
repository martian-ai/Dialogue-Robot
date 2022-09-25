# https://zhuanlan.zhihu.com/p/51945519

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# %matplotlib inline

x = np.linspace(-10, 10, 100).reshape((-1, 1))
y = np.linspace(-20, 20, 100) + np.random.normal(loc=0, scale=3.5, size=(100, ))

df = pd.DataFrame()
df['x'] = x.reshape((-1, ))
df['y'] = y

df.plot.scatter(x='x', y='y', figsize=(12, 7))


class RegressionTree(object):

    def __init__(self):
        self._tree = None
        self.x_data = None
        self.y_data = None
        self.num_nodes = 0

    def fit(self, x, y, max_depth=3):
        self.x_data = x
        self.y_data = y
        # Calculate nodes.
        self.num_nodes = 2 ** max_depth - 1
        # Init root node.
        root_node = self.make_node(x, y)

        def _fit(_x, _y, _node):

            if self.num_nodes <= 0:
                return

            # Make R.
            x_r1, y_r1 = _x[:_node.i], _y[:_node.i]
            x_r2, y_r2 = _x[_node.i:], _y[_node.i:]

            # Make left node.
            l_node = self.make_node(x_r1, y_r1)
            _node.l_node = l_node

            self.num_nodes -= 1

            if _node.l_node:
                # Update offset.
                l_node.offset = _node.offset
                _fit(x_r1, y_r1, _node.l_node)

            # Make right node.
            r_node = self.make_node(x_r2, y_r2)
            _node.r_node = r_node

            self.num_nodes -= 1

            if _node.r_node:
                # Update offset.
                r_node.offset = _node.i + _node.offset
                _fit(x_r2, y_r2, _node.r_node)

        _fit(x, y, root_node)

        self._tree = root_node

    def predict(self, x):

        node = self._tree

        def _predict(_x, _node):

            val = self.x_data[_node.i + _node.offset, _node.j]

            if _x[_node.j] < val:
                if _node.l_node:
                    return _predict(_x, _node.l_node)
                else:
                    return _node.c1
            else:
                if _node.r_node:
                    return _predict(_x, _node.r_node)
                else:
                    return _node.c2

        return _predict(x, node)

    @staticmethod
    def make_node(x, y):
        # Get shape.
        rows, cols = x.shape
        if rows <= 1:
            return None
        # Init params.
        best_i, best_j = 1, 1
        best_c1, best_c2 = 0, 0
        best_loss = np.inf
        # Find best split.
        for i in range(1, rows):
            for j in range(0, cols):
                # Calculate c1, c2, loss.
                c1 = np.mean(y[:i])
                c2 = np.mean(y[i:])
                loss = np.mean(y[:i] - c1) + np.mean(y[i:] - c2)
                # Update best if need.
                if loss < best_loss:
                    best_loss = loss
                    best_i = i
                    best_j = j
                    best_c1 = c1
                best_c2 = c2
        node = Node(best_i, best_j, best_c1, best_c2)
        return node

class Node(object):
    def __init__(self, i, j, c1, c2, l_node=None, r_node=None):
        self.i = i
        self.j = j
        self.c1 = c1
        self.c2 = c2
        self.offset = 0
        self.l_node = l_node
        self.r_node = r_node

t = RegressionTree()

df = pd.DataFrame()
df['x'] = x.reshape((-1, ))
df = df.set_index('x')

for max_depth in range(2, 8):

    t.fit(x, y, max_depth=max_depth)

    y_predict = [t.predict(x[i, :]) for i in range(0, 100)]

    df['MAX_DEPTH_{}'.format(max_depth)] = y_predict

plt.figure(figsize=(12, 7))
plt.scatter(x, y, s=10, color='r')

for max_depth in range(2, 8):
    col_name = 'MAX_DEPTH_{}'.format(max_depth)
    plt.plot(x, df[col_name], label=col_name)

plt.title('Regression Tree')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.show()