import numpy as np
def custom_loss(y_true):
    y_true_pos = y_true.copy()
    y_true_pos[y_true_pos <= 0] = 0
    print(y_true)
    print(y_true_pos)
    return y_true,y_true_pos

y_true = np.array([[5,1,0],[2,-3,-2]])
custom_loss(y_true)