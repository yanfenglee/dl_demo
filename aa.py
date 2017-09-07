
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data.T
Y = digits.target.reshape(1,-1)

ntrain = 1500

train_x = X[:,:ntrain]
train_y = (Y[:,:ntrain]==4)

test_x = X[:,ntrain:]
test_y = (Y[:,ntrain:] == 4)-0

aa = train_x[:,np.squeeze(train_y)]
bb = np.c_[train_x,aa,aa,aa,aa,aa,aa,aa,aa]
cc = np.ones((1,aa.shape[1]),dtype=bool)
dd = np.c_[train_y,cc,cc,cc,cc,cc,cc,cc,cc]

print("newshape: ", train_x.shape,aa.shape,bb.shape,dd[:,2600:])



# print(X.shape, Y.shape, train_x.shape,train_y.shape,test_x.shape,test_y.shape)

# print(train_y[0,14:100])
# print(train_y.shape)

# plt.imshow(train_x[:,14].reshape(8,-1), cmap=plt.cm.gray_r, interpolation='nearest')

# #plt.imshow(digits.images[0])

# #plt.show()
# print(test_y)