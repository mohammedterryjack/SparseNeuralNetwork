from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from matplotlib.pyplot import subplots,show

from dfa.direct_feedback_alignment import NeuralNetwork

(X_train, y_train), _ = mnist.load_data()

X_train = X_train.astype('float32')
X_train /= 255
X_train = X_train.reshape(60000, 28*28)
X_train = X_train[:1000]
X_train = X_train.T

y_train = to_categorical(y_train, 10)
y_train = y_train[:1000]
y_train = y_train.T
print('data loaded',X_train.shape, y_train.shape)
model = NeuralNetwork()
loss,error,angle = model.fit(X_train,y_train)

fig,((axis1, axis2), (axis3, _)) = subplots(2,2)
axis1.plot(loss)
axis2.plot(error,"tab:orange")
axis3.plot(angle,"tab:green")
show()