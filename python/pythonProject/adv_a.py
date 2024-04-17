from keras.models import load_model
from keras.datasets import mnist
import keras
import cv2
from g_alg import *

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# преобразование векторных классов в бинарные матрицы
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
network = load_model('mnist.h5')
score = network.evaluate(x_test, y_test, verbose = 0)
print('Потери на тесте:', score[0])
print('Точность на тесте:', score[1])
test = x_test[0].reshape(1,28,28,1)
print(network.predict(test))
cv2.imwrite('./expect_images/-1.jpg',test.reshape(28,28)*255)

population = getPopulation()
lenp = len(population)
new_cl = 4
for iter in range(400):
    z = np.zeros((lenp,1,28,28,1))
    for i in range(lenp):
        for j in range(28):
            for k in range(28):
                temp = test[0,j,k,0]+population[i,j,k,0]
                if temp<0:
                    z[i,0,j,k,0] = 0
                elif temp>1:
                    z[i,0, j, k, 0] = 1
                else:
                    z[i, 0, j, k, 0] = temp

    confidence_list = np.zeros(lenp)
    child_array = np.zeros((int(lenp/2),28,28,1))
    for i in range(lenp):
        confidence_list[i] = network.predict(z[i])[0,new_cl]
    if iter%5==0:
        conf_best = np.amax(confidence_list)
        print(confidence_list)
        ind_best = np.argmax(confidence_list)
        cv2.imwrite('./expect_images/'+str(iter)+'.jpg', z[ind_best].reshape(28, 28) * 255)
    queue = np.arange(lenp)
    np.random.shuffle(queue)
    for i in range(int(lenp/2)):
        rand1 = rd.randint(0,lenp-1)
        rand2 = rd.randint(0,lenp-1)
        child_array[i] = muteChild(crossover2(z[rand1,0],z[rand2,0]))
    arr = selectionTour(confidence_list,queue)
    for i in range(0,lenp-1,2):
        population[arr[i+1]] = child_array[int(i/2)]

for i in range(lenp):
    confidence_list[i] = network.predict(z[i])[0,new_cl]
print(np.amax(confidence_list))
print(network.predict(z[np.argmax(confidence_list)]))





