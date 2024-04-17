from keras.models import load_model
from keras.datasets import mnist
import keras
from gen_alg import *


if __name__=='__main__':


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
    print(network.predict(x_test[0].reshape(1,28,28,1)))
    # print(calculateConfidence(x_train[3:4],t_train[3:4],network.predict))
    # print(t_train[3:4])
    # cv2.imshow("winname", x_train[3].reshape(28,28))
    # cv2.waitKey(0)

    population=getPopulation()
    confidence_list=np.zeros(population_size)
    best_confidence=-10000.
    best_person=np.zeros(784)
    for iter in range(iteration):
        for person in population:
            current_confidence=calculateConfidence(person, network.predict)
            if current_confidence>best_confidence:
                best_confidence=current_confidence
                best_person=person.copy()
        print('iteration:',iter,'confidence:',best_confidence)
        # print(best_confidence)
        cv2.imwrite('./exp_images/'+str(iter)+'.jpg',best_person.reshape(28,28)*255)
        for i in range(population_size):
            confidence_list[i]=calculateConfidence(population[i], network.predict)
            best_father=population[selectParent(confidence_list)]
            best_mother=population[selectParent(confidence_list)]
            child_from_best_parent=crossover(best_father, best_mother, network.predict)
            # child_from_best_parent=crossover2(best_father, best_mother)
            child_from_best_parent=muteChild(child_from_best_parent)
            population[i]=child_from_best_parent
    print('best_confidence:',best_confidence)
    print('test_confidence:',calculateConfidence(best_person,network.predict))
    # image=best_person.reshape(28,28)
    # cv2.imshow("winname", image)
    # cv2.waitKey(0)