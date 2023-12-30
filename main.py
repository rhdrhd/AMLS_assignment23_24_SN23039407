import A.CNN_model as taskA_CNN
import A.classical_ml_algorithms as taskA_ml



#TASK A
###################################################

#Convolutional Neural Network
training_mode = 0
testing_mode = 1

if training_mode:
    num_epochs = 100
    learning_rate = 0.0001
    taskA_CNN.train_customCNN(num_epochs,learning_rate)

elif testing_mode:
    taskA_CNN.test_customCNN()

#Classical Machine Learning algorithms

taskA_ml.apply_knn(20)

#TASK B
###################################################
    

