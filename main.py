import A.CNN_model as taskA_CNN
import A.classical_ml_algorithms as taskA_ml
import B.DL_model as taskB_ml

#specify task name
task_name = "taskA"

#TASK A
###################################################
if task_name == "taskA":
    print("now runing TaskA")

    training_mode = 0
    testing_mode = 1

    model_name = "CustomCNN"
    if training_mode:
        num_epochs = 100
        learning_rate = 0.0001
        taskA_CNN.train_model(model_name, num_epochs,learning_rate)
    if testing_mode:
        taskA_CNN.test_model(model_name)

    #Classical Machine Learning algorithms

    taskA_ml.apply_knn(20)

task_name = "taskB"

#TASK B
###################################################
if task_name == "taskB":
    print("now runing TaskB")

    #model name can be selected from the list shown in README.md
    model_name = "ResNet18_224"

    #uncomment the following to train model 
    #taskB_ml.train_model(model_name,50,0.00001)

    #test model
    taskB_ml.test_model(model_name)
