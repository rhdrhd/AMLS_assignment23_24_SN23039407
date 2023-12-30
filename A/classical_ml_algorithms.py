from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import os
from sklearn.metrics import roc_auc_score


#specify the core number on windows
os.environ["LOKY_MAX_CPU_COUNT"] = "10"

train_dataset, val_dataset, test_dataset = load_dataset_t1()
x_train, y_train = convert_dataset_for_classical_ml(train_dataset)
x_val, y_val = convert_dataset_for_classical_ml(val_dataset)
x_test, y_test = convert_dataset_for_classical_ml(test_dataset)

x_train = x_train.reshape(x_train.shape[0],-1)
x_val = x_val.reshape(x_val.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0],-1)


y_train = y_train.ravel()
y_test = y_test.ravel()

def apply_naive_bayes():
    mnb = MultinomialNB().fit(x_train, y_train)
    print("score on test: " + str(mnb.score(x_test, y_test)))
    print("score on train: "+ str(mnb.score(x_train, y_train)))

def apply_knn(k):
    #test a range of k Values on the Test Set:
    score_list = []

    for i in range(1, k):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        score_list.append(knn.score(x_test, y_test))

    plt.plot(range(1,k),score_list,color='pink', linestyle='dashed', marker='o', markerfacecolor='grey',markersize=10)
    plt.title("Accuracy vs. K Value")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.savefig("A/plot_images/knn_test.png")

def apply_knn_gridsearch(k):

    param_grid = {
        'n_neighbors': np.arange(1,k),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    # Create GridSearchCV object
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(x_train, y_train)

    # Check the best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    # Retrieve the best estimator
    best_model = grid_search.best_estimator_
    y_pred_probs = best_model.predict(x_test)
    auc = roc_auc_score(y_test, y_pred_probs)
    # Evaluate on the test set
    test_accuracy = best_model.score(x_test, y_test)
    print("Test set accuracy:", test_accuracy)
    print(f"Test AUC: {auc}")

apply_knn_gridsearch(68)