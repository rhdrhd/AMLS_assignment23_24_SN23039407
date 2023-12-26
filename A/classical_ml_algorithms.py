from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import os
#specify the core number on windows
os.environ["LOKY_MAX_CPU_COUNT"] = "10"

x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_dataset()
y_train = y_train.ravel()
y_test = y_test.ravel()

def apply_naive_bayes():
    mnb = MultinomialNB().fit(x_train, y_train)
    print("score on test: " + str(mnb.score(x_test, y_test)))
    print("score on train: "+ str(mnb.score(x_train, y_train)))

def apply_knn():
    #test a range of k Values on the Test Set:
    score_list = []

    for i in range(1, 36):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        score_list.append(knn.score(x_test, y_test))

    plt.plot(range(1,36),score_list,color='pink', linestyle='dashed', marker='o', markerfacecolor='grey',markersize=10)
    plt.title("Accuracy vs. K Value")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.savefig("A/plot_images /knn_test.png")

def apply_knn_gridsearch():

    param_grid = {
        'n_neighbors': np.arange(2,36),
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

    # Evaluate on the test set
    test_accuracy = best_model.score(x_test, y_test)
    print("Test set accuracy:", test_accuracy)
