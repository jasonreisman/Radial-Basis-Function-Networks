import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.patches import Ellipse
from classifiers.rbfn import RadialBasisFunctionNetwork
from sklearn.model_selection import RandomizedSearchCV
from time import time
from scipy.stats import randint as sp_randint
import pandas as pd
from sklearn import preprocessing
import scipy.io as io


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def iris():
    np.random.seed(1)
    digits = datasets.load_iris()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 10),
                  "l1": [0.001, 0.01, 0.1, 1, 10],
                  "l2": [0.001, 0.01, 0.1, 1, 10]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 500
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    # Make predictions
    y_pred = random_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    rbfn = RadialBasisFunctionNetwork(link=1000, max_iter=1000, tol=1e-3, n_components=3, feature_type='post_prob',
                                      covariance_type='full',
                                      equal_covariances=False, component_kill=True, ind_cat_features=(),
                                      laplace_smoothing=0.001,
                                      reg_covar=1e-6, max_iter_gmm=1, init_params='kmeans', random_state=None, l1=0.01,
                                      l2=0.01,
                                      max_iter_logreg=5)

    rbfn.fit(X_train, y_train)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

def wdbc():
    np.random.seed(1)
    df = pd.read_csv(
        'https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wdbc/wdbc.data',
        header=None)
    y = df[[1]].values.ravel()
    X = df.drop([0, 1], axis=1).values
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 50),
                  "l1": [0.001, 0.01, 0.1, 1, 10],
                  "l2": [0.001, 0.01, 0.1, 1, 10]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 500
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1, refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    rbfn = RadialBasisFunctionNetwork(**random_search.best_params_)

    rbfn.fit(X_train, y_train)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

def glass():
    np.random.seed(1)
    df = pd.read_csv('../datasets/glass.csv')
    y = df[['Type']].values.ravel()
    X = df.drop(['Type'], axis=1).values
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 50),
                  "l1": [0.001, 0.01, 0.1, 1, 10],
                  "l2": [0.001, 0.01, 0.1, 1, 10]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 500
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    rbfn = RadialBasisFunctionNetwork(**random_search.best_params_)

    rbfn.fit(X_train, y_train)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

def sonar():
    np.random.seed(1)
    df = pd.read_csv('../datasets/sonar.csv')
    X = np.vstack((df.columns.values, df.values))
    y = X[:, -1]
    X = X[:, :-1]
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.495)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 50),
                  "l1": [0.001, 0.01, 0.1, 1, 10],
                  "l2": [0.001, 0.01, 0.1, 1, 10]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 1
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    rbfn = RadialBasisFunctionNetwork(**random_search.best_params_)

    rbfn.fit(X_train, y_train)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

def wine():
    np.random.seed(1)
    digits = datasets.load_wine()
    X = digits.data
    y = digits.target
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 50),
                  "l1": [0.001, 0.01, 0.1, 1, 10],
                  "l2": [0.001, 0.01, 0.1, 1, 10]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 500
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    rbfn = RadialBasisFunctionNetwork(**random_search.best_params_)

    rbfn.fit(X_train, y_train)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

def colon():
    np.random.seed(1)
    filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/colon.csv'
    df = pd.read_csv(filename, sep=' ', header=None)
    X = df.values.T
    filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/colon_target.csv'
    df = pd.read_csv(filename, sep=' ', header=None)
    df.values[df.values < 0] = 0
    df.values[df.values > 0] = 1
    y = df.values.ravel()
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 20),
                  "l1": [0.001, 0.01, 0.1, 1, 10],
                  "l2": [0.001, 0.01, 0.1, 1, 10]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 10
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1, refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    rbfn = RadialBasisFunctionNetwork(**random_search.best_params_)

    rbfn.fit(X_train, y_train)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

def leukemia():
    np.random.seed(1)
    filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/leukemia.csv'
    df = pd.read_csv(filename)
    X = df.values.T
    filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/colon_target.csv'
    y = np.zeros(df.columns.size)
    for ind, i in enumerate(df.columns):
        if 'AML' in i:
            y[ind] = 1

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.46)
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 25),
                  "l1": [0.001, 0.01, 0.1, 1, 10],
                  "l2": [0.001, 0.01, 0.1, 1, 10]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 5
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    rbfn = RadialBasisFunctionNetwork(**random_search.best_params_)

    rbfn.fit(X_train, y_train)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

def orl():
    file = "/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/ORL"
    data = io.loadmat(file)
    X_ = np.array(data['fea'], dtype=np.float64)
    X = np.array([]).reshape(0, int(X_.size / 400))
    for x in np.split(X_, 400):
        X = np.append(X, x, axis=0)
    y = np.asarray(data['gnd'], dtype=np.float64).ravel()

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(20, 100),
                  "l1": [0.001, 0.01, 0.1, 1, 10],
                  "l2": [0.001, 0.01, 0.1, 1, 10]}

    rbfn = RadialBasisFunctionNetwork(covariance_type="diag")

    # run randomized search
    n_iter_search = 100
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    rbfn = RadialBasisFunctionNetwork(**random_search.best_params_, covariance_type="diag")

    rbfn.fit(X_train, y_train)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

def balance_scale():
    filename = '/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/balance_scale'
    df = pd.read_csv(filename, sep=",", header=None)
    y = df[0].values
    X = df[[1, 2, 3, 4]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 50),
                  "l1": [0.001, 0.01, 0.1, 1, 10],
                  "l2": [0.001, 0.01, 0.1, 1, 10]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 500
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    rbfn = RadialBasisFunctionNetwork(**random_search.best_params_)

    rbfn.fit(X_train, y_train)
    y_pred = rbfn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

if __name__ == '__main__':

    #iris()
    #wdbc()
    #glass()
    sonar()
    #wine()
    #colon()
    #leukemia()
    #orl()