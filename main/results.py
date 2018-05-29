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
import random

def results(best_estimator, X_train, X_test, y_train, y_test):
    np.random.seed(1)
    random.seed(1)
    labeled = [1, 0.85, 0.5, 0.25, 0.05]

    for perc in labeled:
        print('labeled data:', perc)
        ind = random.sample(range(0, y_train.shape[0]), int((1 - perc) * y_train.shape[0]))
        y_train_aux = y_train.copy()
        y_train_aux[ind] = -1

        #params = best_estimator.get_params
        # Make predictions
        rbfn = best_estimator.estimator #RadialBasisFunctionNetwork(**best_estimator.best_params_)

        rbfn.fit(X_train, y_train_aux)
        y_pred = rbfn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('error:', 1 - accuracy)

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
    print('iris')
    digits = datasets.load_iris()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 10),
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 500
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test

def wdbc():
    np.random.seed(1)
    print('wbdc')
    df = pd.read_csv(
        'https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wdbc/wdbc.data',
        header=None)
    y = df[[1]].values.ravel()
    X = df.drop([0, 1], axis=1).values
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 80),
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100],
                  "covariance_type": ["diag"]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1, refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test

def glass():
    np.random.seed(1)
    print('glass')
    df = pd.read_csv('../datasets/glass.csv')
    y = df[['Type']].values.ravel()
    X = df.drop(['Type'], axis=1).values
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 68),
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100],
                  "covariance_type": ["diag"]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 100
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test

def sonar():
    np.random.seed(1)
    print('sonar')
    df = pd.read_csv('../datasets/sonar.csv')
    X = np.vstack((df.columns.values, df.values))
    y = X[:, -1]
    X = X[:, :-1]
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.495)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 60),
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100],
                  "covariance_type": ["diag"]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test

def wine():
    np.random.seed(1)
    print('wine')
    digits = datasets.load_wine()
    X = digits.data
    y = digits.target
    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # specify parameters and distributions to sample from
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 60),
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100],
                  "covariance_type": ["diag"]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test

def colon():
    np.random.seed(1)
    print('colon')
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
    n_iter_search = 50
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1, refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test

def leukemia():
    np.random.seed(1)
    print('leukemia')
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
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100]}

    rbfn = RadialBasisFunctionNetwork()

    # run randomized search
    n_iter_search = 3
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test

def orl():
    np.random.seed(1)
    print('orl')
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
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100]}

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

    return random_search, X_train, X_test, y_train, y_test

def census():
    np.random.seed(1)
    print('census')
    data = pd.read_csv('/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/census_income', header=None)
    data = data.replace(' ?', np.nan)
    for i in range(data.shape[1]):
        if type(data.iat[0, i]) == str:
            data[i] = data[i].fillna(data[i].mode()[0])
        else:
            data[i] = data[i].fillna(data[i].median())
    X = data.values[:, :-1]
    y = np.zeros(data[14].shape)
    y[data[14] == ' >50K'] = 1

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 50),
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100]}

    ind_cat_features = (1, 3, 5, 6, 7, 8, 9, 13)
    cat_features = [np.unique(X[:, 1]), np.unique(X[:, 3]), np.unique(X[:, 5]), np.unique(X[:, 6]), np.unique(X[:, 7]), np.unique(X[:, 8]), np.unique(X[:, 9]), np.unique(X[:, 13])]
    rbfn = RadialBasisFunctionNetwork(ind_cat_features=ind_cat_features, cat_features=cat_features)

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test

def lenses():
    np.random.seed(1)
    print('lenses')
    data = pd.read_fwf('/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/Lenses', header=None)
    X = data.values[:, :-1]
    y = data.values[:, -1]

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 10),
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100]}

    ind_cat_features = (0,1,2,3)
    cat_features = [np.unique(X[:, 0]), np.unique(X[:, 1]), np.unique(X[:, 2]), np.unique(X[:, 3])]
    rbfn = RadialBasisFunctionNetwork(ind_cat_features=ind_cat_features, cat_features=cat_features)

    # run randomized search
    n_iter_search = 500
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test

def mushrooms():
    np.random.seed(1)
    print('mush')
    data = pd.read_csv('/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/mushrooms', header=None)
    data = data.replace(to_replace='?', value=np.nan )

    for i in range(data.shape[1]):
        data[i] = data[i].fillna(data[i].mode()[0])

    X = data.values[:, 1:]
    y = data.values[:, 0]


            # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    param_dist = {"link": [0, 1, 10, 50, 100, 1000, 1e5, 1e10],
                  "n_components": sp_randint(1, 10),
                  "l1": [0.001, 0.01, 0.1, 1, 10, 100],
                  "l2": [0.001, 0.01, 0.1, 1, 10, 100]}

    ind_cat_features = np.arange(X_train.shape[1])
    cat_features = []
    for i in ind_cat_features:
        cat_features.append(np.unique(X[:, i]))

    rbfn = RadialBasisFunctionNetwork(ind_cat_features=ind_cat_features, cat_features=cat_features)

    # run randomized search
    n_iter_search = 100
    random_search = RandomizedSearchCV(rbfn, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1,
                                       refit=False)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    return random_search, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # np.random.seed(1)
    # random.seed(1)
    #
    # datasets_ = [census]
    #
    # for dataset in datasets_:
    #     best_estimator, X_train, X_test, y_train, y_test = dataset()
    #     results(best_estimator, X_train, X_test, y_train, y_test)

    labeled = [1, 0.85, 0.5, 0.25, 0.05]
    #print('iris')
    data = pd.read_csv('/home/joao/Documents/Thesis/Radial_Basis_Funtion_Networks/datasets/census_income', header=None)
    data = data.replace(' ?', np.nan)
    for i in range(data.shape[1]):
        if type(data.iat[0, i]) == str:
            data[i] = data[i].fillna(data[i].mode()[0])
        else:
            data[i] = data[i].fillna(data[i].median())
    X = data.values[:, :-1]
    y = np.zeros(data[14].shape)
    y[data[14] == ' >50K'] = 1


    for perc in labeled:
        accuracy = np.array([])
        print('labeled data:', perc)

        for i in range(1,11):
            np.random.seed(i)
            random.seed(i)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            ind = random.sample(range(0, y_train.shape[0]), int((1 - perc) * y_train.shape[0]))
            y_train_aux = y_train.copy()
            y_train_aux[ind] = -1

            # specify parameters and distributions to sample from
            ind_cat_features = (1, 3, 5, 6, 7, 8, 9, 13)
            cat_features = [np.unique(X[:, 1]), np.unique(X[:, 3]), np.unique(X[:, 5]), np.unique(X[:, 6]),
                            np.unique(X[:, 7]), np.unique(X[:, 8]), np.unique(X[:, 9]), np.unique(X[:, 13])]
            param_dist = {"link": 50, "n_components": 45, "l1": 0.01, "l2": 1, "covariance_type": 'full'}
            rbfn = RadialBasisFunctionNetwork(**param_dist, ind_cat_features=ind_cat_features, cat_features=cat_features)

            rbfn.fit(X_train, y_train_aux)
            y_pred = rbfn.predict(X_test)
            accuracy = np.append(accuracy, accuracy_score(y_test, y_pred))

        mean = np.mean(accuracy)
        std = np.std(accuracy)
        print('mean error: ', 1-mean, ',std error: ', std)




