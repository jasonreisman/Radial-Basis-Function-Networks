import numpy as np
from sklearn.utils.multiclass import unique_labels

class OneHotEncoder(object):
    """ One hot encoder.

    This class one hot encodes vectors to sparse matrices.
    """

    def __init__(self):
        pass

    def fit(self, x):
        """Fit the encoding. Creates two dictionaries that encapsulate the forward and backwards mappings.

         Parameters
         ----------
         x : array-like, shape (n_samples,)
             Vector that contains the elements for which the mappings will be made.

         Returns
         -------
         self : object
             Returns self.
         """

        self.unique_labels_ = unique_labels(x)
        self.mapping_ = {label: i for i, label in enumerate(self.unique_labels_)}
        self.inverted_mapping_ = dict([[v, k] for k, v in self.mapping_.items()])

        return self

    def transform(self, x):
        """One hot encodes vector x according to the forward mapping.

         Parameters
         ----------
         x : array-like, shape (n_samples,)
             Vector that will be one hot encoded.

         Returns
         -------
         X : {array-like, sparse matrix}, shape (n_samples, n_unique_labels)
             One hot encoded vector x.
         """

        # x_tranformed converts the labels from 0 to n_unique_labels
        x_transformed = np.zeros(x.size, dtype=int)
        for i in range(x.size):
            x_transformed[i] = self.mapping_[x[i]]

        # Create X
        X = np.zeros((x.size, self.unique_labels_.size), dtype=int)
        X[np.arange(x.size), x_transformed] = 1

        return X

    def retransform(self, y_transformed):
        """Retransforms a vector of 0 to n_unique_labels-1 to the original mapping.

         Parameters
         ----------
         y_transformed : array-like, shape (n_samples,)
             Vector that will retransformed.

         Returns
         -------
         y : array-like, shape (n_samples,)
             Retransformed vector in the original embedding.
         """

        y = []
        for i in range(y_transformed.size):
            y.append(self.inverted_mapping_[y_transformed[i]])
        y = np.array(y)

        return y

if __name__ == '__main__':

    x = np.array(['1', '2', '3', '3'])
    oneHot = OneHotEncoder().fit(x)
    x_1hot = oneHot.transform(x)

    x_ = oneHot.retransform(x_1hot)
    pass
