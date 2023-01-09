import numpy as np


class BoostingClassifier:
    """ Boosting for binary classification.
    Please build an boosting model by yourself.
    Examples:
    The following example shows how your boosting classifier will be used for evaluation.
    >>> X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    >>> X_test, y_test = load_test_dataset()
    >>> clf = BoostingClassifier().fit(X_train, y_train)
    >>> y_pred =  clf.predict(X_test) # this is how you get your predictions
    >>> evaluation_score(y_pred, y_test) # this is how we get your final score for the problem.
    """

    def __init__(self):
        # initialize the parameters here
        self.models = []
        self.weights = []
        self.predictions = []
        self.labels = []
        self.alpha = []
        self.instance = 1
        self.total_instances = 0
        self.ensemble_predict = 0
        self.ensemble = []
        self.ensemble_neg = 0
        self.ensemble_pos = 0
        self.negative_centroids = []
        self.positive_centroids = []

    def calculate_distance(self, X):
        for point in X:
            pos_cent_dist = np.linalg.norm(point - self.ensemble[0])
            neg_cent_dist = np.linalg.norm(point - self.ensemble[1])
            if neg_cent_dist < pos_cent_dist:
                self.predictions.append(-1)
            else:
                self.predictions.append(1)
        pass

    def calc_weighted_error(self, data, error):
        # print(error)
        w_e = error / len(data)
        return w_e

    def binary_classifier(self, weight, data, label):
        c_0 = []
        c_1 = []
        counter = 0
        # print(weight)
        for point, l in zip(data, label):
            # print(point)
            # print(weight[counter])
            if l == 1:
                c_0.append((point, weight[counter]))
                # print("{} {}".format(c_0, l))
            elif l == -1:
                c_1.append((point, weight[counter]))
                # print("{} {}".format(c_1, l))
            counter += 1
        centroids = [self.calc_centroid(c_0), self.calc_centroid(c_1)]
        # print(centroids)
        return centroids

    def calc_centroid(self, cluster):
        result = 0
        total_weight = 0
        for data, weight in cluster:
            total_weight += weight
            result += (weight * data)

        # print(total_weight)
        centroid = result / total_weight
        # print(centroid)
        return centroid

    # def compare_data(self):

    # train function
    def fit(self, X, y):
        for label in y:
            # print(label)
            self.labels.append(label)
        # end
        """ Fit the boosting model.
        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            The input samples with dtype=np.float32.

        y : { numpy.ndarray } of shape (n_samples,)
            Target values. By default, the labels will be in {-1, +1}.
        Returns
        -------
        self : object
        """

        for i in range(len(X)):
            self.weights.append(1 / len(X))
        T = 9
        # print(self.weights)
        for self.instance in range(T):
            weight_inc = 0
            weight_dec = 0
            print("Iteration:", self.instance + 1)
            self.predictions = []
            misclassified_points = 0
            # print("Iteration #", self.total_instances+1)
            # print("Weight of Points: ", self.weights)
            self.models = self.binary_classifier(self.weights, X, y)
            self.positive_centroids.append(self.models[0])
            self.negative_centroids.append(self.models[1])
            print("Positive Exemplar: ", self.models[0])
            print("Negative Exemplar: ", self.models[1])
            for point in X:
                pos_cent_dist = np.linalg.norm(point - self.models[0])
                neg_cent_dist = np.linalg.norm(point - self.models[1])
                # print("Distance from Pos Centroid", pos_cent_dist)
                # print("Distance from Neg Centroid", neg_cent_dist)
                if neg_cent_dist < pos_cent_dist:
                    self.predictions.append(-1)
                else:
                    self.predictions.append(1)
                # print(self.predictions)
            for predicted, actual in zip(self.predictions, self.labels):
                if predicted == actual:
                    continue
                else:
                    misclassified_points += 1
            # end
            print("Errors: ", misclassified_points)
            weighted_error = self.calc_weighted_error(X, misclassified_points)
            print("Error Rate: ", weighted_error)
            # print("Weighted Error", weighted_error)
            if weighted_error >= .5:
                # print("here")
                break
            self.alpha.append(.5 * np.log((1 - weighted_error) / weighted_error))  # calc confidence
            # increase/decrease weight for incorrectly/correctly classified instances
            print("Confidence/Alpha", self.alpha[self.instance])

            for ind, predicted, actual in zip(range(len(self.weights)), self.predictions, self.labels):
                # print(ind)
                old_weight = self.weights[ind]
                if predicted != actual:  # misclassified
                    self.weights[ind] /= 2 * weighted_error
                    weight_inc = self.weights[ind] / old_weight
                else:  # correctly classified
                    self.weights[ind] /= (2 * (1 - weighted_error))
                    weight_dec = self.weights[ind] / old_weight
                    # weight_dec *= self.weights[ind] / (2 * (1 - weighted_error))

            print("Factor to Increase Weights", weight_inc)
            print("Factor to Decrease Weights", weight_dec)
            # end
            self.total_instances += 1
        # end

        self.ensemble_func()
        return self

    def ensemble_func(self):
        self.ensemble_pos = np.array([self.alpha[ind] * self.positive_centroids[ind]
                                      for ind in range(self.total_instances)])

        self.ensemble_neg = np.array([self.alpha[ind] * self.negative_centroids[ind]
                                      for ind in range(self.total_instances)])

        self.ensemble.append(sum(self.ensemble_pos) / sum(self.alpha))
        self.ensemble.append(sum(self.ensemble_neg) / sum(self.alpha))
        # print(self.ensemble)
        pass

    def predict(self, X):
        """ Predict binary class for X.
        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)

        Returns
        -------
        y_pred : { numpy.ndarray } of shape (n_samples)
                 In this sample submission file, we generate all ones predictions.
        """
        # print("Ensemble Centroids: ", self.ensemble)
        self.predictions = []
        self.calculate_distance(X)
        return self.predictions
