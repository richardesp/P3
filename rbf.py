#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:37:04 2021

IMC: lab assignment 3

@author: pagutierrez
"""

# TODO Include all neccesary imports
import pickle
import os
import click
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv  # IMPORTANT IMPORT
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import false_negative_rate
from fairlearn.metrics import false_positive_rate


# TODO Include the rest of parameters...


@click.command()
@click.option('--train_file', '-t', default=None, required=False,
              help=u'Name of the file with training data.')
@click.option('--test_file', '-T', default=None, required=False,
              help=u'Name of the file with test data.')
@click.option('--classification', '-c', is_flag=True, default=False, show_default=True, required=False,
              help=u'If set, the problem is a classification problem.')
@click.option('--ratio_rbf', '-r', default=0.1, show_default=True, required=False,
              help=u'Ratio of RBFs to be used in the network.')
@click.option('-l', '--l2', is_flag=True, default=False, show_default=True, required=False,
              help=u'If set, the network will use L2 regularization.')
@click.option('-e', '--eta', default=0.00001, show_default=True, required=False,
              help=u'Learning rate.')
@click.option('-f', '--fairness', is_flag=True, default=False, show_default=True, required=False,
              help=u'If set, the network will use fairness regularization.')
@click.option('-o', '--outputs', default=1, show_default=True, required=False,
              help=u'Number of outputs in the last column.')
@click.option('--pred', '-p', is_flag=True, default=False, show_default=True,
              help=u'Use the prediction mode.')  # KAGGLE
@click.option('--model', '-m', default="", show_default=False,
              help=u'Directory name to save the models (or name of the file to load the model, if the prediction mode is active).')  # KAGGLE
def train_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, fairness, outputs, model, pred):
    """ 5 executions of RBFNN training
    
        RBF neural network based on hybrid supervised/unsupervised training.
        We run 5 executions with different seeds.
    """
    if not pred:

        if train_file is None:
            print("You have not specified the training file (-t)")
            return

        train_mses = np.empty(5)
        train_ccrs = np.empty(5)
        test_mses = np.empty(5)
        test_ccrs = np.empty(5)

        if fairness:
            train_fn0 = np.empty(5)
            train_fn1 = np.empty(5)
            test_fn0 = np.empty(5)
            test_fn1 = np.empty(5)

        for s in range(1, 6, 1):
            print("-----------")
            print("Seed: %d" % s)
            print("-----------")
            np.random.seed(s)

            train_results, test_results = \
                train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, fairness, outputs,
                          model and "{}/{}.pickle".format(model, s) or "")

            train_mses[s - 1] = train_results["mse"]
            test_mses[s - 1] = test_results["mse"]
            train_ccrs[s - 1] = train_results["ccr"]
            test_ccrs[s - 1] = test_results["ccr"]

            print("Training MSE: %f" % train_mses[s - 1])
            print("Test MSE: %f" % test_mses[s - 1])
            print("Training CCR: %.2f%%" % train_ccrs[s - 1])
            print("Test CCR: %.2f%%" % test_ccrs[s - 1])

            if fairness:
                train_fn0[s - 1] = train_results["fairnes_metrics"].by_group.to_dict()['false negative rate'][0.0] * 100
                train_fn1[s - 1] = train_results["fairnes_metrics"].by_group.to_dict()['false negative rate'][1.0] * 100
                test_fn0[s - 1] = test_results["fairnes_metrics"].by_group.to_dict()['false negative rate'][0.0] * 100
                test_fn1[s - 1] = test_results["fairnes_metrics"].by_group.to_dict()['false negative rate'][1.0] * 100

        print("******************")
        print("Summary of results")
        print("******************")
        print("Training MSE: %f +- %f" % (np.mean(train_mses), np.std(train_mses)))
        print("Test MSE: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
        print("Training CCR: %.2f%% +- %.2f%%" % (np.mean(train_ccrs), np.std(train_ccrs)))
        print("Test CCR: %.2f%% +- %.2f%%" % (np.mean(test_ccrs), np.std(test_ccrs)))
        if fairness:
            print("Training FN0: %.2f%% +- %.2f%%" % (np.mean(train_fn0), np.std(train_fn0)))
            print("Training FN1: %.2f%% +- %.2f%%" % (np.mean(train_fn1), np.std(train_fn1)))
            print("Test FN0: %.2f%% +- %.2f%%" % (np.mean(test_fn0), np.std(test_fn0)))
            print("Test FN1: %.2f%% +- %.2f%%" % (np.mean(test_fn1), np.std(test_fn1)))

    else:
        # KAGGLE
        if model is None:
            print("You have not specified the file with the model (-m).")
            return

        # Obtain the predictions for the test set
        predictions = predict(test_file, model)

        # Print the predictions in csv format
        print("Id,Category")
        for prediction, index in zip(predictions, range(len(predictions))):
            s = ""
            s += str(index)

            if isinstance(prediction, np.ndarray):
                for output in prediction:
                    s += ",{}".format(output)
            else:
                s += ",{}".format(int(prediction))

            print(s)


def train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, fairness, outputs, model_file=""):
    """ One execution of RBFNN training
    
        RBF neural network based on hybrid supervised/unsupervised training.
        We run 1 executions.

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        classification: bool
            True if it is a classification problem
        ratio_rbf: float
            Ratio (as a fraction of 1) indicating the number of RBFs
            with respect to the total number of patterns
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
        fairness: 
            False. If set to true, it will calculate fairness metrics on the prediction
        outputs: int
            Number of variables that will be used as outputs (all at the end
            of the matrix)
        model_file: string
            Name of the file where the model will be written

        Returns
        -------
        train_results: dict
            Dictionary with each metric results for training data: 'ccr' (float), 
            'mse' (float), 'fairnes_metrics' (fairlearn metrics structure). For 
            regression ccr will be zero and for classification mse will be zero.
            Fairness metrics will be available only if fairness=True is provided. 
        
        test_results: dict
            Dictionary with each metric results for testing data. Equivalent to 
            train_results but for the test set. 

    """
    train_inputs, train_outputs, test_inputs, test_outputs = read_data(train_file,
                                                                       test_file,
                                                                       outputs)

    # TODO: Obtain num_rbf from ratio_rbf

    num_rbf = int(ratio_rbf * len(train_inputs))

    print("Number of RBFs used: %d" % (num_rbf))
    # 1. Init centroids + 2. clustering 
    kmeans, distances, centers = clustering(classification, train_inputs,
                                            train_outputs, num_rbf)

    # 3. Adjust radii
    radii = calculate_radii(centers, num_rbf)

    # 4. R matrix 
    r_matrix = calculate_r_matrix(distances, radii)

    # 5. Calculate betas
    if not classification:
        coefficients = invert_matrix_regression(r_matrix, train_outputs)
    else:
        logreg = logreg_classification(r_matrix, train_outputs, l2, eta)

    """
    TODO: Obtain the distances from the centroids to the test patterns
          and obtain the R matrix for the test set
    """

    test_distances = kmeans.transform(test_inputs)
    test_r_matrix = calculate_r_matrix(test_distances, radii)

    if not classification:
        train_predictions = np.dot(r_matrix, coefficients)
        test_predictions = np.dot(test_r_matrix, coefficients)
        # print np.array_str(test_predictions, suppress_small=True)
        train_mse = sum(sum((train_outputs - train_predictions) ** 2)) / \
                    (outputs * train_predictions.shape[0])
        test_mse = sum(sum((test_outputs - test_predictions) ** 2)) / \
                   (outputs * test_predictions.shape[0])
        if outputs == 1:
            train_ccr = sum(np.around(train_predictions) == train_outputs) / \
                        float(len(train_predictions)) * 100
            test_ccr = sum(np.around(test_predictions) == test_outputs) / \
                       float(len(test_predictions)) * 100
        else:
            train_ccr = 0
            test_ccr = 0

        train_results = {
            'ccr': train_ccr,
            'mse': train_mse}

        test_results = {
            'ccr': test_ccr,
            'mse': test_mse}

    else:
        train_predictions = logreg.predict(r_matrix)
        test_predictions = logreg.predict(test_r_matrix)
        train_0_1 = (train_outputs == range(0, int(max(train_outputs)[0] + 1)))
        test_0_1 = (test_outputs == range(0, int(max(test_outputs)[0] + 1)))
        train_mse = sum(sum((train_0_1 - logreg.predict_proba(r_matrix)) ** 2)) / \
                    ((max(train_outputs) + 1) * train_outputs.shape[0])
        train_mse = sum(train_mse)
        test_mse = sum(sum((test_0_1 - logreg.predict_proba(test_r_matrix)) ** 2)) / \
                   ((max(test_outputs) + 1) * test_outputs.shape[0])
        test_mse = sum(test_mse)
        train_ccr = sum(train_predictions == train_outputs.ravel()) / \
                    float(len(train_predictions)) * 100
        test_ccr = sum(test_predictions == test_outputs.ravel()) / \
                   float(len(test_predictions)) * 100
        # np.set_printoptions( threshold=None, edgeitems=None, linewidth=None, \
        #           formatter = dict( float = lambda x: "%.3g" % x ))  # float arrays %.3g	
        # print(np.array_str(logreg.predict_proba(test_r), suppress_small=True))

        train_results = {
            'ccr': train_ccr,
            'mse': train_mse}

        test_results = {
            'ccr': test_ccr,
            'mse': test_mse}

        # Fairness evaluation
        if fairness:
            # TODO Group label (we assume it is in the last column of input data):
            train_gender_bin = np.floor(train_inputs[:, -1])
            test_gender_bin = np.floor(test_inputs[:, -1])
            # 1 women / 0 men
            train_gender_bin[train_gender_bin == -1] = 0
            test_gender_bin[test_gender_bin == -1] = 0

            metrics = {
                'false negative rate': false_negative_rate,
                'false positive rate': false_positive_rate}

            train_fm = MetricFrame(metrics=metrics,
                                   y_true=train_outputs,
                                   y_pred=train_predictions,
                                   sensitive_features=train_gender_bin)

            train_results['fairnes_metrics'] = train_fm

            test_fm = MetricFrame(metrics=metrics,
                                  y_true=test_outputs,
                                  y_pred=test_predictions,
                                  sensitive_features=test_gender_bin)

            test_results['fairnes_metrics'] = test_fm

            # train_results and test results are expected to be a MetricFrame
            return train_results, test_results

    # # # # KAGGLE # # # #
    if model_file != "":
        save_obj = {
            'classification': classification,
            'radii': radii,
            'kmeans': kmeans
        }
        if not classification:
            save_obj['coefficients'] = coefficients
        else:
            save_obj['logreg'] = logreg

        dir = os.path.dirname(model_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(model_file, 'wb') as f:
            pickle.dump(save_obj, f)

    # # # # # # # # # # #

    if not classification:
        """
        TODO: Obtain the predictions for training and test and calculate
              the MSE
        """
    else:
        """
        TODO: Obtain the predictions for training and test and calculate
              the CCR. Obtain also the MSE, but comparing the obtained
              probabilities and the target probabilities
        """

    return train_results, test_results


def read_data(train_file, test_file, outputs):
    """ Read the input data
        It receives the name of the input data file names (training and test)
        and returns the corresponding matrices

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        outputs: int
            Number of variables to be used as outputs
            (all at the end of the matrix).
              
        Returns
        -------
        train_inputs: array, shape (n_train_patterns,n_inputs)
            Matrix containing the inputs for the training patterns
        train_outputs: array, shape (n_train_patterns,n_outputs)
            Matrix containing the outputs for the training patterns
        test_inputs: array, shape (n_test_patterns,n_inputs)
            Matrix containing the inputs for the test patterns
        test_outputs: array, shape (n_test_patterns,n_outputs)
            Matrix containing the outputs for the test patterns
    """
    # Load a pandas dataframe

    print(train_file)

    train_df = pd.read_csv(train_file, header=None)

    test_df = pd.read_csv(test_file, header=None)

    # Separate inputs and outputs

    train_inputs = train_df.iloc[:, :-outputs].values
    train_outputs = train_df.iloc[:, -outputs:].values

    test_inputs = test_df.iloc[:, :-outputs].values
    test_outputs = test_df.iloc[:, -outputs:].values

    # TODO Complete the code to read the data
    return train_inputs, train_outputs, test_inputs, test_outputs


def init_centroids_classification(train_inputs, train_outputs, num_rbf):
    """ Initialize the centroids for the case of classification
        This method selects in a stratified num_rbf patterns.

        Parameters
        ----------
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        centroids: array, shape (num_rbf,n_inputs)
            Matrix with all the centroids already selected
    """

    centroids = train_test_split(train_inputs, train_outputs, stratify=train_outputs, train_size=num_rbf)[0]

    # TODO: Complete the code of the function
    assert centroids.shape[0] == num_rbf and centroids.shape[1] == train_inputs.shape[1]
    return centroids


def clustering(classification, train_inputs, train_outputs, num_rbf):
    """ It applies the clustering process
        A clustering process is applied to set the centers of the RBFs.
        In the case of classification, the initial centroids are set
        using the method init_centroids_classification(). 
        In the case of regression, the centroids have to be set randomly.

        Parameters
        ----------
        classification: bool
            True if it is a classification problem
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        kmeans: sklearn.cluster.KMeans
            KMeans object after the clustering
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        centers: array, shape (num_rbf,n_inputs)
            Centers after the clustering
    """
    if classification:
        centroids = init_centroids_classification(train_inputs, train_outputs, num_rbf)
        kmeans = KMeans(n_clusters=num_rbf, init=centroids, n_init=1, max_iter=500).fit(train_inputs, train_outputs)
    else:
        kmeans = KMeans(n_clusters=num_rbf, init='random', n_init=1, max_iter=500).fit(train_inputs, train_outputs)

    distances = kmeans.transform(train_inputs)
    centers = kmeans.cluster_centers_

    # TODO: Complete the code of the function
    return kmeans, distances, centers


def calculate_radii(centers, num_rbf):
    """ It obtains the value of the radii after clustering
        This methods is used to heuristically obtain the radii of the RBFs
        based on the centers

        Parameters
        ----------
        centers: array, shape (num_rbf,n_inputs)
            Centers from which obtain the radii
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
    """
    """ Classic version
    radii = np.zeros(num_rbf)

    for index, center_x in enumerate(centers):
        distance_metric = sum(np.sqrt(np.sum((center_x - center_y)) ** 2)
                              for center_y in centers
                              if center_y is not center_x)

        radii[index] = 1 / (2 * (num_rbf - 1)) * distance_metric

    # TODO: Complete the code of the function
    # squareform pdist
    """
    center_distances = squareform(pdist(centers, 'euclidean'))
    radii = sum(center_distances) / (2.0 * (num_rbf - 1))
    return radii


def out_rbf(distances, radii):
    """ It obtains the output of the RBFs
    """
    return np.exp((-1 / 2) * ((np.power(distances, 2)) / np.power(radii, 2)))


def calculate_r_matrix(distances, radii):
    """ It obtains the R matrix
        This method obtains the R matrix (as explained in the slides),
        which contains the activation of each RBF for each pattern, including
        a final column with ones, to simulate bias
        
        Parameters
        ----------
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
            
        Returns
        -------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
    """
    """ Classic version
    r_matrix = np.zeros((distances.shape[0], distances.shape[1] + 1))

    for i in range(r_matrix.shape[0]):
        for j in range(r_matrix.shape[1] - 1):
            r_matrix[i, j] = out_rbf(distances[i, j], radii[j])

        # Bias
        r_matrix[i, -1] = 1

    # TODO: Complete the code of the function
    """
    r_matrix = np.exp(-((distances ** 2) / (2 * (np.tile(radii, (distances.shape[0], 1)) ** 2))))
    r_matrix = np.concatenate((np.ones((distances.shape[0], 1)), r_matrix), axis=1)
    assert r_matrix.shape[0] == distances.shape[0] and r_matrix.shape[1] == distances.shape[1] + 1
    return r_matrix


def invert_matrix_regression(r_matrix, train_outputs):
    """ Inversion of the matrix for regression case
        This method obtains the pseudoinverse of the r matrix and multiplies
        it by the targets to obtain the coefficients in the case of linear
        regression
        
        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
              
        Returns
        -------
        coefficients: array, shape (n_outputs,num_rbf+1)
            For every output, values of the coefficients for each RBF and value 
            of the bias 
    """

    # IMPORTANT!! WE MUST USE THE PSEUDO INVERSE OF THE R MATRIX CALCULATED VIA NUMPY

    coefficients = np.zeros((train_outputs.shape[1], r_matrix.shape[1]))

    r_matrix_pseudo_invert = pinv((np.transpose(r_matrix) @ r_matrix)) @ np.transpose(r_matrix)
    # sum_diadong = sum_diagon * 10e-6, its like rcon param, for add values into the diagonal

    coefficients = r_matrix_pseudo_invert @ train_outputs

    # TODO: Complete the code of the function
    # assert coefficients.shape[0] == train_outputs.shape[1] and coefficients.shape[1] == r_matrix.shape[1]
    return coefficients


def logreg_classification(matriz_r, train_outputs, l2, eta):
    """ Performs logistic regression training for the classification case
        It trains a logistic regression object to perform classification based
        on the R matrix (activations of the RBFs together with the bias)
        
        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
              
        Returns
        -------
        logreg: sklearn.linear_model.LogisticRegression
            Scikit-learn logistic regression model already trained
    """

    # Max_iter must be increased if we use 1 / eta. Verbose must be disabled
    # https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
    logreg = LogisticRegression(penalty=('l2' if l2 else 'l1'), C=(1.0 / eta), solver='liblinear', verbose=False,
                                fit_intercept=False,  # Max iter by default to 100
                                max_iter=1000).fit(
        matriz_r,
        train_outputs[:, 0])

    # TODO: Complete the code of the function
    return logreg


def predict(test_file, model_file):
    """ Performs a prediction with RBFNN model
        It obtains the predictions of a RBFNN model for a test file, using two files, one
        with the test data and one with the model

        Parameters
        ----------
        test_file: string
            Name of the test file
        model_file: string
            Name of the file containing the model data

        Returns
        -------
        test_predictions: array, shape (n_test_patterns,n_outputs)
            Predictions obtained with the model and the test file inputs
    """
    test_df = pd.read_csv(test_file, header=None)
    test_inputs = test_df.values[:, :]

    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)

    radii = saved_data['radii']
    classification = saved_data['classification']
    kmeans = saved_data['kmeans']

    test_distancias = kmeans.transform(test_inputs)
    test_r = calculate_r_matrix(test_distancias, radii)

    if classification:
        logreg = saved_data['logreg']
        test_predictions = logreg.predict(test_r)
    else:
        coeficientes = saved_data['coefficients']
        test_predictions = np.dot(test_r, coeficientes)

    return test_predictions


if __name__ == "__main__":
    train_rbf_total()
