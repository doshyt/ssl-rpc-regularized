from __future__ import division
import matplotlib.pyplot as plt
import random, time
import math
import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from numpy import linalg
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle
import argparse
import logging
from random import randint
from sklearn.cross_validation import train_test_split

# this parameter instructs RPC to stop fitting if the difference between
# the new iteration error rate and the old one is this big
_EXIT_ERROR_THRESHOLD = 0.5
# the number of points in the region of uncertainty when
# the model stops iterating
_MIN_BETA_LENGTH = 5

class RPC_Classifier:
    """
    main class implementing RPC classifier
    """

    def __init__(self, T1=np.array(()), T1Labels=np.array(()), T2=np.array(()), proto_init_type='dataset', loglevel='info', lmbd=1000, iter_num = 10):
        self.T1 = T1
        self.T2 = T2
        self.T1Labels = T1Labels
        self.T2Labels = np.zeros(len(T2))
        self.log = self.set_logging(loglevel)
        self.proto_init_type = proto_init_type
        self.alpha, self.W_LABELS = self.initialize_prototypes(self.proto_init_type)
        self.D = []
        self.D2 = []
        self.fig = 0
        self.Beta = []
        self.iter_count = 0
        self.lmbd = lmbd
        self.iter_num = iter_num

    def set_logging(self, loglevel):
        """
        set RPC logger
        """
        logging.basicConfig(format="%(levelname)s:%(message)s")
        numeric_level = getattr(logging, loglevel.upper(), None)
        logger = logging.getLogger('RPC_Logger')
        logger.setLevel(numeric_level)
        return logger

    def get_pairwise_euc(self):
        """
        pairwise distance matrix calculation function
        """
        D = squareform(pdist(self.T1, metric='euclidean'))
        return D

    def get_pairwise_euc_new(self):
        """
        helper function to calculate pairwise distances matrix for new data
        """
        lengd = len(self.T1)
        lengn = len(self.T2)
        D=np.zeros(shape=(lengn, lengd))
        for j in range(lengn):
            for i in range(lengd):
                euc_sum = 0
                for k in range(len(self.T1[i])):
                    #euclidean sum
                    euc_sum += (self.T2[j][k] - self.T1[i][k])**2
                D[j][i] = math.sqrt(euc_sum)
        return D

    def Fun(self, arg):
        """
        helper function calculating F for the model
        """
        return (1+math.exp(-arg))**(-1)

    def Fun_d(self, arg):
        """
        helper function calculating derivative F' for the model
        """
        if np.abs(arg)>100:
            # this tweaking is made to deal with exp() when arg > 100
            # otherwise it leads to infinity
            arg = 100
        return math.exp(-arg)/(math.exp(-arg)+1)**2

    def alpha_prot_plus(self, i):
        """
        helper function to calculate distance to the closest (+) prototype
        """
        a_plus = np.zeros(len(self.D))
        dist_temp = []
        plus_indexes = []
        for j in range (0, len(self.alpha)):
            if(self.T1Labels[i] == self.W_LABELS[j]):
                dist_temp.append(self.distance_toprot(self.alpha[j], i))
                plus_indexes.append(j)
        sorted_distances_toprot = np.array(dist_temp).argsort(axis=0)
        ss = plus_indexes[sorted_distances_toprot.argmin()]
        a_plus = self.alpha[ss]
        return a_plus, ss

    def alpha_prot_minus(self, i):
        """
        helper function to calculate distance to the closest (-) prototype
        """
        a_minus = np.zeros(len(self.D))
        dist_temp = []
        minus_indexes = []
        for j in range (0, len(self.alpha)):
            if(self.T1Labels[i] != self.W_LABELS[j]):
                dist_temp.append(self.distance_toprot(self.alpha[j], i))
                minus_indexes.append(j)
        sorted_distances_toprot = np.array(dist_temp).argsort(axis=0)
        ss = minus_indexes[sorted_distances_toprot.argmin()]
        a_minus = self.alpha[ss]
        return a_minus, ss

    def alpha_prot_plus_new(self, DATA_LABEL, i, Dx):
        """
        helper function to calculate distance to the closest (+) prototype for new data
        """
        a_plus = np.zeros(len(self.D))
        dist_temp = []
        plus_indexes = []
        for j in range (0,len(self.alpha)):
            if(DATA_LABEL == self.W_LABELS[j]):
                dist_temp.append(self.distance_toprot_new(self.alpha[j], Dx))
                plus_indexes.append(j)
        sorted_distances_toprot = np.array(dist_temp).argsort(axis=0)
        ss = plus_indexes[sorted_distances_toprot.argmin()]
        a_plus = self.alpha[ss]
        return a_plus, ss


    def alpha_prot_minus_new(self, DATA_LABEL, i, Dx):
        """
        helper function to calculate distance to the closest (-) prototype for new data
        """
        a_minus = np.zeros(len(self.D))
        dist_temp = []
        minus_indexes = []
        for j in range (0, len(self.alpha)):
            if(DATA_LABEL != self.W_LABELS[j]):
                dist_temp.append(self.distance_toprot_new(self.alpha[j], Dx))
                minus_indexes.append(j)
        sorted_distances_toprot = np.array(dist_temp).argsort(axis=0)
        ss = minus_indexes[sorted_distances_toprot.argmin()]
        a_minus = self.alpha[ss]
        return a_minus, ss


    def distance_toprot(self, alpha_row, i):
        """
        function calculating distance to prototype
        """
        return np.dot(self.D, alpha_row)[i] - np.dot(np.dot(0.5*alpha_row.T, self.D), alpha_row)


    def distance_toprot_new(self, alpha_row, Dx):
        """
        function calculating distance to prototype on unseen data dissimilarity matrix
        """
        return np.dot(Dx.T, alpha_row) - np.dot(np.dot(0.5*alpha_row.T, self.D), alpha_row)


    def fit_RPC(self):
        """
        RPC fitting function
        """
        self.log.debug("Training RPC")
        E = 0
        E_best = 100000
        E_old = 100000
        rec_num = len(self.D)
        errs = []
        for iter in range(self.iter_num):
            E = 0
            alpha_old = self.alpha.copy()
            for i in range(rec_num):
                alpha_plus, plus_index = self.alpha_prot_plus(i)
                alpha_minus, minus_index = self.alpha_prot_minus(i)
                dp_aplus = self.distance_toprot(alpha_plus, i)
                dp_aminus = self.distance_toprot(alpha_minus, i)
                mu_v = (dp_aplus - dp_aminus)/(dp_aplus + dp_aminus)
                mu_v_plus = 2 * dp_aminus/(dp_aplus + dp_aminus)**2
                mu_v_minus = 2 * dp_aplus/(dp_aplus + dp_aminus)**2
                d_alpha_plus = np.zeros(rec_num)
                d_alpha_minus = np.zeros(rec_num)
                mu_v_Fun_d = self.Fun_d(mu_v)
                for kk in range(rec_num):
                    d_alpha_plus[kk] = -mu_v_Fun_d * mu_v_plus * (self.D[i][kk] - (sum([self.D[l][kk]*alpha_plus[l] for l in range(rec_num)])))
                    d_alpha_minus[kk]= mu_v_Fun_d * mu_v_minus * (self.D[i][kk] - (sum([self.D[l][kk]*alpha_minus[l] for l in range(rec_num)])))
                alpha_plus += d_alpha_plus/self.lmbd
                alpha_minus += d_alpha_minus/self.lmbd

                self.alpha[plus_index] = alpha_plus.copy()
                self.alpha[minus_index] = alpha_minus.copy()

            for kk in range(len(self.alpha)):
                self.alpha[kk] = self.alpha[kk].copy()/sum(self.alpha[kk])

            E = 0
            for ii in range(rec_num):
                alpha_plus, _ = self.alpha_prot_plus(ii)
                alpha_minus, _ = self.alpha_prot_minus(ii)
                dp_aplus = self.distance_toprot(alpha_plus, ii)
                dp_aminus = self.distance_toprot(alpha_minus, ii)
                E += (self.Fun((dp_aplus - dp_aminus)/(dp_aplus + dp_aminus)))
            self.log.debug("E = %s" % E)

            if ((E-E_old) > _EXIT_ERROR_THRESHOLD):
                self.log.debug("Breaking after %s iterations" % iter)
                self.alpha = alpha_best.copy()
                break
            else:
                E_old = E
                if E < E_best:
                    alpha_best = self.alpha.copy()
                    E_best = E
                errs.append(E)
        self.log.debug("RPC training done after total %s iterations" % iter)

        plt.figure(25)
        plt.plot(errs)
        w = []
        plt.figure(self.fig)
        candidate_row = np.array(self.T1)
        colors = cycle('rbm')
        for k, col in zip(range(3), colors):
            my_members = self.T1Labels == k
            plt.plot(candidate_row[my_members, 0], candidate_row[my_members, 1], col + '.')

        colors = ['r','b','m']
        # plot prototype positions
        # each prototype marker is scaled basing on how long ago it was created
        for lab in range(len(np.unique(self.W_LABELS))):
            w = []
            w_count = 0
            for i in range(len(self.alpha)):
                if lab == self.W_LABELS[i]:
                    w.append(np.dot(self.alpha[i], self.T1))
                    #plt.scatter(w[w_count][0], w[w_count][1], c = colors[int(self.W_LABELS[i])], marker = 'o', s = 15*3**w_count, alpha=0.7)
                    w_count += 1

        self.fig += 1


    def conformal_prediction(self):
        """
        conformal prediction for RPC
        """
        self.log.debug("Conformal prediction")
        rl = []
        alpha_plus = []
        alpha_minus = []
        self.Beta = []
        for N in range(len(self.T2)):
            rr = np.zeros(len(np.unique(self.T1Labels)), dtype=float)
            for l in range(len(np.unique(self.T1Labels))):
            #N - index in T2
                alpha_plus, plus_index = self.alpha_prot_plus_new(l, N, self.D2[N])
                alpha_minus, minus_index = self.alpha_prot_minus_new(l, N, self.D2[N])
                n_mi = self.distance_toprot_new(alpha_plus, self.D2[N])/self.distance_toprot_new(alpha_minus, self.D2[N])
                N_len = 0.0
                for i in range(len(self.T1)):
                    #compute mu_i for T1 and remeber it
                    alpha_plus, plus_index = self.alpha_prot_plus(i)
                    alpha_minus, minus_index = self.alpha_prot_minus(i)
                    t1_mi = self.distance_toprot(alpha_plus, i)/self.distance_toprot(alpha_minus, i)
                    if ((t1_mi) >= (n_mi)):
                        N_len += 1.0
                #N_len +=1.0
                rr[l] = float(N_len)/float(len(self.T1)+1)
            rl.append([rr])
            label_index = np.argsort(rr)[len(np.unique(self.T1Labels))-1]
            point_label = np.unique(self.T1Labels)[label_index]
            self.T2Labels[N] = point_label.copy()
            sorted_rr = np.sort(rr)[::-1]
            conf = 1 - sorted_rr[1]
            cred = sorted_rr[0]
            if((conf <= (1 - 1/(len(self.T1)))) or (cred <= 1/(len(self.T1)))) and (self.T2[N].tolist() not in self.T1.tolist()):
                self.Beta.append(self.T2[N])

    def get_new_prototypes_T1_only(self, this_cluster_beta, this_cluster_label):
        """
        method for getting new prototypes
        """
        self.log.debug("Getting new prototypes from T1")
        data_dim = self.T1[0].shape[0]
        alphas_of_betas = []
        beta_median = []

        for i in range(data_dim):
            median_index = 0
            temp_list = []
            element_index = 0
            for j in range(len(this_cluster_beta)):
                temp_list.append(this_cluster_beta[j][i])
            #odd number case
            if ((len(temp_list)%2) == 1):
                index_of_mid_el = (len(temp_list)-1)/2
                sorted_temp_list_indexes = np.argsort(np.array(temp_list).argsort(axis=0))

                for k in range(len(sorted_temp_list_indexes)):
                    if (sorted_temp_list_indexes[k] == index_of_mid_el):
                        element_index = k
            # even number case - get two middle elements
            # return 1 to the closest to the real median
            else:
                index_of_mid_el = len(temp_list)/2 - 1
                sorted_temp_list_indexes = np.argsort(np.array(temp_list).argsort(axis=0))

                for k in range(len(sorted_temp_list_indexes)):
                    if (sorted_temp_list_indexes[k] == int(index_of_mid_el)):
                        element_index_1 = k

                    if (sorted_temp_list_indexes[k] == (int(index_of_mid_el)+1)):
                        element_index_2 = k
                the_goal_median = np.median(temp_list)

                if (np.abs(the_goal_median - temp_list[element_index_1]) <= np.abs(the_goal_median - temp_list[element_index_2])):
                    element_index = element_index_1
                else:
                    element_index = element_index_2

            beta_median.append(temp_list[element_index])
            temp_list = np.zeros(len(temp_list))
            temp_list[element_index] = 1
            alphas_of_betas.append(temp_list)

        label_new_prot = this_cluster_label
        self.T1 = np.append(self.T1, this_cluster_beta, axis=0)

        for i in range(len(this_cluster_beta)):
            self.T1Labels = np.append(self.T1Labels, label_new_prot)

        for i in range(data_dim):
            self.W_LABELS= np.append(self.W_LABELS, label_new_prot)

        bottom_zeros = np.zeros((data_dim, self.alpha.shape[1]))
        right_zeros = np.zeros((self.alpha.shape[0], len(this_cluster_beta)))
        self.alpha = np.concatenate((self.alpha, bottom_zeros))
        right_zeros = np.concatenate((right_zeros, np.array(alphas_of_betas)))
        self.alpha = np.concatenate((self.alpha, right_zeros),axis=1)

    def initialize_prototypes(self, proto_init_type):
        """
        this method enables different techniques for initializing prototypes
        """
        num_clusters = len(np.unique(self.T1Labels))
        rec_num = self.T1.shape[0]
        _alpha = []
        _W_LABELS = []
        self.log.debug("Prototype init type: %s" % proto_init_type)
        if proto_init_type == 'random':
            for counter in range(num_clusters):
                r = [random.random() for i in range(rec_num)]
                s = sum(r)
                r = [ i/s for i in r ]
                _alpha.append(r)
                _W_LABELS.append(counter)

        # NN-initialization of initial prototypes
        if proto_init_type == 'NN':
            flag = True
            while flag == True:
                    no_conv = False
                    n_neighbors = 3
                    clf = KNeighborsClassifier(n_neighbors)
                    clf.fit(self.T1, self.T1Labels)
                    _W_LABELS = []
                    while len(np.unique(_W_LABELS)) != len(np.unique(self.T1Labels)):
                        _W_LABELS = []
                        for counter in range(num_clusters):
                            s = 0
                            r = [random.uniform(-1,1) for i in range(0, rec_num)]
                            s = sum(r)
                            r = [ i/s for i in r ]
                            _alpha.append(r)
                            _W_LABELS.append(clf.predict(np.dot(_alpha[counter], self.T1))[0])
                    for lab in range(0,int(np.max(_W_LABELS))+1):
                        if _W_LABELS.count(lab) == 1:
                           continue
                        else:
                            no_conv = True
                            break
                    if no_conv == False:
                        break

        # here we set initial W to a point in a cluster
        if proto_init_type == 'dataset':
            _W_LABELS = [i for i in range(num_clusters)]
            for k in range(len(_W_LABELS)):
                _alpha.append(np.zeros(len(self.T1)))
                for i in range(len(self.T1Labels)):
                    if self.T1Labels[i] == k:
                        _alpha[k][i] = 1.0
                        break

        self.log.debug("W_LABELS:\n%s" % _W_LABELS)
        self.log.debug(_alpha)
        return np.array(_alpha), np.array(_W_LABELS)


    def nearest_n(self, new_prot):
        """
        helper to find closest point label frrom T1
        """
        min_label = 0
        min = 1000
        for i in range(len(self.T1)):
            dst = distance.euclidean(new_prot, self.T1[i])
            if(dst < min):
                min = dst
                min_label = self.T1Labels[i]
        return min_label


    def model_do_fitting(self):
        """
        Scikit-learn notation for fitting
        """
        self.RPC_iteration()

    def model_do_predicting(self):
        """
        Scikit-learn notation for predicting
        """
        s_index_best = -1
        _alpha_best = []
        _W_LABELS_BEST = []
        _T1_best = []
        _T1_LABELS_BEST = []
        _T2LabelsBest = self.T2Labels.copy()
        while (len(self.Beta)>_MIN_BETA_LENGTH) and (self.iter_count<self.iter_num):
            clusters_list = []
            for i in range(len(self.Beta)):
                clusters_list = np.append(clusters_list, self.nearest_n(self.Beta[i]))
            for ii in np.unique(clusters_list):
                beta_of_this_cluster = []
                for iii in range(len(self.Beta)):
                    if clusters_list[iii] == ii:
                        beta_of_this_cluster.append(self.Beta[iii])
                this_cluster_label = ii
                self.get_new_prototypes_T1_only(beta_of_this_cluster, this_cluster_label)
            self.log.info("%s iteration alpha" % str(self.iter_count))
            self.RPC_iteration()
            s_index = metrics.silhouette_score(np.concatenate((self.T1, self.T2), axis=0), np.concatenate((self.T1Labels,self.T2Labels),axis=0), metric='euclidean')
            self.log.debug("Silhouette index = %s" % s_index)
            self.log.debug("T2 Predicted Labels:\n%s" % self.T2Labels)
            if s_index > s_index_best:
                _T2LabelsBest = self.T2Labels.copy()
                _alpha_best = self.alpha
                _W_LABELS_BEST = self.W_LABELS.copy()
                _T1_best = self.T1
                _T1_LABELS_BEST = self.T1Labels
                self.log.debug("NEW BEST LABELS:\n%s" % _T2LabelsBest)
                s_index_best = s_index

        return _T1_best, _T1_LABELS_BEST, _T2LabelsBest, _alpha_best, _W_LABELS_BEST

    def fit(self, T1, T2, T1Labels):
        """
        scikit-learn notation
        """
        # initialization partially overlaps with constructor to keep
        # compatiblity to both scikit- and non-scikit-learn initializations
        self.T1 = T1
        self.T2 = T2
        self.T1Labels = T1Labels
        self.T2Labels = np.zeros(len(self.T2))
        self.alpha, self.W_LABELS = self.initialize_prototypes(self.proto_init_type)
        self.log.info("Initialized RPC classifier")
        self.log.info("Training RPC classifier")
        self.model_do_fitting()

    def predict(self):
        """
        scikit-learn notation
        """
        _, _, self.T2Labels, _, _ = self.model_do_predicting()
        return self.T2Labels

    def score(self, real_labels, score_type='accuracy'):
        """
        get assessment of the result
        """
        if score_type == 'accuracy':
            num_correct = 0
            for i in range(len(real_labels)):
                if real_labels[i] == self.T2Labels[i]:
                    num_correct += 1

            return num_correct/len(real_labels)
        else:
            return "Unknown accuracy measure"

    def RPC_iteration(self):
        """
        helper function for performing one RPC iteration
        """
        self.D = self.get_pairwise_euc()
        self.D2 = self.get_pairwise_euc_new()
        self.iter_count += 1
        self.fit_RPC()
        self.conformal_prediction()

    def model_do_training(self):
        """
        function defining main model mechanics
        """
        s_index_best = -1
        _alpha_best = []
        _W_LABELS_BEST = []
        _T1_best = []
        _T1_LABELS_BEST = []
        self.RPC_iteration()
        _T2LabelsBest = self.T2Labels.copy()
        while (len(self.Beta)>_MIN_BETA_LENGTH) and (self.iter_count<self.iter_num):
            clusters_list = []
            for i in range(len(self.Beta)):
                clusters_list = np.append(clusters_list, self.nearest_n(self.Beta[i]))
            for ii in np.unique(clusters_list):
                beta_of_this_cluster = []
                for iii in range(len(self.Beta)):
                    if clusters_list[iii] == ii:
                        beta_of_this_cluster.append(self.Beta[iii])
                this_cluster_label = ii
                self.get_new_prototypes_T1_only(beta_of_this_cluster, this_cluster_label)
            self.log.info("%s iteration alpha" % str(self.iter_count))
            self.RPC_iteration()
            s_index = metrics.silhouette_score(np.concatenate((self.T1, self.T2), axis=0), np.concatenate((self.T1Labels,self.T2Labels),axis=0), metric='euclidean')
            self.log.debug("Silhouette index = %s" % s_index)
            self.log.debug("T2 Predicted Labels:\n%s" % self.T2Labels)
            if s_index > s_index_best:
                _T2LabelsBest = self.T2Labels.copy()
                _alpha_best = self.alpha
                _W_LABELS_BEST = self.W_LABELS.copy()
                _T1_best = self.T1
                _T1_LABELS_BEST = self.T1Labels
                self.log.debug("NEW BEST LABELS:\n%s" % _T2LabelsBest)
                s_index_best = s_index

        return _T1_best, _T1_LABELS_BEST, _T2LabelsBest, _alpha_best, _W_LABELS_BEST

    def get_summary(self, T2LabelsBest):
        right_labels = 0
        for i in range(len(self.T2Labels)):
            if int(T2LabelsBest[i]==self.T2Labels[i]):
                right_labels += 1
        accuracy = right_labels/len(self.T2Labels)
        self.log.info("Training Accuracy: %s" % accuracy)
        #plt.show()
        return accuracy

    def cluster_map(self):
        """
        draw a clustermap for the report
        """
        self.log.debug("Plotting Clustermap")
        colors = cycle('rbm')
        k = len(self.T1[0])
        realT1 = self.T1
        realT2 = self.T2
        realT1Labels = self.T1Labels
        realT2Labels = self.T2Labels
        realAlpha = self.alpha
        realW_LABELS = self.W_LABELS
        realD = self.D
        realD2 = self.D2
        if k == 2:
            candidate_x = np.arange(np.min([x[0] for x in self.T1])-1, np.max([x[0] for x in self.T1])+1, 0.1)
            candidate_y = np.arange(np.min([x[1] for x in self.T1])-1, np.max([x[1] for x in self.T1])+1, 0.1)
            plt.figure(20)
            for i in range(len(candidate_x)):
                #candidate_row = []
                for j in range(len(candidate_y)):
                    candidate_row = []
                    candidate_row.append([candidate_x[i], candidate_y[j]])
                    Y_LABELS = np.zeros(len(candidate_row))
                    self.T2 = np.array(candidate_row)
                    self.T2Labels = np.zeros(len(candidate_row))
                    self.D2 = self.get_pairwise_euc_new()
                    self.conformal_prediction()
                    candidate_row = np.array(candidate_row)
                    for k, col in zip(range(3), colors):
                        my_members = self.T2Labels == k
                        plt.scatter(candidate_row[my_members, 0], candidate_row[my_members, 1], marker = '.', color = col, s = 300, alpha = 0.3)

            self.T1 = realT1
            self.T2 = realT2
            self.T1Labels = realT1Labels
            self.T2Labels = realT2Labels
            self.alpha = realAlpha
            self.W_LABELS = realW_LABELS
            self.D = realD
            self.D2 = realD2

            X = self.T1
            for k, col in zip(range(len(np.unique(self.T1Labels))), colors):
                my_members = self.T1Labels == k
                plt.scatter(X[my_members, 0], X[my_members, 1], marker = '.', color = col, s = 150, alpha=0.8)

            X = self.T2
            for k, col in zip(range(len(np.unique(self.T1Labels))), colors):
                my_members = self.T2Labels == k
                plt.scatter(X[my_members, 0], X[my_members, 1], marker = '*', color = col, s = 200, alpha=0.8)

            colors = ['r','b','m']
            # plot prototype positions
            # each prototype marker is scaled basing on how long ago it was created
            for lab in range(len(np.unique(self.W_LABELS))):
                w = []
                w_count = 0
                for i in range(len(self.alpha)):
                    if lab == self.W_LABELS[i]:
                        w.append(np.dot(self.alpha[i], self.T1))
                        plt.scatter(w[w_count][0], w[w_count][1],c = colors[int(self.W_LABELS[i])], marker = 'o', s = 10*10*i, alpha = 0.5)
                        w_count += 1
