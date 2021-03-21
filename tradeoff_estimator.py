# Load all necessary packages
import sys
import numpy as np
import pandas as pd

sys.path.append("../")
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
                import load_preproc_data_adult, load_preproc_data_compas

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from variable_cep import CalibratedEqOddsPostprocessing #modified for varying weight
from variable_cep import normed_rates
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

## import dataset
dataset_used = "compas" # "adult", "german", "compas"
protected_attribute_used = 2 # 1, 2

# code to identify the protected attributes from all of the dataset features
if dataset_used == "adult":
    dataset_orig = AdultDataset()
#     dataset_orig = load_preproc_data_adult()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
    
elif dataset_used == "german":
    dataset_orig = GermanDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
    
elif dataset_used == "compas":
#     dataset_orig = CompasDataset()
    dataset_orig = load_preproc_data_compas()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]  
  
#random seed for calibrated equal odds prediction
randseed = 12345679 

#train validation/test split
dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True)

# Placeholder for predicted and transformed datasets
dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)

# Logistic regression classifier and predictions for training data
scale_orig = StandardScaler()
X_train = scale_orig.fit_transform(dataset_orig_train.features)
y_train = dataset_orig_train.labels.ravel()
lmod = LogisticRegression() #logregression
lmod.fit(X_train, y_train)

fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
y_train_pred_prob = lmod.predict_proba(X_train)[:,fav_idx]

# Prediction probs for training data

class_thresh = 0.5
dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1,1)

y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
dataset_orig_train_pred.labels = y_train_pred




#
#
#
# set up tradeoff cost-benefit calculation
include = False
N_reps = 50
N_values = 500

pbar = tqdm(total=(N_reps*N_values))

negs = []
accs = []
fps = []
fns = []

privileged_options = [True,False,None]

if include == True:
    n_range = np.linspace(0.00,1.00,N_values)
if include == False:
    n_range = np.linspace(0.01,0.99,N_values)

for neg in n_range:
    negative_val = neg
    positive_val = 1.0 - negative_val

    NP = (negative_val,positive_val)

    #should be able to refit
    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                            unprivileged_groups = unprivileged_groups,
                                            cost_constraint='specified',
                                            seed=randseed,NP_rate=NP)


    normed_p,normed_n = normed_rates(NP[1],NP[0])

    gfnr, gfpr, acc = np.zeros(3),np.zeros(3),np.zeros(3)
    
    split = False
        
    for repeat in range(N_reps):

        if (N_reps == 1) and (split == True):
            #If there's only 1 repeat then we use the same validation/test split for each
            pass
        else:
            ##########

            # New Validation/test set reshuffle and prediction for each

            dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)#validation_test split
            dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

            dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
            dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

            X_valid = scale_orig.transform(dataset_orig_valid.features)
            y_valid_pred_prob = lmod.predict_proba(X_valid)[:,fav_idx]

            X_test = scale_orig.transform(dataset_orig_test.features)
            y_test_pred_prob = lmod.predict_proba(X_test)[:,fav_idx]

            dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1,1)
            dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1,1)

            y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
            y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
            y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
            dataset_orig_valid_pred.labels = y_valid_pred
                
            y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
            y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
            y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
            dataset_orig_test_pred.labels = y_test_pred
            split=True

        # Odds equalizing post-processing algorithm

        ##########


        # Learn parameters to equalize odds and apply to create a new dataset
        cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)

        dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

        cm_transf_valid = ClassificationMetric(dataset_orig_valid, dataset_transf_valid_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

        cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

        #cm_transf_test.difference
        
        for idx,PR in enumerate(privileged_options):
            gfnr[idx] += cm_transf_test.false_negative_rate(privileged=PR)
            gfpr[idx] += cm_transf_test.false_positive_rate(privileged=PR)
            result = cm_transf_test.accuracy(privileged=PR)
            acc[idx] += float(result)

        pbar.update(1)

    fns.append(gfnr/N_reps)
    fps.append(gfpr/N_reps)
    accs.append(acc/N_reps)
    negs.append(neg)


collapse = lambda param, idx : [v[idx] for v in param]

getnames = {None:"full data", True:"privileged", False:"unprivileged"}

for idx,PR in enumerate(privileged_options):
    plt.figure()
    plt.plot(negs,collapse(accs,idx),label='accuracy')
    plt.plot(negs,collapse(fns,idx),label='false negative rate')
    plt.plot(negs,collapse(fps,idx),label='false positive rate')
    plt.xlabel('unnormalized fn rate cost')

    plt.legend()
    plt.savefig('compas1d {}.png'.format(getnames[PR]))

    plt.figure()
    plt.plot(negs,collapse(accs,idx),label='accuracy')
    plt.xlabel('unnormalized fn rate cost')
    plt.legend()
    plt.savefig('compas1d_acc {}.png'.format(getnames[PR]))