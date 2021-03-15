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

## import dataset
dataset_used = "compas" # "adult", "german", "compas"
protected_attribute_used = 2 # 1, 2

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
  
# cost constraint of fnr will optimize generalized false negative rates, that of
# fpr will optimize generalized false positive rates, and weighted will optimize
# a weighted combination of both
#random seed for calibrated equal odds prediction
randseed = 12345679 


dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

# print out some labels, names, etc.
display(Markdown("#### Dataset shape"))
print(dataset_orig_train.features.shape)
display(Markdown("#### Favorable and unfavorable labels"))
print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
display(Markdown("#### Protected attribute names"))
print(dataset_orig_train.protected_attribute_names)
display(Markdown("#### Privileged and unprivileged protected attribute values"))
print(dataset_orig_train.privileged_protected_attributes, dataset_orig_train.unprivileged_protected_attributes)
display(Markdown("#### Dataset feature names"))
print(dataset_orig_train.feature_names)


metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

metric_orig_valid = BinaryLabelDatasetMetric(dataset_orig_valid, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Original validation dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_valid.mean_difference())

metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Original test dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

# Placeholder for predicted and transformed datasets
dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

# Logistic regression classifier and predictions for training data
scale_orig = StandardScaler()
X_train = scale_orig.fit_transform(dataset_orig_train.features)
y_train = dataset_orig_train.labels.ravel()
lmod = LogisticRegression()
lmod.fit(X_train, y_train)

fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
y_train_pred_prob = lmod.predict_proba(X_train)[:,fav_idx]

# Prediction probs for validation and testing data
X_valid = scale_orig.transform(dataset_orig_valid.features)
y_valid_pred_prob = lmod.predict_proba(X_valid)[:,fav_idx]

X_test = scale_orig.transform(dataset_orig_test.features)
y_test_pred_prob = lmod.predict_proba(X_test)[:,fav_idx]

class_thresh = 0.5
dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1,1)
dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1,1)
dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1,1)

y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
dataset_orig_train_pred.labels = y_train_pred

y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
dataset_orig_valid_pred.labels = y_valid_pred
    
y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
dataset_orig_test_pred.labels = y_test_pred


"""
cm_pred_train = ClassificationMetric(dataset_orig_train, dataset_orig_train_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Original-Predicted training dataset"))
print("Difference in GFPR between unprivileged and privileged groups")
print(cm_pred_train.difference(cm_pred_train.generalized_false_positive_rate))
print("Difference in GFNR between unprivileged and privileged groups")
print(cm_pred_train.difference(cm_pred_train.generalized_false_negative_rate))

cm_pred_valid = ClassificationMetric(dataset_orig_valid, dataset_orig_valid_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Original-Predicted validation dataset"))
print("Difference in GFPR between unprivileged and privileged groups")
print(cm_pred_valid.difference(cm_pred_valid.generalized_false_positive_rate))
print("Difference in GFNR between unprivileged and privileged groups")
print(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))

cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Original-Predicted testing dataset"))
print("Difference in GFPR between unprivileged and privileged groups")
print(cm_pred_test.difference(cm_pred_test.generalized_false_positive_rate))
print("Difference in GFNR between unprivileged and privileged groups")
print(cm_pred_test.difference(cm_pred_test.generalized_false_negative_rate))
"""


# Odds equalizing post-processing algorithm


##########



##########


N_reps = 20

negs = []
accs = []
fps = []
fns = []

for neg in tqdm(np.linspace(0.01,0.99,100)):
    negative_val = neg
    positive_val = 1.0 - negative_val

    NP = (negative_val,positive_val)
    normed_p,normed_n = normed_rates(NP[1],NP[0])

    gfnr, gfpr, acc = 0.0,0.0,0.0

    for repeat in range(N_reps):
        # Learn parameters to equalize odds and apply to create a new dataset
        cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                            unprivileged_groups = unprivileged_groups,
                                            cost_constraint='specified',
                                            seed=randseed,NP_rate=NP)
        
        cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)

        dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

        cm_transf_valid = ClassificationMetric(dataset_orig_valid, dataset_transf_valid_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

        cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

        gfnr += cm_transf_test.difference(cm_transf_test.generalized_false_negative_rate)
        gfpr += cm_transf_test.difference(cm_transf_test.generalized_false_positive_rate)
        acc += float(cm_transf_test.difference(cm_transf_test.accuracy))

    fns.append(gfnr/N_reps)
    fps.append(gfpr/N_reps)
    accs.append(acc/N_reps)
    negs.append(neg)


plt.plot(negs,accs,label='accuracy')
plt.plot(negs,fns,label='false negative rate')
plt.plot(negs,fps,label='false positive rate')
plt.xlabel('Unnormalized fn rate cost')
plt.legend()
plt.savefig('compas1d.png')




##########



##########



accs = []
fps = []
fns = []

normp,normn = [],[]

N=20
pbar = tqdm(total=N**2)
for positive_val in np.linspace(0.0,20.0,N):
    for negative_val in np.linspace(0.0,20.0,N):

        NP = (negative_val,positive_val)

        gfnr, gfpr, acc = 0.0,0.0,0.0

        normed_p,normed_n = normed_rates(NP[1],NP[0])
        normp.append(normed_p)
        normn.append(normed_n)

        # Learn parameters to equalize odds and apply to create a new dataset
        for repeat in range(N_reps):
            cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                                unprivileged_groups = unprivileged_groups,
                                                cost_constraint='specified',
                                                seed=randseed,NP_rate=NP)
            
            cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)

            dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
            dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

            cm_transf_valid = ClassificationMetric(dataset_orig_valid, dataset_transf_valid_pred,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)

            """
            display(Markdown("#### Original-Transformed validation dataset"))
            print("Difference in GFPR between unprivileged and privileged groups")
            print(cm_transf_valid.difference(cm_transf_valid.generalized_false_positive_rate))
            print("Difference in GFNR between unprivileged and privileged groups")
            print(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate))
            """
            cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)

            """
            display(Markdown("#### Original-Transformed testing dataset"))
            print("Difference in GFPR between unprivileged and privileged groups "+str(NP))
            print(cm_transf_test.difference(cm_transf_test.generalized_false_positive_rate))
            print("Difference in GFNR between unprivileged and privileged groups "+str(NP))
            print(cm_transf_test.difference(cm_transf_test.generalized_false_negative_rate))
            print("Test Acc diff:")
            print(cm_transf_test.difference(cm_transf_test.accuracy))
            """
            gfnr += cm_transf_test.difference(cm_transf_test.generalized_false_negative_rate)
            gfpr += cm_transf_test.difference(cm_transf_test.generalized_false_positive_rate)
            acc += float(cm_transf_test.difference(cm_transf_test.accuracy))

        fns.append(gfnr/N_reps)
        fps.append(gfpr/N_reps)
        accs.append(acc/N_reps)
        pbar.update(1)


ax = plt.axes(projection='3d')
ax.scatter3D(np.array(normp), np.array(normn), np.array(accs),label='Accuracy difference')
ax.scatter3D(np.array(normp), np.array(normn), np.array(fns),label='False negative difference')
ax.scatter3D(np.array(normp), np.array(normn), np.array(fps),label='False positive difference')
ax.set_xlabel("Normalized fp rate cost")
ax.set_ylabel("Normalized fn rate cost")
plt.legend()
plt.show()
plt.savefig("compass2d.png")