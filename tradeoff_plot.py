import numpy as np
from collections import namedtuple
from tqdm import tqdm

import matplotlib.pyplot as plt


class Model(namedtuple('Model', 'pred label')):
    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label

    def calib_eq_odds(self, other, fp_rate, fn_rate, mix_rates=None):
        if fn_rate == 0:
            self_cost = self.fp_cost()
            other_cost = other.fp_cost()
            #print(self_cost, other_cost)
            self_trivial_cost = self.trivial().fp_cost()
            other_trivial_cost = other.trivial().fp_cost()
        elif fp_rate == 0:
            self_cost = self.fn_cost()
            other_cost = other.fn_cost()
            self_trivial_cost = self.trivial().fn_cost()
            other_trivial_cost = other.trivial().fn_cost()
        else:
            self_cost = self.weighted_cost(fp_rate, fn_rate)
            other_cost = other.weighted_cost(fp_rate, fn_rate)
            self_trivial_cost = self.trivial().weighted_cost(fp_rate, fn_rate)
            other_trivial_cost = other.trivial().weighted_cost(fp_rate, fn_rate)
        
        other_costs_more = other_cost > self_cost
        self_mix_rate = (other_cost - self_cost) / (self_trivial_cost - self_cost) if other_costs_more else 0
        other_mix_rate = 0 if other_costs_more else (self_cost - other_cost) / (other_trivial_cost - other_cost)

        # New classifiers
        self_indices = np.random.permutation(len(self.pred))[:int(self_mix_rate * len(self.pred))]
        self_new_pred = self.pred.copy()
        self_new_pred[self_indices] = self.base_rate()
        calib_eq_odds_self = Model(self_new_pred, self.label)

        other_indices = np.random.permutation(len(other.pred))[:int(other_mix_rate * len(other.pred))]
        other_new_pred = other.pred.copy()
        other_new_pred[other_indices] = other.base_rate()
        calib_eq_odds_other = Model(other_new_pred, other.label)

        if mix_rates is None:
            return calib_eq_odds_self, calib_eq_odds_other, (self_mix_rate, other_mix_rate)
        else:
            return calib_eq_odds_self, calib_eq_odds_other

    def trivial(self):
        """
        Given a classifier, produces the trivial classifier
        (i.e. a model that just returns the base rate for every prediction)
        """
        base_rate = self.base_rate()
        pred = np.ones(len(self.pred)) * base_rate
        return Model(pred, self.label)

    def weighted_cost(self, fp_rate, fn_rate):
        """
        Returns the weighted cost
        If fp_rate = 1 and fn_rate = 0, returns self.fp_cost
        If fp_rate = 0 and fn_rate = 1, returns self.fn_cost
        If fp_rate and fn_rate are nonzero, returns fp_rate * self.fp_cost * (1 - self.base_rate) +
            fn_rate * self.fn_cost * self.base_rate
        """
        norm_const = float(fp_rate + fn_rate) if (fp_rate != 0 and fn_rate != 0) else 1
        res = fp_rate / norm_const * self.fp_cost() * (1 - self.base_rate()) + \
            fn_rate / norm_const * self.fn_cost() * self.base_rate()
        return res

    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])


def tradeoff(fn_rate,fp_rate,property,dataset,use_test = True):
    data_filename = "data\{}.csv".format(dataset)
    test_and_val_data = pd.read_csv(data_filename)

    # Randomly split the data into two sets - one for computing the fairness constants
    order = np.random.permutation(len(test_and_val_data))
    val_indices = order[0::2]
    test_indices = order[1::2]
    val_data = test_and_val_data.iloc[val_indices]
    test_data = test_and_val_data.iloc[test_indices]

    # Create model objects - one for each group, validation and test
    group_0_val_data = val_data[val_data['group'] == 0]
    group_1_val_data = val_data[val_data['group'] == 1]
    group_0_test_data = test_data[test_data['group'] == 0]
    group_1_test_data = test_data[test_data['group'] == 1]

    group_0_val_model = Model(group_0_val_data['prediction'].to_numpy(), group_0_val_data['label'].to_numpy())
    group_1_val_model = Model(group_1_val_data['prediction'].to_numpy(), group_1_val_data['label'].to_numpy())
    group_0_test_model = Model(group_0_test_data['prediction'].to_numpy(), group_0_test_data['label'].to_numpy())
    group_1_test_model = Model(group_1_test_data['prediction'].to_numpy(), group_1_test_data['label'].to_numpy())

    # Find mixing rates for equalized odds models

    #print(fp_rate,fn_rate)
    #print(group_0_val_model)
    #print(group_1_val_model)

    # Apply the mixing rates to the test models
    """
    # Print results on test model
    print('Original group 0 model:\n%s\n' % repr(group_0_test_model))
    print('Original group 1 model:\n%s\n' % repr(group_1_test_model))
    print('Equalized odds group 0 model:\n%s\n' % repr(calib_eq_odds_group_0_test_model))
    print('Equalized odds group 1 model:\n%s\n' % repr(calib_eq_odds_group_1_test_model))
    """
    #group_0_acc = calib_eq_odds_group_0_test_model.accuracy()
    #group_1_acc = calib_eq_odds_group_1_test_model.accuracy()
    

    if use_test is False:
        #note that Validation accuracy does NOT depend on fp_rate,fn_rate - this is only useful for finding out how accurate the model is by default
        group_0_acc = getattr(group_0_val_model,property)()
        group_1_acc = getattr(group_1_val_model,property)()
    elif use_test is True:
        _, _, mix_rates = Model.calib_eq_odds(group_0_val_model, group_1_val_model, fp_rate, fn_rate)
        calib_eq_odds_group_0_test_model, calib_eq_odds_group_1_test_model = Model.calib_eq_odds(group_0_test_model,
                                                                                            group_1_test_model,
                                                                                            fp_rate, fn_rate,
                                                                                            mix_rates)
        group_0_acc = getattr(calib_eq_odds_group_0_test_model,property)()
        group_1_acc = getattr(calib_eq_odds_group_1_test_model,property)()

    return group_0_acc, group_1_acc

def plot_constraint(result,ax,title):
    ax.set_xlabel("fn_rate")
    ax.set_ylabel("fp_rate")
    ax.set_title(title)

    im = ax.imshow(result,extent=[minv,maxv,minv,maxv])
    return im

def constraintplot(result,step,dataset,redos,usetest=True):
    x,y = np.linspace(minv,maxv,step),np.linspace(minv,maxv,step)

    if usetest == True:
        titl = dataset+" test " + result
    else:
        titl = dataset+" val " + result

    result0 = np.zeros((step,step))
    result1 = np.zeros((step,step))

    pbar = tqdm(total=int(redos * step**2),ncols=100)
    pbar.set_description(titl)

    for idy,fp in enumerate(y):
        for idx,fn in enumerate(x):
            g0,g1 = [],[]
            for redo in range(redos):
                group_0,group_1 = tradeoff(fn,fp,result,dataset,usetest)
                g0.append(group_0)
                g1.append(group_1)
                pbar.update(1)
            #print(fn,fp)
            #print([round(g,3) for g in g0])
            #print([round(g,3) for g in g1])
            #print(np.mean(g0),np.std(g0))
            result0[idx,idy] = float(np.mean(g0))
            result1[idx,idy] = float(np.mean(g1))
            #print(group_0,idx,idy,fn,fp)

    fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,figsize=(20,20))

    fig.tight_layout()


    im1 = plot_constraint(result0,ax1,titl+"_0")
    im2 = plot_constraint(result1,ax2,titl+ "_1")

    fig.tight_layout()

    fig.colorbar(im1,label=titl+"_0")
    fig.colorbar(im2,label=titl+"_1")

    savestr = str(titl)+str(step)+"_"+str(redos) +".png"

    plt.savefig(savestr)

if __name__ == '__main__':
    import pandas as pd
    import sys
    #sets the minimum and maximum values for each 
    minv = 0.01
    maxv = 0.99
    #Note that setting either value to 0 leads to confusing results as special cases are invoked in original code

    sample_number = 100 #number of samples to take
    repeats = 1 #repeats per value for cost function
    csv_name = 'criminal_recidivism'

    #constraints = ["accuracy", "precision", "fpr", "fnr","base_rate"]
    constraints = ["accuracy","base_rate"]

    for c in constraints:
        result0 = []
        result1 = []
        for count in tqdm(range(100)):
            r_0,r_1 = tradeoff(None,None,c,csv_name,use_test=False)
            result0.append(r_0)
            result1.append(r_1)
        
        result0 = round(np.mean(result0),3)
        result1 = round(np.mean(result1),3)

        print("Group 0 Validation set mean {}: {}".format(c,result0))
        print("Group 1 Validation set mean {}: {}".format(c,result1))

    for constraint in constraints:
        constraintplot(constraint,sample_number,csv_name,repeats)