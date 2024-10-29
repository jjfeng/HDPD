"""
Utility functions such as for calculating risk, binning, model fitting
"""
import logging
import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.model_selection import GridSearchCV

COND_OUTCOME_STR = "Cond_Outcome"
COND_COV_STR = "Cond_Cov"
N_ESTIMATORS = 800
DEFAULT_DEPTH = 4
CV = 3
kernel_model = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LogisticRegression(penalty='l1', fit_intercept=True, solver='saga', max_iter=20000))])

class LossEvaluator:
    def __init__(self, loss_name: str, ml_mdl: BaseEstimator):
        self.loss_name = loss_name
        self.ml_mdl = ml_mdl
    
    def get_loss(self, x, y):
        if self.loss_name == "brier":
            return np.power(self.ml_mdl.predict_proba(x)[:,1] - y.flatten(), 2)
        elif self.loss_name == "accuracy":
            # Thresholding at 0.5
            pred_val = (self.ml_mdl.predict_proba(x)[:,1] > 0.5).astype(int)
            return (pred_val == y.flatten()).astype(float)

class PipelineWeightedFit(Pipeline):
    """extends Pipeline class in sklearn to pass sample_weight param during fit
    """
    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            # NOTE: assumes classifier step in pipeline is called clf
            return super().fit(X, y, **{'clf__sample_weight': sample_weight})
        else:
            return super().fit(X, y)


def to_safe_prob(prob, eps=1e-10):
    return np.maximum(eps, np.minimum(1 - eps, prob))

def convert_logit_to_prob(logit):
    return 1/(1 + np.exp(-logit))

def convert_prob_to_logit(prob, eps=1e-10):
    safe_prob = to_safe_prob(prob, eps)
    return np.log(safe_prob/(1 - safe_prob))

def get_complementary_logit(logit):
    return convert_prob_to_logit(1 - convert_logit_to_prob(logit))

def get_sigmoid_deriv(logit):
    p = convert_logit_to_prob(logit)
    return p * (1 - p)

def get_inv_sigmoid_deriv(prob):
    return 1/prob + 1/(1 - prob)

def read_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    return X, Y

def get_n_jobs():
    n_cpu = int(os.getenv('OMP_NUM_THREADS')) if os.getenv('OMP_NUM_THREADS') is not None else 0
    n_jobs = max(n_cpu - 1, 1) if n_cpu > 0 else -1
    logging.info("NUM JOBS %d", n_jobs)
    return n_jobs

def get_variance_of_sums(alist):
    """
    Returns variance of sum of independent random variables
    :param alist: list of numpy arrays each being iid observations of a variable
    """
    var = 0
    for obs in alist:
        var += 1/len(obs) * np.var(obs)
    return var

def binning_prob(prob, num_bins=10, eps=1e-10):
    """bin probabilities into bins of width 0.1, {0,0.1,0.2,0.3,...,1}
    """
    bins = np.linspace(-eps,1+eps,num_bins+1)
    centers = (bins[1:]+bins[:-1])/2
    centers = np.insert(np.insert(centers,len(centers),1),0,0)
    ind = np.digitize(prob, bins, right=True)
    assert (np.min(ind) > 0) and (np.max(ind) <= num_bins)
    return centers[ind]

def compute_risk(loss_func, X, proba):
    loss_Y1 = loss_func(X, np.ones(X.shape[0]))
    loss_Y0 = loss_func(X, np.zeros(X.shape[0]))
    assert loss_Y1.shape == proba.shape
    assert loss_Y0.shape == proba.shape
    return loss_Y1 * proba + loss_Y0 * (1 - proba)

def min_max_scale(x, abs:bool = False):
    if abs:
        x = np.abs(x)
    return (x - x.min())/(x.max() - x.min())

def get_density_model(max_feats:list[int], do_grid_search=False, model_args=None, gridsearch_polynom_lr: bool = False):
    # Probabilistic classifier for the density ratio model
    if model_args is None:
        model_args = {'max_depth': DEFAULT_DEPTH, 'max_features': None, 'n_estimators': N_ESTIMATORS}
    if do_grid_search:
        pipeline = Pipeline([
            ('clf', RandomForestClassifier())
        ])  # a dummy classifier type is needed
        parameters = [
            {
                'clf': (RandomForestClassifier(
                    max_depth=model_args['max_depth'],
                    criterion='log_loss',
                    max_features=model_args['max_features'],
                    n_estimators=model_args['n_estimators'],
                    n_jobs=get_n_jobs(),
                    random_state=0),),
                'clf__max_depth': [4,6],
                'clf__max_features': max_feats
            },
        ]
        if gridsearch_polynom_lr:
            kernel_model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LogisticRegression(penalty='l2', fit_intercept=True, max_iter=20000, solver="saga"))])
            parameters += [{
                'clf': [kernel_model],
                'clf__linear__penalty': ['l1', 'l2'],
                'clf__linear__C': [100,10,1,0.1],
            }]
        model = GridSearchCV(pipeline, parameters, n_jobs=1, cv=CV, scoring="neg_log_loss", verbose=2)
    else:
        model = RandomForestClassifier(
            max_depth=model_args['max_depth'],
            max_features=model_args['max_features'],
            n_estimators=model_args['n_estimators'],
            n_jobs=-1,
            random_state=0,
            )
        # model = LogisticRegression(penalty=None)
    print("density model", model)
    return model

def get_outcome_model(
        is_binary: bool,
        max_feats: list[int],
        model_args=None,
        do_grid_search=False,
        n_jobs:int=-1,
        is_oracle: bool = False,
        gridsearch_polynom_lr: bool = False):
    if model_args is None:
        model_args = {'max_depth': DEFAULT_DEPTH, 'max_features': None, 'n_estimators': N_ESTIMATORS, 'bootstrap': True}
    if is_binary:
        if do_grid_search:
            pipeline = PipelineWeightedFit([
                ('clf', RandomForestClassifier())
            ])
            parameters = []
            if max_feats is None:
                parameters = [
                    {
                        'clf': (LogisticRegression(penalty='l1', solver='saga', max_iter=10000),),
                        'clf__C': [0.001,0.1,1,10,100,1000],
                    },
                    {
                        'clf': (LogisticRegression(penalty='l2', max_iter=10000),),
                        'clf__C': [1e-5,1e-04,0.001,0.1,1,10,100,1000],
                    }
                    ]
            else:
                parameters = [
                {
                    'clf': (RandomForestClassifier(
                        max_depth=model_args['max_depth'],
                        criterion="log_loss",
                        max_features=model_args['max_features'],
                        n_estimators=model_args['n_estimators'],
                        n_jobs=get_n_jobs(),
                        random_state=0,
                        bootstrap=model_args['bootstrap']),),
                    'clf__max_depth': [4,6,8],
                    'clf__max_features': max_feats
                },
                {
                    'clf': (GradientBoostingClassifier(
                        init=LogisticRegression(penalty=None, max_iter=2000),
                        max_depth=model_args['max_depth'],
                        n_estimators=model_args['n_estimators']),),
                    'clf__max_depth': [1],
                    'clf__n_estimators': [100,200,400,800]
                }
            ]
            if is_oracle:
                print("ORACLE")
                parameters += [
                    {
                        'clf': (LogisticRegression(penalty=None),)
                    }
                ]
            if gridsearch_polynom_lr:
                kernel_model = Pipeline([
                    ('poly', PolynomialFeatures(degree=2)),
                    ('linear', LogisticRegression(penalty="l2", solver='saga', max_iter=20000))])
                parameters += [{
                    'clf': [kernel_model],
                    'clf__linear__penalty': ['l1', 'l2'],
                    'clf__linear__C': [100,10,1,0.1,0.001]
                }]
            model = GridSearchCV(pipeline, parameters, scoring='neg_log_loss',
                    cv=CV, n_jobs=1, verbose=2)
        else:
            if is_oracle:
                model = LogisticRegression(penalty=None)
            else:
                model = RandomForestClassifier(
                    criterion="log_loss",
                    max_depth=model_args['max_depth'],
                    max_features=model_args['max_features'],
                    n_estimators=model_args['n_estimators'],
                    bootstrap=model_args['bootstrap'],
                    n_jobs=get_n_jobs(),
                    random_state=0
                )
    else:
        if do_grid_search:
            pipeline = PipelineWeightedFit([
                ('clf', RandomForestRegressor())
            ])
            kernel_model = Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('linear', Ridge(alpha=1, max_iter=20000))])
            
            parameters = [{
                'clf': (RandomForestRegressor(
                    max_depth=model_args['max_depth'],
                    max_features=model_args['max_features'],
                    n_estimators=model_args['n_estimators'],
                    n_jobs=get_n_jobs(),
                    random_state=0),),
                'clf__max_depth': [2,4,6,8],
                'clf__max_features': max_feats
            }]
            if len(max_feats) <= 2:
                parameters += [{
                    'clf': [kernel_model],
                    'clf__linear__alpha': [100,10,1,0.1,0.001,0.0001]
                }]
            model = GridSearchCV(
                pipeline,
                parameters,
                scoring='neg_mean_squared_error',
                cv=CV,
                n_jobs=1,
                verbose=2)
        else:
            model = RandomForestRegressor(max_depth=model_args['max_depth'],
                    max_features=model_args['max_features'],
                    n_estimators=model_args['n_estimators'], n_jobs=get_n_jobs(), random_state=0)
            # model = LinearRegression()
    print(f"outcome model (binary {is_binary})", model)
    return model
