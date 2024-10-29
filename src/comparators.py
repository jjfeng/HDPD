"""
Code for comparators described in Section 5.2 of the manuscript
"""
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split

from common import convert_prob_to_logit, compute_risk, min_max_scale
from decomp_explainer import ExplainerInference, InferenceResult
from data_loader import DataLoader
from data_generator import DataGenerator
from estimate_datashifter import EstimateShiftExplainerBaseVariables
from shapley import ShapleyInference

class ParametricChangeExplanation(ExplainerInference):
    def __init__(self, ml_mdl, source_data_loader: DataLoader, target_data_loader: DataLoader, source_generator: DataGenerator, target_generator: DataGenerator, loss_func):
        self.ml_mdl = ml_mdl
        self.source_data_loader = source_data_loader
        self.source_generator = source_generator
        self.sourceXW = self.source_data_loader._get_XW()
        self.sourceY = self.source_data_loader._get_Y()
        self.target_data_loader = target_data_loader
        self.target_generator = target_generator
        self.targetXW = self.target_data_loader._get_XW()
        self.targetY = self.target_data_loader._get_Y()
        self.loss_func = loss_func
        self.explanation = []

        self.num_p = source_data_loader.num_p
        self.w_indices = source_data_loader.w_indices
        self.w_mask = source_data_loader.w_mask
        self.num_w = self.w_indices.size
        print("NUM W", self.num_w)

    def do_decomposition(self):
        source_loss = self.loss_func(self.sourceXW, self.sourceY)
        source_dummy = np.zeros((self.sourceXW.shape[0], 1))
        target_loss = self.loss_func(self.targetXW, self.targetY)
        target_dummy = np.ones((self.targetXW.shape[0], 1))

        lr = LogisticRegression(penalty=None)
        lr_feats = np.concatenate([
            np.concatenate([source_dummy, self.sourceXW, source_dummy * self.sourceXW], axis=1),
            np.concatenate([target_dummy, self.targetXW, target_dummy * self.targetXW], axis=1),
        ])
        lr.fit(
            lr_feats,
            np.concatenate([self.sourceY, self.targetY])
        )
        print("LR CHANGE EXPLAIN", lr.coef_)
        logging.info("parametric mechanism LR explain %s", lr.coef_)
        self.explanation = lr.coef_.flatten()[-self.sourceXW.shape[1]:]

    def get_detailed_res(self) -> pd.DataFrame:
        df = []
        for i, coef in enumerate(self.explanation):
            df.append(
                {
                    "value": coef,
                    "se": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "ci_widths": np.nan,
                    "component": type(self).__name__,
                    "level": "detail",
                    "decomp": "Cond_Outcome",
                    "vars": "X%d" % (i+1),
                    "est": type(self).__name__,
                }
            )
        return pd.DataFrame(df)

    def summary(self) -> pd.DataFrame:
        return self.get_detailed_res()

class OaxacaBlinderExplanation(ParametricChangeExplanation):
    """
    Returns the Oaxaca Blinder decomposition which assumes a linear model 
    for log odds of expected loss. Difference in log odds of expected 0-1 loss 
    is decomposed into outcome and covariate shift components. Outcome shift is 
    attributed to each variable as difference in corresponding linear model 
    coefficient of the variable times the feature average in target. 
    Covariate shift is attributed as coefficient times the difference in 
    feature averages between target and source.
    """
    def _get_source_probability(self, X):
        return self.source_generator._get_prob(X).flatten()
        
    def _get_target_probability(self, X):
        return self.target_generator._get_prob(X).flatten()
    
    def do_decomposition(self):
        source_loss = self.loss_func(self.sourceXW, self.sourceY)
        source_dummy = np.zeros((self.sourceXW.shape[0], 1))
        if self.source_generator:
            source_prob = self._get_source_probability(self.sourceXW)
            source_risk = compute_risk(self.loss_func, self.sourceXW, source_prob)
        target_loss = self.loss_func(self.targetXW, self.targetY)
        target_dummy = np.ones((self.targetXW.shape[0], 1))
        if self.target_generator:
            target_prob = self._get_target_probability(self.targetXW)
            target_risk = compute_risk(self.loss_func, self.targetXW, target_prob)

        lpm_source = LinearRegression()
        lpm_source.fit(
            self.sourceXW,
            convert_prob_to_logit(source_risk) if self.source_generator else source_loss
        )
        lpm_target = LinearRegression()
        lpm_target.fit(
            self.targetXW,
            convert_prob_to_logit(target_risk) if self.target_generator else target_loss
        )
        
        feature_average_source = self.sourceXW.mean(axis=0)
        feature_average_target = self.targetXW.mean(axis=0)
        print("FEATURE AVERAGE source %s target %s", feature_average_source, feature_average_target)
        print("COEF LPM", lpm_source.coef_, lpm_target.coef_, lpm_target.coef_ - lpm_source.coef_)

        outcome_decomp = feature_average_target * (lpm_target.coef_ - lpm_source.coef_)
        cov_decomp =  (feature_average_target - feature_average_source) * lpm_source.coef_
        print("OUTCOME SHIFT EXPLAIN", outcome_decomp)
        print("COVARIATE SHIFT EXPLAIN", cov_decomp)
        logging.info("Oaxaca Blinder decomposition outcome %s", outcome_decomp)
        logging.info("Oaxaca Blinder decomposition covariate %s", cov_decomp)
        self.explanation_outcome_decomp = outcome_decomp.flatten()
        self.explanation_cov_decomp = cov_decomp.flatten()
        self.explanation = (self.explanation_outcome_decomp, self.explanation_cov_decomp)

    def get_detailed_res(self) -> pd.DataFrame:
        df = []
        for i, value in enumerate(self.explanation_outcome_decomp):
            df.append(
                {
                    "value": value,
                    "se": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "ci_widths": np.nan,
                    "component": type(self).__name__,
                    "level": "detail",
                    "decomp": "Cond_Outcome",
                    "vars": "X%d" % (i+1),
                    "est": type(self).__name__,
                }
            )
        for i, value in enumerate(self.explanation_cov_decomp):
            df.append(
                {
                    "value": value,
                    "se": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "ci_widths": np.nan,
                    "component": type(self).__name__,
                    "level": "detail",
                    "decomp": "Cond_Cov",
                    "vars": "X%d" % (i+1),
                    "est": type(self).__name__,
                }
            )
        return pd.DataFrame(df)

class ParametricAccExplanation(ParametricChangeExplanation):
    def do_decomposition(self):
        source_loss = self.loss_func(self.sourceXW, self.sourceY)
        source_dummy = np.zeros((self.sourceXW.shape[0], 1))
        target_loss = self.loss_func(self.targetXW, self.targetY)
        target_dummy = np.ones((self.targetXW.shape[0], 1))

        acc_lr = LogisticRegression(penalty=None)
        lr_feats = np.concatenate([
            np.concatenate([source_dummy, self.sourceXW, source_dummy * self.sourceXW], axis=1),
            np.concatenate([target_dummy, self.targetXW, target_dummy * self.targetXW], axis=1),
        ])
        acc_lr.fit(
            lr_feats,
            np.concatenate([source_loss, target_loss])
        )
        print("LR LOSS EXPLAIN", acc_lr.coef_)
        logging.info("parametric LR loss explain %s", acc_lr.coef_)
        self.explanation = acc_lr.coef_.flatten()[-self.sourceXW.shape[1]:]

class RandomForestExplanation(ParametricAccExplanation):
    def do_decomposition(self):
        rf_source = RandomForestClassifier()
        rf_source.fit(self.sourceXW, self.sourceY)
        rf_target = RandomForestClassifier()
        rf_target.fit(self.targetXW, self.targetY)
        
        source_vi = rf_source.feature_importances_[~self.w_mask]
        source_vi = source_vi/source_vi.sum()
        target_vi = rf_target.feature_importances_[~self.w_mask]
        target_vi = target_vi/target_vi.sum()

        self.explanation = np.abs(target_vi - source_vi)
        self.explanation /= self.explanation.sum()
        print("RF EXPLAIN", self.explanation)
        logging.info("RF explain %s", self.explanation)

class GBTAccExplanation(ParametricAccExplanation):
    def do_decomposition(self):
        target_loss = self.loss_func(self.targetXW, self.targetY)
        pred_logit_target = convert_prob_to_logit(self.ml_mdl.predict_proba(self.targetXW)[:,1:])

        acc_mdl = GridSearchCV(
            GradientBoostingClassifier(init=LogisticRegression(penalty="l1", solver="saga", max_iter=20000), n_estimators=50, max_depth=2),
            param_grid={
                'n_estimators': [50,100,200],
            },
            n_jobs=-1
        )
        feats = np.concatenate([pred_logit_target, self.targetXW], axis=1)
        acc_mdl.fit(feats, target_loss)
        self.explanation = acc_mdl.best_estimator_.feature_importances_[-self.num_p:]
        print("GBT EXPLAIN", self.explanation)
        logging.info("GBT explain %s", self.explanation)
        logging.info("GBT explain %s", acc_mdl.best_estimator_.feature_importances_)

class RandomForestAccExplanation(ParametricAccExplanation):
    def do_decomposition(self):
        source_loss = self.loss_func(self.sourceXW, self.sourceY)
        source_dummy = np.zeros((self.sourceXW.shape[0], 1))
        target_loss = self.loss_func(self.targetXW, self.targetY)
        target_dummy = np.ones((self.targetXW.shape[0], 1))
        # pred_logit_source = convert_prob_to_logit(self.ml_mdl.predict_proba(self.sourceXW)[:,1:])
        # pred_logit_target = convert_prob_to_logit(self.ml_mdl.predict_proba(self.targetXW)[:,1:])

        acc_mdl = RandomForestRegressor()
        feats = np.concatenate([
            np.concatenate([source_dummy, self.sourceXW], axis=1),
            np.concatenate([target_dummy, self.targetXW], axis=1)
        ], axis=0)
        acc_mdl.fit(
            feats,
            np.concatenate([source_loss, target_loss]))
        self.explanation = acc_mdl.feature_importances_[-self.num_p:]
        print("RF ACC EXPLAIN", self.explanation)
        logging.info("RF ACC explain %s", self.explanation)
        logging.info("RF ACC explain %s", acc_mdl.feature_importances_)

class WuShiftExplain(ParametricChangeExplanation, EstimateShiftExplainerBaseVariables):
    """Marginal shift explanations from the paper
    Wu, E., Wu, K., and Zou, J. Explaining medical AI
    performance disparities across sites with confounder
    shapley value analysis. http://arxiv.org/abs/2111.08168.
    """
    is_oracle = True
    gridsearch_polynom_lr = False #True
    def __init__(self, ml_mdl, source_data_loader: DataLoader, target_data_loader: DataLoader, source_generator: DataGenerator, target_generator: DataGenerator, loss_func, do_grid_search: bool=False, gamma: float = 20):
        self.ml_mdl = ml_mdl
        self.source_loader = source_data_loader
        self.num_p = self.source_loader.num_p
        self.num_obs = self.source_loader.num_n
        self.source_generator = source_generator
        self.target_loader = target_data_loader
        self.target_generator = target_generator
        self.gamma = gamma
        
        if source_generator is not None:
            self.sourceXW_test = self.source_loader._get_XW()
            self.sourceW_test = self.source_loader._get_W()
            self.sourceY_test = self.source_loader._get_Y()
            self.targetXW_test = self.target_loader._get_XW()
            self.targetW_test = self.target_loader._get_W()
            self.targetY_test = self.target_loader._get_Y()
            self.sourceXW_train, self.sourceY_train = self.source_generator.generate(self.num_obs * 2)
            self.sourceW_train = self.source_generator._get_W(self.sourceXW_train)
            self.targetXW_train, self.targetY_train = self.target_generator.generate(self.num_obs * 2)
            self.targetW_train = self.target_generator._get_W(self.targetXW_train)
        else:
            sourceXW = self.source_loader._get_XW()
            sourceW = self.source_loader._get_W()
            sourceY = self.source_loader._get_Y()
            targetXW = self.target_loader._get_XW()
            targetW = self.target_loader._get_W()
            targetY = self.target_loader._get_Y()

            # Split into train and test
            self.targetXW_train, self.targetXW_test, self.targetW_train, self.targetW_test,\
            self.targetY_train, self.targetY_test = train_test_split(
                targetXW, targetW, targetY, test_size=self.split_ratio
            )
            self.sourceXW_train, self.sourceXW_test, self.sourceW_train, self.sourceW_test,\
            self.sourceY_train, self.sourceY_test = train_test_split(
                sourceXW, sourceW, sourceY, test_size=self.split_ratio
            )

        self.do_grid_search = do_grid_search
        self.loss_func = loss_func
        self.w_mask = source_data_loader.w_mask
        self.num_w = self.w_mask.sum()
    
        self.source_outcome_model, self.target_outcome_model = self._estimate_outcome_models()
        self.mu_yA0_xA0_fn_w = self._estimate_mu_yA0_xA0_fn_w()
        self.mu_yA0_xA1_fn_w = self._estimate_mu_yA0_xA1_fn_w()

    def explain_marginal_shift_detailed(self, s):
        """
        Create shifted covariate distribution by replacing conditional covariate distribution at target in place of the non-shifted covariate subset in the graph, 
        value is determined by loss difference between shifted and source covariate distribution
        """
        exp_loss_yA0_xA0_target = self.mu_yA0_xA0_fn_w.predict(self.targetW_test)
        exp_loss_yA0_xA1_target = self.mu_yA0_xA1_fn_w.predict(self.targetW_test)
        diffs = exp_loss_yA0_xA1_target - exp_loss_yA0_xA0_target

        if s.sum() == 0:
            diffs = np.zeros(self.targetW_test.shape[0])
        elif np.all(s):
            exp_loss_yA0_xA0_target = self.mu_yA0_xA0_fn_w.predict(self.targetW_test)
            exp_loss_yA0_xA1_target = self.mu_yA0_xA1_fn_w.predict(self.targetW_test)
            diffs = exp_loss_yA0_xA1_target - exp_loss_yA0_xA0_target
        else:
            subgroup_mask = np.ones(self.num_p, dtype=bool)
            subgroup_mask[~self.w_mask] = s
            
            ## FIT models
            mu_yA0_xmsA0_model = self._estimate_mu_yA0_xmsA0_fn_xsw(subgroup_mask)
            mu_yA0_xmsA0_xsA1_model = self._estimate_mu_yA0_xmsA0_xsA1_fn_w(mu_yA0_xmsA0_model, subgroup_mask)
            
            # Estimate loss change
            exp_loss_yA0_xA0_target = self.mu_yA0_xA0_fn_w.predict(self.targetW_test)
            exp_loss_yA0_xmsA0_xsA1_target = mu_yA0_xmsA0_xsA1_model.predict(self.targetW_test)
            diffs = exp_loss_yA0_xmsA0_xsA1_target - exp_loss_yA0_xA0_target
        logging.info("Wu DECOMP %s %f", s, diffs.mean())
        return {
            "wu": InferenceResult(
                estim=diffs.mean(),
                ic=diffs - diffs.mean())
        }

    def do_decomposition(self):        
        self.detailed_cond_cov_shapler = ShapleyInference(
            self.num_obs, self.num_p - self.num_w, self.explain_marginal_shift_detailed, self.gamma
        )
        res = self.detailed_cond_cov_shapler.get_point_est()['wu']
        self.explanation = res.flatten()[1:]
        logging.info("orig shapleys %s", self.explanation)
        self.explanation = np.abs(self.explanation)/np.sum(np.abs(self.explanation))
        logging.info("abs shapleys %s", self.explanation)

    def get_detailed_res(self) -> pd.DataFrame:
        df = []
        for i, value in enumerate(self.explanation):
            df.append(
                {
                    "value": value,
                    "se": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "ci_widths": np.nan,
                    "component": type(self).__name__,
                    "level": "detail",
                    "decomp": "Cond_Cov",
                    "vars": f"X{i+self.num_w+1}",
                    "est": type(self).__name__,
                }
            )
        return pd.DataFrame(df)

class MeanChange(WuShiftExplain):
    def do_decomposition(self):
        ttest_results = np.zeros(self.sourceXW_train.shape[1])
        for i in range(self.sourceXW_train.shape[1]):
            print("MEAN CHANGE", i, self.sourceXW_train[:,i].mean(), self.targetXW_train[:,i].mean())
            ttest_results[i] = 1 - ttest_ind(self.sourceXW_train[:,i], self.targetXW_train[:,i]).pvalue

        self.explanation = ttest_results[~self.w_mask]
        print("MEAN CHANGE", self.explanation)
