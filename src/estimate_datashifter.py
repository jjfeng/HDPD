"""Computes shift explanations by models estimated from data.
Does not use data generators.
"""

import time
import logging
from typing import Tuple, Dict

from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from decomp_explainer import BaseShiftExplainer, InferenceResult, to_str_inf_res
from decomp_explainer import ConstantModel
from data_loader import DataLoader
from common import *


class EstimateShiftExplainerBaseVariables(BaseShiftExplainer):
    split_ratio = 0.2
    num_modeling_attempts = 10
    def __init__(self, source_loader: DataLoader, target_loader: DataLoader, loss_func, ml_mdl, num_bins, do_grid_search, do_clipping, gridsearch_polynom_lr: bool=False, is_oracle: bool=False, reps_ustatistics: int = 2000):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.loss_func = loss_func
        self.ml_mdl = ml_mdl
        self.num_bins = num_bins
        self.do_grid_search = do_grid_search
        self.min_ustats_dens_ratio_cutoff = 1e-6 if do_clipping else 1e-10
        self.gridsearch_polynom_lr = gridsearch_polynom_lr
        self.w_mask = self.target_loader.w_mask
        self.is_oracle = is_oracle
        self.reps_ustatistics = reps_ustatistics

        targetXW = self.target_loader._get_XW()
        sourceXW = self.source_loader._get_XW()
        targetX = self.target_loader._get_X()
        sourceX = self.source_loader._get_X()
        targetW = self.target_loader._get_W()
        targetY = self.target_loader._get_Y()
        sourceW = self.source_loader._get_W()
        sourceY = self.source_loader._get_Y()

        # Split into train and test
        self.targetXW_train, self.targetXW_test, self.targetX_train, self.targetX_test, self.targetW_train, self.targetW_test,\
        self.targetY_train, self.targetY_test = train_test_split(
            targetXW, targetX, targetW, targetY, test_size=self.split_ratio
        )
        self.sourceXW_train, self.sourceXW_test, self.sourceX_train, self.sourceX_test, self.sourceW_train, self.sourceW_test,\
        self.sourceY_train, self.sourceY_test = train_test_split(
            sourceXW, sourceX, sourceW, sourceY, test_size=self.split_ratio
        )
        self.test_source_n = self.sourceXW_test.shape[0]
        self.test_target_n = self.targetXW_test.shape[0]
        self.source_prevalence = self.test_source_n/(self.test_source_n + self.test_target_n)
        self.target_prevalence = 1 - self.source_prevalence
        logging.info("prevalence source %f, target %f", self.source_prevalence, self.target_prevalence)

        self.cond_outcome_denom_res = None
        self.cond_cov_denom_res = None

        # Fit outcome models for Y|X,W,A=0 and Y|X,W,A=1
        self.source_outcome_model, self.target_outcome_model = self._estimate_outcome_models()
        
        # binned_preds = self._bin_probabilities(self.source_outcome_model.predict_proba(self.sourceXW_test)[:,1])
        # beta = np.array([0.3,1,0.5,1]).reshape((-1,1))
        # true_prob = 1/(1 + np.exp(-np.matmul(self.sourceXW_test, beta))).flatten()
        # binned_true = self._bin_probabilities(true_prob)
        # logging.info("BIN OVERLAP %f", np.mean(binned_preds == binned_true))

        # Fit density models for X,W and W
        self.density_model_xw, self.density_model_w = self._estimate_density_models()

        # Fit model for expected loss in source for each W in source
        self.mu_yA0_xA0_fn_w = self._estimate_mu_yA0_xA0_fn_w()

        # Fit model for expected loss in source for each W in target
        self.mu_yA0_xA1_fn_w = self._estimate_mu_yA0_xA1_fn_w()

        self.agg_res_w = None
        self.agg_res_x = None
        self.agg_res_y = None
        self.detail_cond_cov_res_dict = {}
        self.detail_cond_outcome_res_dict = {}
    
    def _estimate_density_models(self):
        """Fit density models for X,W and for W, defined in Section 4.1
        """
        # Fit density models for X,W
        density_model_xw = get_density_model(
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(self.targetXW_train.shape[1]),
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
            )
        density_model_xw.fit(
            np.concatenate([self.sourceXW_train, self.targetXW_train]),
            np.concatenate([np.zeros(self.sourceXW_train.shape[0]), np.ones(self.targetXW_train.shape[0])])
        )
        if self.do_grid_search:
            logging.info("selected model densityXW %s %s", density_model_xw.best_estimator_, density_model_xw.cv_results_)

        # Fit density models for W
        density_model_w = get_density_model(
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(self.targetW_train.shape[1]),
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
        )
        if self.sourceW_train.shape[1]!=0:
            density_model_w.fit(
                np.concatenate([self.sourceW_train, self.targetW_train]),
                np.concatenate([np.zeros(self.sourceW_train.shape[0]), np.ones(self.targetW_train.shape[0])]),
            )
            if self.do_grid_search:
                logging.info("selected model densityW %s %s", density_model_w.best_estimator_, density_model_w.cv_results_)
        return density_model_xw, density_model_w

    def _estimate_outcome_models(self):
        """Fit target and source outcome models, defined in Section 4.1
        """
        # Fit target outcome model
        target_outcome_model = get_outcome_model(
            is_binary=True,
            is_oracle=self.is_oracle,
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(self.targetXW_train.shape[1]),
            model_args={'n_estimators': 800, 'max_depth': 4, 'max_features': 11, 'bootstrap': True},
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
        )
        target_outcome_model.fit(
            self.targetXW_train,
            self.targetY_train
        )

        # Fit source outcome model
        source_outcome_model = get_outcome_model(
            is_binary=True,
            is_oracle=self.is_oracle,
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(self.targetXW_train.shape[1]),
            model_args={'n_estimators': 800, 'max_depth': 6, 'max_features': 9, 'bootstrap': True},
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
        )
        source_outcome_model.fit(
            self.sourceXW_train, self.sourceY_train
        )
        if self.do_grid_search and not self.is_oracle:
            logging.info("CV RESULTS %s", source_outcome_model.cv_results_)
            print("selected model sourceY", source_outcome_model.best_estimator_, source_outcome_model.best_score_)
            logging.info("selected model sourceY %s %f", source_outcome_model.best_estimator_, source_outcome_model.best_score_)
            logging.info("CV RESULTS %s", target_outcome_model.cv_results_)
            print("selected model targetY", target_outcome_model.best_estimator_, target_outcome_model.best_score_)
            logging.info("selected model targetY %s %f", target_outcome_model.best_estimator_, target_outcome_model.best_score_)

        return source_outcome_model, target_outcome_model

    def _estimate_mu_yA0_xA1_fn_w(self):
        """Regress exp loss from fitted outcome model on w on target data,
        defined in eq. (8)
        """
        # Expected loss on source
        # TODO: is ithis better than a weighted regression on the loss itself?
        source_prob = self._get_source_probability(self.targetXW_train)
        exp_loss_source_on_targetXW = compute_risk(self.loss_func, self.targetXW_train, source_prob)
        print("MEAN LOSS ON TARGET", exp_loss_source_on_targetXW.mean())

        if self.targetW_train.shape[1]!=0:
            exp_outcome_model = self._project_model(
                trainX=self.targetW_train,
                trainY=exp_loss_source_on_targetXW,
            )
            if self.do_grid_search:
                logging.info("_estimate_mu_yA0_xA1_fn_w %s", exp_outcome_model.best_estimator_)
        else:
            exp_outcome_model = ConstantModel(np.mean(exp_loss_source_on_targetXW))
        return exp_outcome_model
    
    def _estimate_mu_yA0_xmsA0_fn_xsw(self, subgroup_mask: np.ndarray):
        """
        Fit a regression model for mu_0(x,w) on xs, w, defined in eq. (7)
        Assumes subgroup_mask contains w_indices
        """
        assert np.all(subgroup_mask[self.target_loader.w_indices]), "w_indices should be set to true for conditional covariate"
        # Regress sample loss on xs, w from source data
        loss_on_source = self.loss_func(self.sourceXW_train, self.sourceY_train)

        exp_outcome_model = self._project_model(
            trainX=self.sourceXW_train[:, subgroup_mask], trainY=loss_on_source
        )
        return exp_outcome_model
    
    def _estimate_mu_yA0_xmsA0_xsA1_fn_w(self, mu_yA0_xmsA0_model, subgroup_mask: np.ndarray):
        """
        Fit a regression model for mu_0(x,w) on xs, w in target data,
        defined in eq. (9)
        Assumes subgroup_mask contains w_indices
        """
        assert np.all(subgroup_mask[self.target_loader.w_indices]), "w_indices should be set to true for conditional covariate"
        # Option 1: Expected loss on source
        exp_loss_source_on_targetXW = mu_yA0_xmsA0_model.predict(
            self.targetXW_train[:, subgroup_mask]
        )

        exp_outcome_model = self._project_model(
            self.targetW_train, exp_loss_source_on_targetXW
        )
        # Option 2: Regress sample loss on w weighted by density ratio of xs, w in source data
        # loss_on_source = self.loss_func(self.sourceXW_train, self.sourceY_train)
        # if subgroup_mask.sum():
        #     prob_target_xsw = density_model_xsw.predict_proba(self.sourceXW_train[:, subgroup_mask])[:,1]
        #     print("ODDS Xs", prob_target_xsw.min(), prob_target_xsw.max())
        #     odds_target_xsw = prob_target_xsw/(1 - prob_target_xsw) * self.sourceXW_train.shape[0]/self.targetXW_train.shape[0]
        # else:
        #     odds_target_xsw = 1
        # exp_outcome_model = get_outcome_model(is_binary=False, do_grid_search=self.do_grid_search)  # TODO: grid search in Pipeline
        # exp_outcome_model.fit(
        #     self.sourceW_train, loss_on_source, sample_weight=odds_target_xsw
        # )
        return exp_outcome_model
    
    def _estimate_density_model_xsw(self, subgroup_mask_plus_w: np.ndarray):
        """Fits density ratio model for X_s, W, defined in Section B.1
        """
        density_model_xsw = get_density_model(
            do_grid_search=self.do_grid_search,
            max_feats=self._make_max_feats_list(subgroup_mask_plus_w.sum()),
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
            )
        density_model_xsw.fit(
            np.concatenate([
                self.sourceXW_train[:, subgroup_mask_plus_w], self.targetXW_train[:, subgroup_mask_plus_w]
            ], axis=0),
            np.concatenate([
                np.zeros(self.sourceXW_train.shape[0]), np.ones(self.targetXW_train.shape[0])
            ], axis=0)
        )
        if self.do_grid_search:
            logging.info("selected model density_model_xsw %s %s", density_model_xsw.best_estimator_, density_model_xsw.cv_results_)
        return density_model_xsw

    ## Marginal shift in baseline variables
    def _baseline_shift_loss_aggregate(self, sourceXW: np.ndarray, sourceW: np.ndarray, sourceY: np.ndarray, targetW: np.ndarray):
        """Loss difference when only marginal distribution of baseline variables shifts,
        Estimator is defined in Section 4.1, influence function in eq. (12)
        """
        loss_source = self.loss_func(sourceXW, sourceY)
        source_exp_outcome_w = self.mu_yA0_xA0_fn_w.predict(sourceW)
        target_exp_outcome_w = self.mu_yA0_xA0_fn_w.predict(targetW)

        if sourceW.shape[1]!=0:
            odds_target_w = self._get_density_ratio_w(sourceW)
        else:
            odds_target_w = 1

        loss_diff_plugin_ics = np.concatenate([
            -loss_source/self.source_prevalence,
            target_exp_outcome_w/self.target_prevalence])
        print("loss diff", loss_diff_plugin_ics.mean())

        loss_diff_onestep_ics = np.concatenate([
            ((loss_source - source_exp_outcome_w) * odds_target_w - loss_source)/self.source_prevalence,
            target_exp_outcome_w/self.target_prevalence])

        logging.info(
            "BASELINE AGGREGATE contrib term1 %f term2 %f", 
            ((loss_source - source_exp_outcome_w) * odds_target_w - loss_source).mean(), 
            target_exp_outcome_w.mean()
        )
        return {
            'plugin': InferenceResult(
                loss_diff_plugin_ics.mean(),
                loss_diff_plugin_ics - loss_diff_plugin_ics.mean()
            ),
            'onestep': InferenceResult(
                loss_diff_onestep_ics.mean(),
                loss_diff_onestep_ics - loss_diff_onestep_ics.mean()
            )
        }
    
    def _marginal_shift_loss_aggregate(self, sourceXW: np.ndarray, sourceW: np.ndarray, sourceY: np.ndarray, targetXW: np.ndarray, targetW: np.ndarray) -> Dict[str, InferenceResult]:
        """Loss difference when only conditional covariate distribution shifts
        Estimator is defined in Section 4.1, influence function in eq. (13)
        """
        loss_source = self.loss_func(sourceXW, sourceY)
        odds_target_xw = self._get_density_ratio_xw(sourceXW)
        if sourceW.shape[1]!=0:
            odds_target_w = self._get_density_ratio_w(sourceW)
        else:
            odds_target_w = 1
        # Source data
        exp_loss_source_on_sourceXW = compute_risk(
            self.loss_func,
            sourceXW,
            self.source_outcome_model.predict_proba(sourceXW)[:,1])
        exp_loss_w_on_sourceW = self.mu_yA0_xA0_fn_w.predict(sourceW)
        # Target data
        exp_loss_source_on_targetXW = compute_risk(
            self.loss_func,
            targetXW,
            self.source_outcome_model.predict_proba(targetXW)[:,1])
        exp_loss_w_on_targetW = self.mu_yA0_xA0_fn_w.predict(targetW)

        # Plugin
        plugin_ics = np.concatenate([
            np.zeros(self.test_source_n),
            (exp_loss_source_on_targetXW - exp_loss_w_on_targetW)/self.target_prevalence
        ])
        print("loss_diff", plugin_ics.mean())

        # One step
        onestep_ics = np.concatenate([
            # Using source obs
            ((loss_source - exp_loss_source_on_sourceXW) * odds_target_xw - (loss_source - exp_loss_w_on_sourceW) * odds_target_w)/self.source_prevalence,
            # Using target obs
            (exp_loss_source_on_targetXW - exp_loss_w_on_targetW)/self.target_prevalence
        ])

        logging.info(
            "COND COV AGGREGATE contrib term1 %f term2 %f", 
            ((loss_source - exp_loss_source_on_sourceXW) * odds_target_xw - (loss_source - exp_loss_w_on_sourceW) * odds_target_w).mean(), 
            (exp_loss_source_on_targetXW - exp_loss_w_on_targetW).mean()
        )
        return {
            'plugin': InferenceResult(
                plugin_ics.mean(),
                plugin_ics - plugin_ics.mean()
            ),
            'onestep': InferenceResult(
                onestep_ics.mean(),
                onestep_ics - onestep_ics.mean()
            )
        }

    def _conditional_shift_loss_aggregate(self, targetXW: np.ndarray, targetY: np.ndarray, sourceXW: np.ndarray, sourceY: np.ndarray):
        """Loss difference when only conditional outcome distribution shifts
        Estimator is defined in Section 4.1, influence function in eq. (14)
        """
        loss_target = self.loss_func(targetXW, targetY)
        loss_source = self.loss_func(sourceXW, sourceY)
        prob_target_xw = self.density_model_xw.predict_proba(sourceXW)[:,1]
        odds_target_xw = prob_target_xw/(1 - prob_target_xw) * self.sourceX_train.shape[0]/self.targetX_train.shape[0]
        print("ODDS XW conditional", prob_target_xw.min(), prob_target_xw.max())

        # Evaluate loss on test data from target
        exp_loss_source_on_targetXW = compute_risk(self.loss_func, targetXW, self.source_outcome_model.predict_proba(targetXW)[:,1])

        # Evaluate loss on test data from source
        exp_loss_source_on_sourceXW = compute_risk(self.loss_func, sourceXW, self.source_outcome_model.predict_proba(sourceXW)[:,1])

        # plugin
        plugin_ics = np.concatenate([
            np.zeros(self.test_source_n),
            (loss_target - exp_loss_source_on_targetXW)/self.target_prevalence
        ])
        print("loss_diff", plugin_ics.mean())

        # onestep
        onestep_ics = np.concatenate([
            - (loss_source - exp_loss_source_on_sourceXW) * odds_target_xw/self.source_prevalence,
            (loss_target - exp_loss_source_on_targetXW)/self.target_prevalence,
        ])
        logging.info(
            "aggY %f %f",
            (loss_target - exp_loss_source_on_targetXW).mean(),
            -((loss_source - exp_loss_source_on_sourceXW) * odds_target_xw).mean())
        return {
            'plugin': InferenceResult(
                plugin_ics.mean(),
                plugin_ics - plugin_ics.mean()
            ),
            'onestep': InferenceResult(
                onestep_ics.mean(),
                onestep_ics - onestep_ics.mean()
            )
        }
    
    def _estimate_shifted_outcome_model(self, source_prob_train: np.ndarray, subgroup_mask: np.ndarray):
        """Fit manipulated outcome model p(y|xs,w,q), defined in eq. (3)
        """
        shifted_outcome_model = get_outcome_model(
            is_binary=True,
            is_oracle=self.is_oracle,
            do_grid_search=self.do_grid_search,
            model_args={'n_estimators': 800, 'max_features': 2, 'max_depth': 4, 'bootstrap': True},
            max_feats=self._make_max_feats_list(subgroup_mask.sum() + 1),
            gridsearch_polynom_lr=self.gridsearch_polynom_lr,
        )
        shifted_outcome_model.fit(
            np.concatenate([
                self.targetXW_train[:, subgroup_mask], self._bin_probabilities(source_prob_train)[:, np.newaxis]
            ], axis=1),
            self.targetY_train,
        )
        if self.do_grid_search:
            print("selected shifted outcome model", shifted_outcome_model.best_estimator_, shifted_outcome_model.best_score_)
            logging.info("shifted outcome model CV %s", shifted_outcome_model.cv_results_)
            logging.info("selected shifted outcome model %s %f", shifted_outcome_model.best_estimator_, shifted_outcome_model.best_score_)
        return shifted_outcome_model
    
    def _estimate_density_model_ustatistic(self, source_prob_train: np.ndarray, subgroup_mask: np.ndarray):
        """Fits density model for detailed decomposition of conditional outcome shift,
        defined in Section 4.2.1 before eq. (4)
        """
        REPS = 1
        
        # TODO: return 1 when subgroup_mask is all 1s
        binned_source_prob_train = self._bin_probabilities(source_prob_train)[:, np.newaxis]
        binned_source_prob_train_exp = np.repeat(binned_source_prob_train, repeats=REPS, axis=0)
        logging.info("ORIG BINNED SOURCE %s", np.unique(binned_source_prob_train, return_counts=True))
        
        # Shuffle XW_{-s} to get tildeXW_{-s}, rest of features are same as XW
        print(self.targetXW_train.shape)
        tilde_targetXW_rep = np.repeat(self.targetXW_train, repeats=REPS, axis=0)
        print(tilde_targetXW_rep.shape)
        perm = np.random.choice(tilde_targetXW_rep.shape[0], size=tilde_targetXW_rep.shape[0], replace=True)
        tilde_targetXW_train = tilde_targetXW_rep[perm, :]
        tilde_targetXW_train[:, subgroup_mask] = tilde_targetXW_rep[:, subgroup_mask]
        
        # Engineer a feature that checks if the binned tilde target XW matches the original binned source prob?
        binned_tilde_source_prob_train = self._bin_probabilities(self._get_source_probability(tilde_targetXW_train))[:, np.newaxis]
        logging.info("TILDE BINNED SOURCE %s", np.unique(binned_tilde_source_prob_train, return_counts=True))
        is_same_bin = (binned_tilde_source_prob_train == binned_source_prob_train_exp).astype(int)
        print("is_same_bin", is_same_bin.mean())

        print("BIN UNIQ ROW", np.unique(binned_source_prob_train, return_counts=True))
        # p(A=0|X-s, Xs, Q)/p(A=1|X-s, Xs, Q) = p(X-s, Xs, Q, A=0)/p(X-s, Xs, Q, A=1) = p(X-s, Xs, Q | A=0)/p(X-s, Xs, Q | A=1)
        if subgroup_mask.sum() == 0:
            density_feats = np.concatenate([
                np.concatenate([binned_source_prob_train_exp, is_same_bin], axis=1),
                np.concatenate([binned_source_prob_train, np.ones(binned_source_prob_train.shape)], axis=1),
            ], axis=0)
            density_model = get_density_model(
                do_grid_search=self.do_grid_search,
                max_feats=[1,2],
                model_args = {'max_depth': 4, 'max_features': 2, 'n_estimators': 800},
                gridsearch_polynom_lr=self.gridsearch_polynom_lr,
            )
        else:
            density_feats = np.concatenate([
                np.concatenate([tilde_targetXW_train, binned_source_prob_train_exp, is_same_bin], axis=1),
                np.concatenate([self.targetXW_train, binned_source_prob_train, np.ones(binned_source_prob_train.shape)], axis=1),
            ], axis=0)
            density_model = get_density_model(
                do_grid_search=self.do_grid_search,
                max_feats=self._make_max_feats_list(self.targetXW_train.shape[1] + 2, min_num_feats=2),
                model_args = {'max_depth': 6, 'max_features': 10, 'n_estimators': 800},
                gridsearch_polynom_lr=self.gridsearch_polynom_lr,
            )
        
        density_model.fit(
            density_feats,
            np.concatenate([np.zeros(binned_source_prob_train_exp.shape[0]), np.ones(binned_source_prob_train.shape[0])], axis=0),
        )
        print("density model fitted")
        if self.do_grid_search:
            logging.info("density ustatistic CV RESULTS %s", density_model.cv_results_)
            print('selected density ustatistic', density_model.best_estimator_)
            logging.info('selected density ustatistic %s', density_model.best_estimator_)
            # logging.info('VI density ustat %s', density_model.best_estimator_.steps[0][1].feature_importances_)

        # Check predictions from density model
        trained_preds = density_model.predict_proba(density_feats)[:,1]
        logging.info("DENSITY U-stat TRAINED PRED max %f min %f", trained_preds.max(), trained_preds.min())
        return density_model
    
    def _get_source_probability(self, X):
        return self.source_outcome_model.predict_proba(X)[:,1]
        
    def _get_target_probability(self, X):
        return self.target_outcome_model.predict_proba(X)[:,1]
    
    def _get_density_ratio_w(self, W):
        prob_a_given_w = self.density_model_w.predict_proba(W)[:,1]
        odds_w = prob_a_given_w/(1 - prob_a_given_w) * self.sourceW_train.shape[0]/self.targetW_train.shape[0]  # multiply odds by p(a=0)/p(a=1) to get density ratio
        logging.info("ODDS W %f %f", prob_a_given_w.min(), prob_a_given_w.max())
        return odds_w
    
    def _get_density_ratio_xw(self, XW):
        prob_a_given_xw = self.density_model_xw.predict_proba(XW)[:,1]
        odds_xw = prob_a_given_xw/(1 - prob_a_given_xw) * self.sourceX_train.shape[0]/self.targetX_train.shape[0]  # multiply odds by p(a=0)/p(a=1) to get density ratio
        logging.info("ODDS XW %f %f", prob_a_given_xw.min(), prob_a_given_xw.max())
        return odds_xw
        
    def _marginal_shift_loss_detailed_denom(self, sourceW: np.ndarray, sourceXW: np.ndarray, sourceY: np.ndarray, targetW: np.ndarray, targetXW: np.ndarray):
        """
        Estimate and get EIFs for conditional covariate detailed denominator
        Estimator defined in eq. (10) and EIF in eq. (17) when evalauted for an empty subset
        Shared across multiple subgroup_masks
        Returns:
            _type_: _description_
        """
        logging.info("conditional covariate detailed denom")
        ## EVALUATE plugin
        exp_loss_yA0_xA0_target = self.mu_yA0_xA0_fn_w.predict(targetW)
        exp_loss_yA0_xA1_target = self.mu_yA0_xA1_fn_w.predict(targetW)
        # tot_var_plugin_estim = np.power(np.std(exp_loss_yA0_xA1_target - exp_loss_yA0_xA0_target), 2)
        denom_plugin_ics = np.power(exp_loss_yA0_xA1_target - exp_loss_yA0_xA0_target, 2)

        ## EVALUATE onestep
        # On source
        source_loss = self.loss_func(sourceXW, sourceY)
        source_prob_on_sourceXW = self._get_source_probability(sourceXW)
        exp_loss_yA0_source = compute_risk(self.loss_func, sourceXW, source_prob_on_sourceXW)
        exp_loss_yA0_xA0_source = self.mu_yA0_xA0_fn_w.predict(sourceW)
        exp_loss_yA0_xA1_source = self.mu_yA0_xA1_fn_w.predict(sourceW)
        exp_loss_diff_source = exp_loss_yA0_xA1_source - exp_loss_yA0_xA0_source

        # On target
        source_prob_on_targetXW = self._get_source_probability(targetXW)
        exp_loss_yA0_target = compute_risk(self.loss_func, targetXW, source_prob_on_targetXW)
        exp_loss_diff_target = exp_loss_yA0_xA1_target - exp_loss_yA0_xA0_target

        # Get density ratios
        odds_target_w = self._get_density_ratio_w(sourceW)
        odds_target_xw = self._get_density_ratio_xw(sourceXW)

        denom_onestep_term2_source = 2 * exp_loss_diff_source * (source_loss - exp_loss_yA0_source) * odds_target_xw
        denom_onestep_term3_source = -2 * exp_loss_diff_source * (source_loss - exp_loss_yA0_xA0_source) * odds_target_w
        denom_onestep_term4_target = 2 * exp_loss_diff_target * (exp_loss_yA0_target - exp_loss_yA0_xA1_target)

        denom_onestep_ics = np.concatenate([
            (denom_onestep_term2_source + denom_onestep_term3_source)/self.source_prevalence,
            (denom_plugin_ics + denom_onestep_term4_target)/self.target_prevalence
        ])
        logging.info("conditional covariate detailed denom %f %f %f %f", denom_plugin_ics.mean(), denom_onestep_term2_source.mean(), denom_onestep_term3_source.mean(), denom_onestep_term4_target.mean())

        return {
            'plugin': InferenceResult(
                denom_plugin_ics.mean(),
                np.concatenate([np.zeros(self.test_source_n), denom_plugin_ics/self.target_prevalence])
            ),
            'onestep': InferenceResult(
                denom_onestep_ics.mean(),
                denom_onestep_ics
            )}
    
    def _marginal_shift_loss_detailed_numerator(self, sourceW: np.ndarray, sourceXW: np.ndarray, sourceY: np.ndarray, targetW: np.ndarray, targetXW: np.ndarray, subgroup_mask: np.ndarray) -> Dict[str, InferenceResult]:
        """
        Create shifted covariate distribution by replacing conditional covariate distribution at target in place of the non-shifted covariate subset in the graph, 
        value is determined by loss difference between shifted and source covariate distribution
        Estimator defined in eq. (10) and EIF in eq. (17)
        """
        logging.info("conditional covariate detailed numerator %s", subgroup_mask)
        # check baseline variables all included
        assert np.all(subgroup_mask[self.w_mask])
        
        ## FIT models
        density_model_xsw = self._estimate_density_model_xsw(subgroup_mask)
        mu_yA0_xmsA0_model = self._estimate_mu_yA0_xmsA0_fn_xsw(subgroup_mask)
        mu_yA0_xmsA0_xsA1_model = self._estimate_mu_yA0_xmsA0_xsA1_fn_w(mu_yA0_xmsA0_model, subgroup_mask)
        
        ## EVALUATE plugin numerator
        # On target
        exp_loss_yA0_xA1_target = self.mu_yA0_xA1_fn_w.predict(targetW)
        exp_loss_yA0_xmsA0_xsA1_target = mu_yA0_xmsA0_xsA1_model.predict(targetW)
        exp_loss_diff_xms_target = exp_loss_yA0_xmsA0_xsA1_target - exp_loss_yA0_xA1_target
        mse_plugin_ics = np.power(exp_loss_diff_xms_target, 2)

        ## EVALUATE one-step numerator
        # On source
        source_loss = self.loss_func(sourceXW, sourceY)
        source_prob_on_sourceXW = self._get_source_probability(sourceXW)
        exp_loss_yA0_source = compute_risk(self.loss_func, sourceXW, source_prob_on_sourceXW)
        exp_loss_yA0_xA1_source = self.mu_yA0_xA1_fn_w.predict(sourceW)
        exp_loss_yA0_xmsA0_xsA1_source = mu_yA0_xmsA0_xsA1_model.predict(sourceW)
        exp_loss_diff_xms_source = exp_loss_yA0_xmsA0_xsA1_source - exp_loss_yA0_xA1_source
        exp_loss_yA0_xmsA0_source = mu_yA0_xmsA0_model.predict(sourceXW[:, subgroup_mask])
        # On target
        source_prob_on_targetXW = self._get_source_probability(targetXW)
        exp_loss_yA0_target = compute_risk(self.loss_func, targetXW, source_prob_on_targetXW)
        exp_loss_yA0_xmsA0_target = mu_yA0_xmsA0_model.predict(targetXW[:, subgroup_mask])
        # Get density ratios
        odds_target_xw = self._get_density_ratio_xw(sourceXW)
        prob_target_xsw = density_model_xsw.predict_proba(sourceXW[:, subgroup_mask])[:,1]
        odds_target_xsw = prob_target_xsw/(1 - prob_target_xsw) * self.sourceXW_train.shape[0]/self.targetXW_train.shape[0]
        logging.info("COND COV ODDs W min %f max %f", odds_target_xw.min(), odds_target_xw.max())
        logging.info("COND COV ODDs XsW min %f max %f", odds_target_xsw.min(), odds_target_xsw.max())
        # Get correction term
        mse_one_step_term1_source = 2 * exp_loss_diff_xms_source\
                                        * (source_loss - exp_loss_yA0_xmsA0_source) * odds_target_xsw
        mse_one_step_term2_source = -2 * exp_loss_diff_xms_source\
                                        * (source_loss - exp_loss_yA0_source) * odds_target_xw
        mse_one_step_term3_target = 2 * exp_loss_diff_xms_target\
                                        * (exp_loss_yA0_xmsA0_target - exp_loss_yA0_xmsA0_xsA1_target)
        mse_one_step_term4_target = -2 * exp_loss_diff_xms_target\
                                        * (exp_loss_yA0_target - exp_loss_yA0_xA1_target)
        
        mse_one_step_ics = np.concatenate([
            (mse_one_step_term1_source + mse_one_step_term2_source)/self.source_prevalence,
            (mse_plugin_ics + mse_one_step_term3_target + mse_one_step_term4_target)/self.target_prevalence
        ])
        print("MSE EIF", mse_plugin_ics.mean(), mse_one_step_term1_source.mean(), mse_one_step_term2_source.mean(), mse_one_step_term3_target.mean(), mse_one_step_term4_target.mean())
        logging.info("VAR EIF %f %f %f %f %f %f", mse_plugin_ics.var(), mse_one_step_term1_source.var(), mse_one_step_term2_source.var(), mse_one_step_term3_target.var(), mse_one_step_term4_target.var(), mse_one_step_ics.var())
        logging.info("MSE EIF %f %f %f %f %f", mse_plugin_ics.mean(), mse_one_step_term1_source.mean(), mse_one_step_term2_source.mean(), mse_one_step_term3_target.mean(), mse_one_step_term4_target.mean())

        return {
            'plugin': InferenceResult(
                mse_plugin_ics.mean(),
                np.concatenate([np.zeros(self.test_source_n), mse_plugin_ics/self.target_prevalence]),
            ),
            'onestep': InferenceResult(
                mse_one_step_ics.mean(),
                mse_one_step_ics,
            )
        }

    def _compute_ustatistic_terms(self, targetXW, targetY, subgroup_mask, shifted_outcome_model, density_model_ustatistic, reps):
        """Approximates double average in eq. (4) by subsampling the inner average
        """
        source_prob = self._get_source_probability(targetXW)
        binned_source_prob = self._bin_probabilities(source_prob)

        # Make consecutive copies of each row reps times and 
        # shuffle XW_-s columns to get reps copies of tildeXW
        targetXW = np.repeat(targetXW, repeats=reps, axis=0)
        if reps == targetXW.shape[0]:
            tildeXW = np.tile(targetXW, reps=(reps, 1))
        else:
            perm = np.random.permutation(targetXW.shape[0])
            tildeXW = targetXW[perm, :]
        tildeXW[:, subgroup_mask] = targetXW[:, subgroup_mask]
        
        targetY = np.repeat(targetY, repeats=reps, axis=0)
        binned_source_prob = np.repeat(binned_source_prob, repeats=reps, axis=0)
        
        tilde_target_loss = self.loss_func(tildeXW, targetY)
        tilde_source_prob = self._get_source_probability(tildeXW)  # outcome probability on shuffled tildeXW
        binned_tilde_source_prob = self._bin_probabilities(tilde_source_prob)
        tilde_target_prob = self._get_target_probability(tildeXW)
        is_same_bin = np.isclose(binned_source_prob, binned_tilde_source_prob).astype(int)
        print("is_same_bin", is_same_bin.mean(), is_same_bin[:10])
        print("got source and target probs")

        exp_loss_target_on_tildeXW = compute_risk(self.loss_func, tildeXW, tilde_target_prob)
        tilde_shifted_prob = shifted_outcome_model.predict_proba(
            np.concatenate([
                tildeXW[:, subgroup_mask],
                self._bin_probabilities(tilde_source_prob)[:, np.newaxis]
            ], axis=1)
        )[:,1]
        exp_loss_shifted_on_tildeXW = compute_risk(self.loss_func, tildeXW, tilde_shifted_prob)
        
        delta_exp_loss_on_tildeXW = exp_loss_target_on_tildeXW - exp_loss_shifted_on_tildeXW
        print("all risks computed...")
        
        if subgroup_mask.sum() == 0:
            density_feats = np.concatenate([
               binned_source_prob[:, np.newaxis],
               is_same_bin[:, np.newaxis]
            ], axis=1)
        else:
            density_feats = np.concatenate([
                tildeXW,
                binned_source_prob[:, np.newaxis],
                is_same_bin[:, np.newaxis]
            ], axis=1)
        prob_target_xms_xs_q = to_safe_prob(
            density_model_ustatistic.predict_proba(density_feats)[:,1], 
            eps=self.min_ustats_dens_ratio_cutoff
        )
        logging.info("DENSITY USTAT prob_target_xms_xs_q mean %f min %f max %f", np.mean(prob_target_xms_xs_q), np.min(prob_target_xms_xs_q), np.max(prob_target_xms_xs_q))
        odds_target_xms_xs_q = prob_target_xms_xs_q/(1 - prob_target_xms_xs_q) # TODO: multiply by ratio of number of data points in tilde by number in original
        odds_target_xms_xs_q = odds_target_xms_xs_q * is_same_bin.astype(int)
        print("ODDS Xs", odds_target_xms_xs_q.max(), odds_target_xms_xs_q.min())

        weights = odds_target_xms_xs_q
        ustatistics_term = delta_exp_loss_on_tildeXW * (tilde_target_loss - exp_loss_shifted_on_tildeXW) * odds_target_xms_xs_q

        # Average consecutive reps rows
        weights = np.mean(weights.reshape((-1, reps)), axis=1)
        logging.info("WEIGHTS %f %f %f", weights.mean(), weights.min(), weights.max())
        ustatistics_term = np.mean(ustatistics_term.reshape((-1, reps)), axis=1)

        weights = np.clip(weights, a_min=self.min_ustats_dens_ratio_cutoff, a_max=None)
        ustatistic_ics = -2 * ustatistics_term
        logging.info("RAW V-statistic mean %f", -2 * ustatistics_term.mean())
        logging.info("re-weighted V-statistic mean %f", -2 * (ustatistics_term/weights).mean())
        return ustatistic_ics

    def _conditional_shift_loss_detailed_denom(self, sourceXW: np.ndarray, sourceY: np.ndarray, targetXW: np.ndarray, targetY: np.ndarray) -> Dict[str, InferenceResult]:
        """
        Estimate and get EIFs for conditional outcome detailed denominator
        Estimator is defined in eq. (5) and EIF in eq. (49)
        Shared across multiple subgroup_masks
        Returns:
            _type_: _description_
        """
        logging.info("conditional outcome detailed denom")
        # Evaluate loss on test data from source
        source_loss = self.loss_func(sourceXW, sourceY)
        target_loss = self.loss_func(targetXW, targetY)

        source_prob = self._get_source_probability(targetXW)
        target_prob = self._get_target_probability(targetXW)
        exp_loss_source_on_targetXW = compute_risk(self.loss_func, targetXW, source_prob)
        exp_loss_target_on_targetXW =  compute_risk(self.loss_func, targetXW, target_prob)

        # Density model
        odds_target_xw_on_sourceXW = self._get_density_ratio_xw(sourceXW)
        logging.info("COND OUTCOME OUTCOME ODDS XW %f %f", odds_target_xw_on_sourceXW.min(), odds_target_xw_on_sourceXW.max())
        # Outcome models
        source_prob_on_sourceXW = self._get_source_probability(sourceXW)
        target_prob_on_sourceXW = self._get_target_probability(sourceXW)
        exp_loss_source_on_sourceXW = compute_risk(self.loss_func, sourceXW, source_prob_on_sourceXW)
        exp_loss_target_on_sourceXW = compute_risk(self.loss_func, sourceXW, target_prob_on_sourceXW)

        # plugin
        denom_plugin_ics = np.concatenate([
            np.zeros(self.test_source_n),
            np.power(exp_loss_target_on_targetXW - exp_loss_source_on_targetXW, 2)/self.target_prevalence  # eq. 33 in methods.tex
        ])

        # onestep
        denom_correction_target = 2 * (exp_loss_target_on_targetXW - exp_loss_source_on_targetXW) * (target_loss - exp_loss_target_on_targetXW)  # eq. 34 in methods.tex
        denom_correction_source = -2 * (exp_loss_target_on_sourceXW - exp_loss_source_on_sourceXW) * (source_loss - exp_loss_source_on_sourceXW) * odds_target_xw_on_sourceXW  # eq. 35 in methods.tex
        denom_onestep_ics = denom_plugin_ics + np.concatenate([
            denom_correction_source/self.source_prevalence,
            denom_correction_target/self.target_prevalence
        ])
        print("np.mean odds_target_xw_on_sourceXW", np.mean(odds_target_xw_on_sourceXW))
        logging.info("COND OUTCOME DENOM onestep %f %f %f", denom_plugin_ics.mean(), denom_correction_target.mean(), denom_correction_source.mean())

        return {
            'plugin': InferenceResult(
                denom_plugin_ics.mean(),
                denom_plugin_ics - denom_plugin_ics.mean()
            ),
            'onestep': InferenceResult(
                denom_onestep_ics.mean(),
                denom_onestep_ics - denom_onestep_ics.mean()
            )}

    def _conditional_shift_loss_detailed_numerator(self, targetXW: np.ndarray, targetY: np.ndarray, subgroup_mask: np.ndarray) -> Dict[str, InferenceResult]:
        """
        Create shifted outcome distribution by replacing source outcome distribution in place of the non-shifted feature subset in the graph, 
        value is determined by loss difference between shifted and source outcome distribution
        Estimator is defined in eq. (4) and EIF in eq. (37)
        """
        logging.info("conditional outcome detailed numerator %s", subgroup_mask)
        # check baseline variables all included
        assert np.all(subgroup_mask[self.w_mask])

        # TRAIN for plugin
        # Generate Q(x,w) for x,w from target
        source_prob_train = self._get_source_probability(self.targetXW_train)
        print("source_prob train", source_prob_train.min(), source_prob_train.max())

        # Fit manipulated outcome model p(y|xs,w,q) for target y and source q
        shifted_outcome_model = self._estimate_shifted_outcome_model(source_prob_train, subgroup_mask)
        # If the target model is very close to the shift model, then we just return zero
        # This is a bit hacky for now, but good enough for shapley computation
        if self.do_grid_search and not self.is_oracle and (self.target_outcome_model.best_score_ < shifted_outcome_model.best_score_):
            logging.info("TARGET OUTCOME MODEL NOT BETTER THAN SHIFT OUTCOME MODEL. RETURN ZERO")
            return {
                'plugin': InferenceResult(
                    0,
                    np.zeros(self.test_target_n + self.test_source_n)
                ),
                'onestep': InferenceResult(
                    0,
                    np.zeros(self.test_target_n + self.test_source_n)
                )
            }

        density_model_ustatistic = self._estimate_density_model_ustatistic(source_prob_train, subgroup_mask)

        # EVALUATE for numerator plugin
        # Evaluate loss on test data from target
        source_prob = self._get_source_probability(targetXW)
        target_prob = self._get_target_probability(targetXW)
        print("SOURCE BINNS", np.unique(self._bin_probabilities(source_prob), return_counts=True))
        # print("TARGET BINNS", np.unique(self._bin_probabilities(target_prob), return_counts=True))
        shifted_prob = shifted_outcome_model.predict_proba(
            np.concatenate([
                targetXW[:, subgroup_mask],
                self._bin_probabilities(source_prob)[:, np.newaxis]
            ], axis=1)
        )[:,1]

        exp_loss_source_on_targetXW = compute_risk(self.loss_func, targetXW, source_prob)
        exp_loss_target_on_targetXW =  compute_risk(self.loss_func, targetXW, target_prob)
        exp_loss_shifted_on_targetXW = compute_risk(self.loss_func, targetXW, shifted_prob)
        
        # plugin
        mse_plugin_ics = np.concatenate([
            np.zeros(self.test_source_n),
            np.power(exp_loss_target_on_targetXW - exp_loss_shifted_on_targetXW, 2)/self.target_prevalence # eq. 89 in random_notes.tex
        ])
        logging.info("plugin %f", mse_plugin_ics.mean())

        print("loss_diff", (exp_loss_shifted_on_targetXW-exp_loss_source_on_targetXW).mean(), "tot_loss_diff", (exp_loss_target_on_targetXW-exp_loss_source_on_targetXW).mean())
        print("diff_loss_proj, tot_loss_proj", mse_plugin_ics.mean(), np.power(exp_loss_target_on_targetXW - exp_loss_source_on_targetXW, 2).mean())
        
        # EVALUATE for numerator one-step correction
        target_loss = self.loss_func(targetXW, targetY)
        logging.info("TERM 2 mean diff %f", target_loss.mean() - exp_loss_target_on_targetXW.mean())
        mse_one_step_term2 = 2 * (exp_loss_target_on_targetXW - exp_loss_shifted_on_targetXW) * (target_loss - exp_loss_target_on_targetXW)  # eq. 93 in random_notes.tex        

        logging.info("mse_one_step_term2 min %f max %f mean %f sd %f", mse_one_step_term2.min(), mse_one_step_term2.max(), mse_one_step_term2.mean(), np.sqrt(np.var(mse_one_step_term2)))

        mse_one_step_term3 = self._compute_ustatistic_terms(
            targetXW,
            targetY,
            subgroup_mask,
            shifted_outcome_model,
            density_model_ustatistic,
            reps=min(self.reps_ustatistics, targetY.shape[0]))

        mse_one_step_ics = mse_plugin_ics + np.concatenate([
            np.zeros(self.test_source_n),
            (mse_one_step_term2 + mse_one_step_term3)/self.target_prevalence])
        print("MSE EIF", mse_plugin_ics.mean(), mse_one_step_term2.mean(), mse_one_step_term3.mean())
        logging.info("COND OUTCOME NUMER onestep var %f %f %f %f", mse_plugin_ics.var(), mse_one_step_term2.var(), mse_one_step_term3.var(), mse_one_step_ics.var())
        logging.info("COND OUTCOME NUMER onestep mean %f %f %f", mse_plugin_ics.mean(), mse_one_step_term2.mean(), mse_one_step_term3.mean())

        return {
            'plugin': InferenceResult(
                mse_plugin_ics.mean(),
                mse_plugin_ics - mse_plugin_ics.mean(),
            ),
            'onestep': InferenceResult(
                mse_one_step_ics.mean(),
                mse_one_step_ics - mse_one_step_ics.mean(),
            )
        }
    
    def get_aggregate_terms(self) -> Tuple[dict, dict, dict]:
        if self.agg_res_y is not None:
            # if we already have computed these terms
            return self.agg_res_w, self.agg_res_x, self.agg_res_y

        self.agg_res_w = self._baseline_shift_loss_aggregate(self.sourceXW_test, self.sourceW_test, self.sourceY_test, self.targetW_test)
        self.agg_res_x = self._marginal_shift_loss_aggregate(
            self.sourceXW_test, self.sourceW_test, self.sourceY_test, self.targetXW_test, self.targetW_test
        )
        self.agg_res_y = self._conditional_shift_loss_aggregate(
            self.targetXW_test, self.targetY_test, self.sourceXW_test, self.sourceY_test
        )
        logging.info(
            "AGGREGATE DECOMP loss: BASELINE plug-in %f eif %f; COVARIATE plug-in %f eif %f; OUTCOME plug-in %f eif %f",
            self.agg_res_w['plugin'].estim,
            self.agg_res_w['onestep'].estim,
            self.agg_res_x['plugin'].estim,
            self.agg_res_x['onestep'].estim,
            self.agg_res_y['plugin'].estim,
            self.agg_res_y['onestep'].estim,
            )
        return self.agg_res_w, self.agg_res_x, self.agg_res_y

    def get_cond_cov_decomp_term(self, subgroup_mask: np.ndarray) -> dict:
        logging.info("COND COV DECOMP %s", subgroup_mask)
        subgroup_mask_tuple = tuple(subgroup_mask.tolist())
        if subgroup_mask_tuple in self.detail_cond_cov_res_dict:
            # Check if computed already
            return self.detail_cond_cov_res_dict[subgroup_mask_tuple]
        
        if self.cond_cov_denom_res is None:
            # Check if computed denom already
            self.cond_cov_denom_res = self._marginal_shift_loss_detailed_denom(self.sourceW_test, self.sourceXW_test, self.sourceY_test, self.targetW_test, self.targetXW_test)

        print("subgroup_mask[~self.w_mask]", subgroup_mask[~self.w_mask])
        if np.all(subgroup_mask[~self.w_mask]):
            # Special case of full set
            zero_inf_res = InferenceResult(0, np.zeros(self.test_source_n + self.test_target_n))
            cond_cov_numer_res = {
                'plugin': zero_inf_res,
                'onestep': zero_inf_res,
            }
        elif subgroup_mask[~self.w_mask].sum() == 0:
            # Special case of empty set
            cond_cov_numer_res = {
                'plugin': self.cond_cov_denom_res['plugin'],
                'onestep': self.cond_cov_denom_res['onestep'],
            }
        else:
            cond_cov_numer_res = self._marginal_shift_loss_detailed_numerator(
                self.sourceW_test,
                self.sourceXW_test,
                self.sourceY_test,
                self.targetW_test,
                self.targetXW_test,
                subgroup_mask)
        
        logging.info(
            "CONDITIONAL COVARIATE var exp %s: plug-in %f eif %f, tot var plug-in %f eif %f",
            subgroup_mask,
            cond_cov_numer_res['plugin'].estim,
            cond_cov_numer_res['onestep'].estim,
            self.cond_cov_denom_res['plugin'].estim,
            self.cond_cov_denom_res['onestep'].estim,
            )
        
        explained_ratio_plugin_res = self.apply_delta_method_explained_ratio(cond_cov_numer_res['plugin'], self.cond_cov_denom_res['plugin'])
        explained_ratio_onestep_res = self.apply_delta_method_explained_ratio(cond_cov_numer_res['onestep'], self.cond_cov_denom_res['onestep'])
        self.detail_cond_cov_res_dict[subgroup_mask_tuple] = {
            "explained_ratio": {
                'plugin': explained_ratio_plugin_res,
                'onestep': explained_ratio_onestep_res,
            },
            'numer': {
                "plugin": cond_cov_numer_res['plugin'],
                'onestep': cond_cov_numer_res['onestep'],
            },
            'denom':{
                'plugin': self.cond_cov_denom_res['plugin'],
                'onestep': self.cond_cov_denom_res['onestep'],
            },
        }
        for k, res_dict in self.detail_cond_cov_res_dict[subgroup_mask_tuple].items():
            for plot_k, v in res_dict.items():
                logging.info("COND COV DECOMP %s %s: %s", k, plot_k, to_str_inf_res(v))
        return self.detail_cond_cov_res_dict[subgroup_mask_tuple]
        
    def get_cond_outcome_decomp_term(self, subgroup_mask: np.ndarray) -> dict:
        logging.info("COND OUTCOME DECOMP %s", subgroup_mask)
        subgroup_mask_tuple = tuple(subgroup_mask.tolist())

        if subgroup_mask_tuple in self.detail_cond_outcome_res_dict:
            # Check if computed already
            return self.detail_cond_outcome_res_dict[subgroup_mask_tuple]
        if self.cond_outcome_denom_res is None:
            # Check if computed denom already
            self.cond_outcome_denom_res = self._conditional_shift_loss_detailed_denom(self.sourceXW_test, self.sourceY_test, self.targetXW_test, self.targetY_test)

        if np.all(subgroup_mask[~self.w_mask]):
            # Special case of full set
            zero_inf_res = InferenceResult(0, np.zeros(self.test_source_n + self.test_target_n))
            cond_outcome_numer_res = {
                'plugin': zero_inf_res,
                'onestep': zero_inf_res,
            }
        else:
            cond_outcome_numer_res = self._conditional_shift_loss_detailed_numerator(
                self.targetXW_test,
                self.targetY_test,
                subgroup_mask)
            
        logging.info(
            "CONDITIONAL OUTCOME var exp %s: plug-in %f eif %f, tot var plug-in %f eif %f",
            subgroup_mask,
            cond_outcome_numer_res['plugin'].estim,
            cond_outcome_numer_res['onestep'].estim,
            self.cond_outcome_denom_res['plugin'].estim,
            self.cond_outcome_denom_res['onestep'].estim,
            )
    
        explained_ratio_plugin_res = self.apply_delta_method_explained_ratio(cond_outcome_numer_res['plugin'], self.cond_outcome_denom_res['plugin'])
        explained_ratio_onestep_res = self.apply_delta_method_explained_ratio(cond_outcome_numer_res['onestep'], self.cond_outcome_denom_res['onestep'])
        self.detail_cond_outcome_res_dict[subgroup_mask_tuple] = {
            "explained_ratio": {
                'plugin': explained_ratio_plugin_res,
                'onestep': explained_ratio_onestep_res,
            },
            'numer': {
                "plugin": cond_outcome_numer_res['plugin'],
                'onestep': cond_outcome_numer_res['onestep'],
            },
            'denom':{
                'plugin': self.cond_outcome_denom_res['plugin'],
                'onestep': self.cond_outcome_denom_res['onestep'],
            },
        }
        for k, res_dict in self.detail_cond_outcome_res_dict[subgroup_mask_tuple].items():
            for plot_k, v in res_dict.items():
                logging.info("COND OUTCOME DECOMP %s %s: %s", k, plot_k, to_str_inf_res(v))
                print("COND OUTCOME DECOMP %s %s: %s", k, plot_k, to_str_inf_res(v))
        return self.detail_cond_outcome_res_dict[subgroup_mask_tuple]
