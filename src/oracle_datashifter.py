import time
import logging
import numpy as np
from typing import Tuple, Dict

from sklearn.base import clone
from data_generator import DataGenerator
from decomp_explainer import InferenceResult, BaseShiftExplainer, to_str_inf_res
from decomp_explainer import ConstantModel
from common import *

def return_dup_entry(inf_res) -> dict[str, InferenceResult]:
    return {
        "plugin": inf_res,
        "onestep": inf_res,
    }

class OracleShiftExplainerBaseVariables(BaseShiftExplainer):
    ustatistic_reps = 2000
    do_grid_search = True
    is_oracle = True
    def __init__(self, source_generator: DataGenerator, target_generator: DataGenerator, loss_func, ml_mdl, num_bins: int, num_obs: int=10000):
        self.source_generator = source_generator
        self.target_generator = target_generator
        self.w_mask = target_generator.w_mask
        self.loss_func = loss_func
        self.ml_mdl = ml_mdl
        self.num_bins = num_bins
        self.num_obs = num_obs
        
        # Full shift
        self.targetXW_train = self.target_generator._generate_X(num_obs * 4)
        self.sourceXW_train = self.source_generator._generate_X(num_obs * 4)
        self.targetW_train = self.target_generator._get_W(self.targetXW_train)
        self.sourceW_train = self.source_generator._get_W(self.sourceXW_train)
        self.targetX_train = self.target_generator._get_XminusW(self.targetXW_train)
        self.sourceX_train = self.source_generator._get_XminusW(self.sourceXW_train)
        print("DATA TRAIN", self.sourceXW_train.mean(axis=0), self.targetXW_train.mean(axis=0))

        # Get true ratio of densities
        self.density_model_xw = lambda xw: self.target_generator._get_density_XW(xw)/self.source_generator._get_density_XW(xw)
        self.density_model_w = lambda w: self.target_generator._get_density_W(w)/self.source_generator._get_density_W(w)
        self.density_model_xsw = lambda xw, subgroup_mask: self.target_generator._get_density_XWs(xw, subgroup_mask)/self.source_generator._get_density_XWs(xw, subgroup_mask)

        # Fit model for expected loss in source for each W in source
        self.mu_yA0_xA0_fn_w = self._estimate_mu_yA0_xA0_fn_w()
        # Fit model for expected loss in source for each W in target
        self.mu_yA0_xA1_fn_w = self._estimate_mu_yA0_xA1_fn_w()

        self.shift_models = {}
        self.detail_cond_outcome_res_dict = {}
        self.detail_cond_cov_res_dict = {}
        self.cond_outcome_denom_res = None
        self.cond_cov_denom_res = None
        self.agg_res_w = None
        self.agg_res_x = None
        self.agg_res_y = None

        self.test_source_n = num_obs
        self.test_target_n = num_obs
        self.source_prevalence = 0.5
        self.target_prevalence = 0.5
       
    def _get_source_probability(self, X):
        return self.source_generator._get_prob(X).flatten()
        
    def _get_target_probability(self, X):
        return self.target_generator._get_prob(X).flatten()
    
    def _get_density_ratio_w(self, W):
        return self.density_model_w(W)
    
    def _get_density_ratio_xw(self, XW):
        return self.density_model_xw(XW)
    
    def _get_density_ratio_xsw(self, XW, subgroup_mask):
        return self.density_model_xsw(XW, subgroup_mask)
    
    ## Marginal shift in baseline variables
    def _baseline_shift_loss_aggregate(self, sourceXW: np.ndarray) -> dict[str, InferenceResult]:
        """Loss difference when only marginal distribution of baseline variables shifts
        Estimator is defined in Section 4.1, influence function in eq. (12)
        """
        sourceW = self.source_generator._get_W(sourceXW)
        source_prob = self._get_source_probability(sourceXW)
        
        exp_loss = compute_risk(self.loss_func, sourceXW, source_prob)
        if self.sourceW_train.shape[1]!=0:
            odds_target_w = self.density_model_w(sourceW)
        else:
            odds_target_w = np.array([1])
        loss_diff = np.concatenate([
            exp_loss * (odds_target_w - 1)/self.source_prevalence,
            np.zeros(self.test_target_n)
        ])
        print("loss diff", loss_diff.mean(), odds_target_w.min(), odds_target_w.max())

        return return_dup_entry(InferenceResult(
                estim=loss_diff.mean(),
                ic=loss_diff - loss_diff.mean()
            ))
    
    def _marginal_shift_loss_aggregate(self, sourceXW: np.ndarray) -> dict[str, InferenceResult]:
        """
        Loss difference when only conditional covariate distribution of X shifts
        Estimator is defined in Section 4.1, influence function in eq. (13)
        """
        sourceW = self.source_generator._get_W(sourceXW)
        source_prob = self._get_source_probability(sourceXW)
        exp_loss_source_on_sourceXW = compute_risk(self.loss_func, sourceXW, source_prob)
        odds_target_xw = self.density_model_xw(sourceXW)
        if sourceW.shape[1]!=0:
            odds_target_w = self.density_model_w(sourceW)
        else:
            odds_target_w = np.array([1])
        logging.info("_marginal_shift_loss_aggregate odds_target_w %f %f", odds_target_w.max(), odds_target_w.min())
        loss_diff = np.concatenate([
            exp_loss_source_on_sourceXW * (odds_target_xw - odds_target_w)/self.source_prevalence,
            np.zeros(self.test_target_n)
        ])
        
        print("odds_target_xw", odds_target_xw.min(), odds_target_xw.max())
        print("odds_target_xw", odds_target_w.min(), odds_target_w.max())
        return return_dup_entry(InferenceResult(
                estim=loss_diff.mean(),
                ic=loss_diff - loss_diff.mean()
            ))
    
    def _conditional_shift_loss_aggregate(self, targetXW: np.ndarray) -> dict[str, InferenceResult]:
        """
        Loss difference when only conditional outcome distribution of Y shifts
        Estimator is defined in Section 4.1, influence function in eq. (14)
        """
        source_prob = self._get_source_probability(targetXW)
        target_prob = self._get_target_probability(targetXW)
        exp_loss_source_on_targetXW = compute_risk(self.loss_func, targetXW, source_prob)
        exp_loss_target = compute_risk(self.loss_func, targetXW, target_prob)
        loss_diff = np.concatenate([
            np.zeros(self.test_source_n),
            (exp_loss_target - exp_loss_source_on_targetXW)/self.target_prevalence
        ])

        return return_dup_entry(InferenceResult(
                estim=loss_diff.mean(),
                ic=loss_diff - loss_diff.mean()
            ))
    
    def _get_support(self):
        """
        Returns which indices are relevant to the source distribution, target distribution, and both
        Restricting to the support variables improves estimation of models and speed
        """
        target_support = ~np.isclose(self.target_generator.beta, 0)
        source_support = ~np.isclose(self.source_generator.beta, 0)
        full_support = np.logical_or(
            target_support,
            source_support
            )
        return source_support, target_support, full_support

    def _estimate_mu_yA0_xA0_fn_w(self):
        """
        Expected loss on source as a function of w, defined in Section 4
        """
        # Regress exp loss from fitted outcome model on W on source data
        tildeXW = self.targetXW_train.copy()
        tildeXW[:, ~self.w_mask] = self.source_generator._generate_Xs_Xms(
            ~self.w_mask,
            self.targetXW_train[:, self.w_mask])
        source_prob = self._get_source_probability(tildeXW)
        exp_loss_source_on_tildeXW = compute_risk(
            self.loss_func, tildeXW, source_prob
        )
        logging.info("avg loss on target with source %f", exp_loss_source_on_tildeXW.mean())
        if self.sourceW_train.shape[1]!=0:
            exp_outcome_model = get_outcome_model(
                is_binary=False,
                do_grid_search=self.do_grid_search,
                is_oracle=self.is_oracle,
                max_feats=self._make_max_feats_list(self.sourceW_train.shape[1])
            )
            exp_outcome_model.fit(self.targetW_train, exp_loss_source_on_tildeXW)
            if self.do_grid_search:
                logging.info("_estimate_mu_yA0_xA0_fn_w %s", exp_outcome_model.cv_results_)
                logging.info("_estimate_mu_yA0_xA0_fn_w %s", exp_outcome_model.best_estimator_)
        else:
            exp_outcome_model = ConstantModel(np.mean(exp_loss_source_on_tildeXW))
        return exp_outcome_model

    def _estimate_mu_yA0_xA1_fn_w(self):
        """Regress exp loss from fitted outcome model on w on target data,
        defined in eq. (8)"""
        # Expected loss on source
        source_prob = self._get_source_probability(self.targetXW_train)
        exp_loss_source_on_targetXW = compute_risk(self.loss_func, self.targetXW_train, source_prob)
        logging.info("avg exp_loss_source_on_targetXW %f", exp_loss_source_on_targetXW.mean())
        print("mu_yA0_xA1_fn_w", self.targetW_train.shape[0])
        if self.targetW_train.shape[1]!=0:
            print("mu_yA0_xA1_fn_w", self.targetW_train.shape[0])
            exp_outcome_model = self._project_model(
                trainX=self.targetW_train, trainY=exp_loss_source_on_targetXW
            )
            if self.do_grid_search:
                logging.info("_estimate_mu_yA0_xA1_fn_w %s", exp_outcome_model.cv_results_)
                logging.info("_estimate_mu_yA0_xA1_fn_w %s", exp_outcome_model.best_estimator_)
        else:
            exp_outcome_model = ConstantModel(np.mean(exp_loss_source_on_targetXW))
        return exp_outcome_model

    def _estimate_mu_yA0_xmsA0_xsA1_fn_w(self, subgroup_mask: np.ndarray):
        """
        Fit a regression model for mu_0(x,w) on xs, w in target data,
        defined in eq. (9)
        Assumes subgroup_mask contains w_indices
        """
        assert np.all(subgroup_mask[self.w_mask])
        tildeXW = self.targetXW_train.copy()
        tildeXW[:, ~subgroup_mask] = self.source_generator._generate_Xs_Xms(~subgroup_mask, self.targetXW_train[:, subgroup_mask])
        source_prob = self._get_source_probability(tildeXW)
        true_risk = compute_risk(self.loss_func, tildeXW, source_prob)
        logging.info("avg _estimate_mu_yA0_xmsA0_xsA1_fn_w %f", true_risk.mean())

        exp_outcome_model = self._project_model(
            self.targetW_train, true_risk
        )
        if self.do_grid_search:
            logging.info("_estimate_mu_yA0_xmsA0_xsA1_fn_w %s", exp_outcome_model.cv_results_)
            logging.info("_estimate_mu_yA0_xmsA0_xsA1_fn_w %s", exp_outcome_model.best_estimator_)
        return exp_outcome_model

    def _estimate_mu_yA0_xmsA0_fn_xsw(self, subgroup_mask: np.ndarray):
        """
        Fit a regression model for mu_0(x,w) on xs, w
        Assumes subgroup_mask contains w_indices
        """
        # Regress sample loss on xs, w from source data
        source_prob = self._get_source_probability(self.sourceXW_train)
        loss_on_source = compute_risk(
            self.loss_func,
            self.sourceXW_train,
            source_prob)

        exp_outcome_model = self._project_model(
            trainX=self.sourceXW_train[:, subgroup_mask], trainY=loss_on_source
        )
        if self.do_grid_search:
            logging.info("_estimate_mu_yA0_xmsA0_fn_xsw %s", exp_outcome_model.cv_results_)
            logging.info("_estimate_mu_yA0_xmsA0_fn_xsw %s", exp_outcome_model.best_estimator_)
        return exp_outcome_model
    
    def _estimate_shifted_outcome_model(self, subgroup_mask: np.ndarray):
        """Fit manipulated outcome model p(y|xs,w,q), defined in eq. (3)
        """
        _, target_support_vars, support_vars = self._get_support()
        relevant_target_support = subgroup_mask[target_support_vars]
        is_full_target_support = np.all(relevant_target_support)
        relevant_support = subgroup_mask[support_vars]
        relevant_support_key = tuple(relevant_support.tolist())
        print(subgroup_mask, relevant_support, relevant_support_key)
        
        if is_full_target_support:
            return None
        elif relevant_support_key in self.shift_models.keys():
            return self.shift_models[relevant_support_key]
        else:
            # Need to fit a shifted outcome model
            # Generate q(x,w) for x,w|a=1
            source_prob_train = self._get_source_probability(self.targetXW_train)[:, np.newaxis]
            print("SOURCE PROBABILITY TRAIN", np.unique(self._bin_probabilities(source_prob_train), return_counts=True))
            
            # Fit manipulated outcome model p(y|xs,w,q) -- uses a REGRESSION model since we already know the true probabilities
            keep_idxs = (support_vars * subgroup_mask).astype(bool)
            shifted_outcome_model_xsw = get_outcome_model(
                is_binary=False,
                do_grid_search=self.do_grid_search,
                is_oracle=self.is_oracle,
                max_feats=self._make_max_feats_list(keep_idxs.sum() + 1),
                model_args={'max_depth': 4, 'max_features': 2, 'n_estimators': 800, 'bootstrap': False if subgroup_mask.sum() == 0 else True})
            shifted_outcome_model_xsw.fit(
                np.concatenate([
                    self.targetXW_train[:, keep_idxs],
                    self._bin_probabilities(source_prob_train)], axis=1),
                self._get_target_probability(self.targetXW_train),
            )
            if self.do_grid_search:
                logging.info("CV RESULTS %s", shifted_outcome_model_xsw.cv_results_)
                print("fitted shifted outcome model", shifted_outcome_model_xsw.best_estimator_, shifted_outcome_model_xsw.best_score_)
                logging.info("fitted shifted outcome model %s %f", shifted_outcome_model_xsw.best_estimator_, shifted_outcome_model_xsw.best_score_)
            
            self.shift_models[relevant_support_key] = shifted_outcome_model_xsw

            return shifted_outcome_model_xsw

    def _estimate_density_model_ustatistic(self, source_prob_train: np.ndarray, subgroup_mask: np.ndarray):
        """Fits density model for detailed decomposition of conditional outcome shift,
        defined in Section 4.2.1 before eq. (4)
        We know what the true support, so the density model is estimated at a faster rate
        """
        binned_source_prob_train = self._bin_probabilities(source_prob_train)
        
        # Shuffle XW_{-s} to get tildeXW_{-s}, rest of features are same as XW
        perm = np.random.choice(self.targetXW_train.shape[0], size=self.targetXW_train.shape[0], replace=True)
        tilde_targetXW_train= self.targetXW_train[perm,:]
        tilde_targetXW_train[:, subgroup_mask] = self.targetXW_train[:, subgroup_mask]
        print("tilde_targetXW_train",  tilde_targetXW_train.shape, self.targetXW_train.shape, binned_source_prob_train.shape)

        # Engineer a feature that checks if the binned tilde target XW matches the original binned source prob?
        binned_tilde_source_prob_train = self._bin_probabilities(self._get_source_probability(tilde_targetXW_train))[:, np.newaxis]
        is_same_bin = (binned_tilde_source_prob_train == binned_source_prob_train).astype(int)
        print("is_same_bin", is_same_bin.mean())

        if subgroup_mask.sum() == 0:
            density_model = get_density_model(do_grid_search=self.do_grid_search, max_feats=[None])
            density_model.fit(
                np.concatenate([
                    np.concatenate([binned_source_prob_train, is_same_bin], axis=1),
                    np.concatenate([binned_source_prob_train, np.ones(binned_source_prob_train.shape)], axis=1),
                ], axis=0),
                np.concatenate([np.zeros(binned_source_prob_train.shape[0]), np.ones(binned_source_prob_train.shape[0])], axis=0),
            )
        else:
            # We know which variables are not relevant, so remove them. Density model should improve
            _, _, support_vars = self._get_support()
            density_model = get_density_model(
                do_grid_search=self.do_grid_search,
                model_args={'max_depth':6, 'max_features':4, 'n_estimators':800},
                max_feats=self._make_max_feats_list(support_vars.sum() + 2))
            density_model.fit(
                np.concatenate([
                    np.concatenate([tilde_targetXW_train[:, support_vars], binned_source_prob_train, is_same_bin], axis=1),
                    np.concatenate([self.targetXW_train[:, support_vars], binned_source_prob_train, np.ones(binned_source_prob_train.shape)], axis=1),
                ], axis=0),
                np.concatenate([np.zeros(binned_source_prob_train.shape[0]), np.ones(binned_source_prob_train.shape[0])], axis=0),
            )
        print("density model fitted")
        # p(A=0|X-s, Xs, Q)/p(A=1|X-s, Xs, Q) = p(X-s, Xs, Q, A=0)/p(X-s, Xs, Q, A=1) = p(X-s, Xs, Q | A=0)/p(X-s, Xs, Q | A=1)
        if self.do_grid_search:
            logging.info("density ustatistic CV RESULTS %s", density_model.cv_results_)
            print('selected density ustatistic', density_model.best_estimator_)
            # logging.info('VI density ustat %s', density_model.best_estimator_.steps[0][1].feature_importances_)
            logging.info('selected density ustatistic %s', density_model.best_estimator_)
        return density_model
    
    def _compute_ustatistic_terms(self, targetXW, targetY, subgroup_mask, shifted_outcome_model, density_model_ustatistic, reps=10):
        """Approximates double average in eq. (4) by subsampling the inner average
        """
        source_prob = self._get_source_probability(targetXW)[:, np.newaxis]
        binned_source_prob = self._bin_probabilities(source_prob)

        # shuffle XW_-s columns to get reps copies of tildeXW
        targetXW = np.repeat(targetXW, repeats=reps, axis=0)
        tildeXW = self.target_generator._generate_X(targetXW.shape[0])
        tildeXW[:, subgroup_mask] = targetXW[:, subgroup_mask]
        
        targetY = np.repeat(targetY, repeats=reps, axis=0)
        binned_source_prob = np.repeat(binned_source_prob, repeats=reps, axis=0)
        
        tilde_target_loss = self.loss_func(tildeXW, targetY)
        tilde_source_prob = self._get_source_probability(tildeXW)[:, np.newaxis] # outcome probability on shuffled tildeXW
        binned_tilde_source_prob = self._bin_probabilities(tilde_source_prob)
        tilde_target_prob = self._get_target_probability(tildeXW)
        is_same_bin = np.isclose(binned_source_prob, binned_tilde_source_prob).astype(float)
        print("is_same_bin", is_same_bin.mean())
        print("got source and target probs")
    
        exp_loss_target_on_tildeXW = compute_risk(self.loss_func, tildeXW, tilde_target_prob)
        shifted_tilde_prob = self._get_estimated_shifted_probability(shifted_outcome_model, subgroup_mask, tildeXW, binned_source_prob)
        exp_loss_shifted_on_tildeXW = compute_risk(self.loss_func, tildeXW, shifted_tilde_prob)
        delta_exp_loss_on_tildeXW = exp_loss_target_on_tildeXW - exp_loss_shifted_on_tildeXW
        print("all risks computed... mean diff tilde loss", np.mean(tilde_target_loss - exp_loss_shifted_on_tildeXW))
        
        _, _, support_vars = self._get_support()
        tildeXW_sub = [tildeXW[:, support_vars]] if subgroup_mask.sum() else []
        prob_target_xms_xs_q = to_safe_prob(density_model_ustatistic.predict_proba(
            np.concatenate(tildeXW_sub + [
                binned_tilde_source_prob,
                is_same_bin
            ], axis=1)
        ))[:,1]
        odds_target_xms_xs_q = prob_target_xms_xs_q/(1 - prob_target_xms_xs_q) * is_same_bin.astype(int).flatten()
        ustatistics_term = delta_exp_loss_on_tildeXW * (tilde_target_loss - exp_loss_shifted_on_tildeXW) * odds_target_xms_xs_q
        ustatistics_term = np.mean(ustatistics_term.reshape((-1, reps)), axis=1)
        obs_weights = np.mean(odds_target_xms_xs_q.reshape((-1, reps)), axis=1)
        logging.info("obs WEIGHTS %f %f %f", obs_weights.min(), obs_weights.max(), obs_weights.mean())
        
        res = -2 * ustatistics_term
        return res

    def _get_estimated_shifted_probability(self, shifted_outcome_model_xsw, subgroup_mask: np.ndarray, targetXW: np.ndarray, binned_source_prob: np.ndarray):
        if shifted_outcome_model_xsw is None:
            return self._get_target_probability(targetXW)
        else:
            print("binned_source_prob", binned_source_prob.shape)
            _, _, support_vars = self._get_support()
            keep_idxs = (subgroup_mask * support_vars).astype(bool)
            pred_shift_probs = shifted_outcome_model_xsw.predict(
                np.concatenate([targetXW[:, keep_idxs], binned_source_prob], axis=1)
            )
            logging.info("MIN MAX SHIFTED PROBS %f %f", pred_shift_probs.max(), pred_shift_probs.min())
            return to_safe_prob(pred_shift_probs)
    
    def _conditional_shift_loss_detailed_denom(self, targetXW: np.ndarray) -> Dict[str, InferenceResult]:
        """Returns the plugin estimate for conditional outcome detailed denominator
        since we do not require influence curve for oracle
        """
        source_prob = self._get_source_probability(targetXW)
        target_prob = self._get_target_probability(targetXW)
        exp_loss_source_on_targetXW = compute_risk(self.loss_func, targetXW, source_prob)
        exp_loss_target = compute_risk(self.loss_func, targetXW, target_prob)
        loss_sq_diff = np.power(exp_loss_target - exp_loss_source_on_targetXW, 2)
        logging.info("_conditional_shift_loss_detailed_denom %f", loss_sq_diff.mean())
        return return_dup_entry(InferenceResult(
                estim=loss_sq_diff.mean(),
                ic=loss_sq_diff - loss_sq_diff.mean(),
            ))
    
    def _conditional_shift_loss_detailed_numer(self, targetXW: np.ndarray, targetY: np.ndarray, subgroup_mask: np.ndarray):
        """
        Calculates the oracle value per the EIF
        Estimator is defined in eq. (4) and EIF in eq. (37)
        """
        # TRAIN
        shifted_outcome_model_xsw = self._estimate_shifted_outcome_model(subgroup_mask)

        # EVALUTE
        # Loss on source, target, and manipulated target data
        source_prob = self._get_source_probability(targetXW)[:, np.newaxis]
        binned_source_prob = self._bin_probabilities(source_prob)
        print("BINNED PROBS", np.unique(binned_source_prob, return_counts=True))
        target_prob = self._get_target_probability(targetXW)
        print("target_prob", target_prob.shape)
        shifted_prob = self._get_estimated_shifted_probability(
            shifted_outcome_model_xsw, subgroup_mask, targetXW, binned_source_prob
        )
        # print("SHIFTED PROBABILITY", np.unique(shifted_prob, return_counts=True))
        exp_loss_target = compute_risk(self.loss_func, targetXW, target_prob)
        exp_loss_xs = compute_risk(self.loss_func, targetXW, shifted_prob)
        
        plugin = np.power(exp_loss_target - exp_loss_xs, 2)
        logging.info("PLUGIN %f", plugin.mean())

        mse_correction_term3 = 0
        if shifted_outcome_model_xsw is not None:
            density_model_ustatistic = self._estimate_density_model_ustatistic(
                self._get_source_probability(self.targetXW_train)[:,np.newaxis],
                subgroup_mask)
            # Batch process the u-statistics
            batch_size = self.ustatistic_reps
            mse_correction_term3 = []
            for idx in range(0, targetY.shape[0], batch_size):
                print("IDX", idx, targetY.shape[0])
                mse_correction_term3.append(self._compute_ustatistic_terms(
                    targetXW[idx: idx + batch_size],
                    targetY[idx: idx + batch_size],
                    subgroup_mask,
                    shifted_outcome_model_xsw,
                    density_model_ustatistic,
                    reps=self.ustatistic_reps))
                logging.info("MSE CORRECTION %f", mse_correction_term3[-1].mean())
            mse_correction_term3 = np.concatenate(mse_correction_term3)
            logging.info("plugin %f mse_correction_term3 %f %s %s", plugin.mean(), np.mean(mse_correction_term3), plugin.shape, mse_correction_term3.shape)

        one_step_estim = plugin + mse_correction_term3
        return {
            "plugin": InferenceResult(
                estim=plugin.mean(),
                ic=plugin - plugin.mean()
            ),
            "onestep": InferenceResult(
                estim=one_step_estim.mean(),
                ic=one_step_estim - one_step_estim.mean()
            ),
        }

    def _conditional_cov_shift_loss_detailed_denom(self, sourceXW: np.ndarray, sourceY: np.ndarray, targetXW: np.ndarray):
        """
        Estimate and get EIFs for conditional covariate detailed denominator
        Estimator defined in eq. (10) and EIF in eq. (17) when evalauted for an empty subset
        Shared across multiple subgroup_masks
        Returns:
            _type_: _description_
        """
        logging.info("conditional covariate detailed denom")
        targetW = self.source_generator._get_W(targetXW)
        sourceW = self.source_generator._get_W(sourceXW)
        
        ## EVALUATE plugin
        exp_loss_yA0_xA0_target = self.mu_yA0_xA0_fn_w.predict(targetW)
        exp_loss_yA0_xA1_target = self.mu_yA0_xA1_fn_w.predict(targetW)
        exp_loss_diff_target = exp_loss_yA0_xA1_target - exp_loss_yA0_xA0_target
        # tot_var_plugin_estim = np.power(np.std(exp_loss_diff_target), 2)
        logging.info("exp_loss_yA0_xA1_target %s %f", exp_loss_yA0_xA1_target, np.var(exp_loss_yA0_xA1_target))
        logging.info("exp_loss_yA0_xA0_target %s %f", exp_loss_yA0_xA0_target, np.var(exp_loss_yA0_xA0_target))
        print("diff", exp_loss_diff_target, np.var(exp_loss_diff_target))
        logging.info("DIFF %s", -exp_loss_diff_target)
        denom_plugin_ics = np.concatenate([
            np.zeros(sourceW.shape[0]),
            np.power(exp_loss_diff_target, 2)
        ]) * 2
        logging.info("PLUGIN %f", denom_plugin_ics.mean())

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

        # Get density ratios
        odds_target_w = self._get_density_ratio_w(sourceW)
        odds_target_xw = self._get_density_ratio_xw(sourceXW)

        denom_onestep_term2_source = 2 * exp_loss_diff_source * (source_loss - exp_loss_yA0_source) * odds_target_xw
        denom_onestep_term3_source = -2 * exp_loss_diff_source * (source_loss - exp_loss_yA0_xA0_source) * odds_target_w
        denom_onestep_term4_target = 2 * exp_loss_diff_target * (exp_loss_yA0_target - exp_loss_yA0_xA1_target)

        denom_onestep_ics = denom_plugin_ics + np.concatenate([
            denom_onestep_term2_source + denom_onestep_term3_source,
            denom_onestep_term4_target
        ]) * 2
        logging.info("_conditional_cov_shift_loss_detailed_denom EIF %f %f %f %f", denom_plugin_ics.mean(), denom_onestep_term2_source.mean(), denom_onestep_term3_source.mean(), denom_onestep_term4_target.mean())

        return {
            'plugin': InferenceResult(
                estim=denom_plugin_ics.mean(),
                ic=denom_plugin_ics - denom_plugin_ics.mean()
            ),
            'onestep': InferenceResult(
                estim=denom_onestep_ics.mean(),
                ic=denom_onestep_ics - denom_onestep_ics.mean()
            ),
        }

    def _conditional_cov_shift_loss_detailed_numer(self, sourceXW: np.ndarray, sourceY: np.ndarray, targetXW: np.ndarray, subgroup_mask: np.ndarray) -> Dict[str, InferenceResult]:
        """
        Create shifted covariate distribution by replacing conditional covariate distribution at target in place of the non-shifted covariate subset in the graph, 
        value is determined by loss difference between shifted and source covariate distribution
        Estimator defined in eq. (10) and EIF in eq. (17)
        """
        logging.info("conditional covariate detailed numerator %s", subgroup_mask)
        # check baseline variables all included
        assert np.all(subgroup_mask[self.w_mask])

        ## FIT models
        mu_yA0_xmsA0_model = self._estimate_mu_yA0_xmsA0_fn_xsw(subgroup_mask)
        mu_yA0_xmsA0_xsA1_model = self._estimate_mu_yA0_xmsA0_xsA1_fn_w(subgroup_mask)
        
        targetW = self.source_generator._get_W(targetXW)
        sourceW = self.source_generator._get_W(sourceXW)

        ## EVALUATE plugin numerator
        # On target
        exp_loss_yA0_xA1_target = self.mu_yA0_xA1_fn_w.predict(targetW)
        exp_loss_yA0_xmsA0_xsA1_target = mu_yA0_xmsA0_xsA1_model.predict(targetW)
        exp_loss_diff_xms_target = exp_loss_yA0_xmsA0_xsA1_target - exp_loss_yA0_xA1_target
        logging.info("exp_loss_diff_xms_target %s", exp_loss_diff_xms_target)
        logging.info("exp_loss_yA0_xmsA0_xsA1_target %s", exp_loss_yA0_xmsA0_xsA1_target)
        logging.info("exp_loss_yA0_xA1_target %s", exp_loss_yA0_xA1_target)
        mse_plugin_ics = np.concatenate([
            np.zeros(sourceXW.shape[0]),
            np.power(exp_loss_diff_xms_target, 2)
        ]) * 2

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
        odds_target_xsw = self._get_density_ratio_xsw(sourceXW, subgroup_mask)
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
        
        mse_one_step_ics = mse_plugin_ics + np.concatenate([
            mse_one_step_term1_source + mse_one_step_term2_source,
            mse_one_step_term3_target + mse_one_step_term4_target
        ], axis=0) * 2
        print("MSE EIF", mse_plugin_ics.mean(), mse_one_step_term1_source.mean(), mse_one_step_term3_target.mean(), mse_one_step_term4_target.mean())
        logging.info("_conditional_cov_shift_loss_detailed_numer EIF %f %f %f %f", mse_plugin_ics.mean(), mse_one_step_term1_source.mean(), mse_one_step_term3_target.mean(), mse_one_step_term4_target.mean())

        return {
            'plugin': InferenceResult(
                estim=mse_plugin_ics.mean(),
                ic=mse_plugin_ics - mse_plugin_ics.mean(),
            ),
            'onestep': InferenceResult(
                estim=mse_one_step_ics.mean(),
                ic=mse_one_step_ics - mse_one_step_ics.mean(),
            )
        }
    

    def get_aggregate_terms(self) -> Tuple[dict, dict, dict]:
        if self.agg_res_y is not None:
            # if we already have computed these terms
            return self.agg_res_w, self.agg_res_x, self.agg_res_y
        
        sourceXW_test = self.source_generator._generate_X(self.num_obs)
        targetXW_test = self.target_generator._generate_X(self.num_obs)

        self.agg_res_w = self._baseline_shift_loss_aggregate(sourceXW_test)
        self.agg_res_x = self._marginal_shift_loss_aggregate(sourceXW_test)
        self.agg_res_y = self._conditional_shift_loss_aggregate(targetXW_test)
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
    
    def get_cond_cov_decomp_term(self, subgroup_mask: np.ndarray) -> tuple[dict[str, InferenceResult], dict[str, InferenceResult]]:
        logging.info("COND COV DECOMP %s", subgroup_mask)

        # check all W has True
        assert np.all(subgroup_mask[self.w_mask])

        subgroup_mask_tuple = tuple(subgroup_mask.tolist())
        if subgroup_mask_tuple in self.detail_cond_cov_res_dict:
            # Check if computed already
            return self.detail_cond_cov_res_dict[subgroup_mask_tuple]

        sourceXW_test, sourceY_test = self.source_generator.generate(self.num_obs)
        targetXW_test, targetY_test = self.target_generator.generate(self.num_obs)

        if self.cond_cov_denom_res is None:
            # Check if computed denom already
            self.cond_cov_denom_res = self._conditional_cov_shift_loss_detailed_denom(sourceXW_test, sourceY_test, targetXW_test)

        if np.all(subgroup_mask[~self.w_mask]):
            zero_inf_res = InferenceResult(0, np.zeros(targetXW_test.shape[0] * 2))
            cond_cov_numer_res = return_dup_entry(zero_inf_res)
        elif subgroup_mask[~self.w_mask].sum() == 0:
            # Special case of empty set
            cond_cov_numer_res = {
                'plugin': self.cond_cov_denom_res['plugin'],
                'onestep': self.cond_cov_denom_res['onestep'],
            }
        else:
            cond_cov_numer_res = self._conditional_cov_shift_loss_detailed_numer(
                sourceXW_test,
                sourceY_test,
                targetXW_test,
                subgroup_mask)
            
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


    def get_cond_outcome_decomp_term(self, subgroup_mask: np.ndarray) -> dict[str, dict]:
        """Get conditional outcome decomposition terms

        Returns:
            tuple[dict[str, InferenceResult], dict[str, InferenceResult]]: first element is the explained ratio, the second are the numerator and denominators
        """
        logging.info("COND OUTCOME DECOMP subgroup %s", subgroup_mask)
        subgroup_mask_tuple = tuple(subgroup_mask.tolist())
        if subgroup_mask_tuple in self.detail_cond_outcome_res_dict:
            # Check if computed already
            return self.detail_cond_outcome_res_dict[subgroup_mask_tuple]

        targetXW_test, targetY_test = self.target_generator.generate(self.num_obs)

        if self.cond_outcome_denom_res is None:
            # Check if computed denom already
            self.cond_outcome_denom_res = self._conditional_shift_loss_detailed_denom(targetXW_test)
        logging.info(to_str_inf_res(self.cond_outcome_denom_res['plugin']))
        logging.info(to_str_inf_res(self.cond_outcome_denom_res['onestep']))

        cond_outcome_numer_res = self._conditional_shift_loss_detailed_numer(
            targetXW_test,
            targetY_test,
            subgroup_mask)
            
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
        return self.detail_cond_outcome_res_dict[subgroup_mask_tuple]
