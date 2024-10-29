"""
Shift explainer wrappers and base classes
"""
import time
import logging
from typing import Tuple, Dict
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from shapley import ShapleyInference
from common import *

InferenceResult = namedtuple("InferenceResult", ["estim", "ic"])
def to_str_inf_res(inf_res: InferenceResult) -> str:
    return f"Inference res Estim: %f SE: %f (VAR: %f)" % (inf_res.estim, np.sqrt(np.var(inf_res.ic)/inf_res.ic.shape[0]), np.var(inf_res.ic))

class ConstantModel:
    def __init__(self, val):
        self.val = val
    def predict(self, testX):
        return self.val * np.ones(testX.shape[0])

class BaseShiftExplainer:
    num_bins = 4
    do_grid_search = False
    is_oracle = True

    def _baseline_shift_loss_aggregate(self) -> dict[str, InferenceResult]:
        raise NotImplementedError()

    def _marginal_shift_loss_aggregate(self) -> Dict[str, InferenceResult]:
        raise NotImplementedError()

    def _conditional_shift_loss_aggregate(self) -> dict[str, InferenceResult]:
        raise NotImplementedError()

    def _get_source_probability(self, X):
        raise NotImplementedError()

    def _get_target_probability(self, X):
        raise NotImplementedError()

    def _marginal_shift_loss_detailed(self) -> dict[str, InferenceResult]:
        raise NotImplementedError()

    def _make_max_feats_list(self, num_feats, min_num_feats=1) -> list:
        max_feats = list(range(min_num_feats, num_feats + 1, max(1, (num_feats + 1)//4)))
        if (len(max_feats)==0) or (max_feats[-1] != num_feats):
            return max_feats + [None]
        else:
            return max_feats

    def _estimate_mu_yA0_xA0_fn_w(self):
        """
        Expected loss on source as a function of w
        """
        # Regress exp loss from fitted outcome model on W on source data
        # TODO: is this better than just regressing on the loss itself?
        source_prob = self._get_source_probability(self.sourceXW_train)
        exp_loss_source_on_sourceXW = compute_risk(
            self.loss_func, self.sourceXW_train, source_prob
        )

        if self.sourceW_train.shape[1]!=0:
            exp_outcome_model = get_outcome_model(
                is_binary=False,
                do_grid_search=self.do_grid_search,
                is_oracle=self.is_oracle,
                max_feats=self._make_max_feats_list(self.sourceW_train.shape[1])
            )
            exp_outcome_model.fit(self.sourceW_train, exp_loss_source_on_sourceXW)
            if self.do_grid_search:
                logging.info("_estimate_mu_yA0_xA0_fn_w %s", exp_outcome_model.best_params_)
        else:
            exp_outcome_model = ConstantModel(np.mean(exp_loss_source_on_sourceXW))
        return exp_outcome_model

    def _bin_probabilities(self, source_prob):
        bin_source_probability = (
            self.num_bins > 0
        )  # return as it is for non-positive values of num_bins
        if bin_source_probability:
            binned_prob = binning_prob(source_prob, self.num_bins)
            return convert_prob_to_logit(binned_prob)
        else:
            return source_prob

    def _project_model(self, trainX: np.ndarray, trainY: np.ndarray):
        """Get E[Y|X] using a regression model
        Returns:
            projection_model: regression model if trained, otherwise outputs mean of trainY
        """
        if trainX.shape[1]!=0:
            projection_model = get_outcome_model(
                is_binary=False,
                do_grid_search=self.do_grid_search,
                is_oracle=self.is_oracle,
                max_feats=self._make_max_feats_list(trainX.shape[1])
            )
            projection_model.fit(trainX, trainY)
            return projection_model
        else:
            return ConstantModel(np.mean(trainY))

    def _conditional_shift_loss_detailed_denom(self) -> Dict[str, InferenceResult]:
        raise NotImplementedError()

    def _conditional_shift_loss_detailed_numerator(self) -> Dict[str, InferenceResult]:
        raise NotImplementedError()

    def get_aggregate_terms(self) -> Tuple[dict, dict, dict]:
        if self.agg_res_y is not None:
            # if we already have computed these terms
            return self.agg_res_w, self.agg_res_x, self.agg_res_y

        self.agg_res_w = self._baseline_shift_loss_aggregate(
            self.sourceXW_test, self.sourceW_test, self.sourceY_test, self.targetW_test
        )
        self.agg_res_x = self._marginal_shift_loss_aggregate(
            self.sourceXW_test,
            self.sourceW_test,
            self.sourceY_test,
            self.targetXW_test,
            self.targetW_test,
        )
        self.agg_res_y = self._conditional_shift_loss_aggregate(
            self.targetXW_test, self.targetY_test, self.sourceXW_test, self.sourceY_test
        )
        return self.agg_res_w, self.agg_res_x, self.agg_res_y

    def apply_delta_method_explained_ratio(
        self,
        cond_outcome_numer_res: InferenceResult,
        cond_outcome_denom_res: InferenceResult,
    ) -> InferenceResult:
        ratio_ic = (
            cond_outcome_numer_res.ic / cond_outcome_denom_res.estim
            - cond_outcome_denom_res.ic * cond_outcome_numer_res.estim / np.power(cond_outcome_denom_res.estim, 2)
        )
        return InferenceResult(
            1 - cond_outcome_numer_res.estim / cond_outcome_denom_res.estim,
            ratio_ic)

    def get_cond_cov_decomp_term(self, subgroup_mask: np.ndarray):
        raise NotImplementedError()

    def get_cond_outcome_decomp_term(
        self, subgroup_mask: np.ndarray
    ) -> Dict[str, InferenceResult]:
        raise NotImplementedError()


class ExplainerInference:
    """Runs aggregate and detailed decomposition for the given list of subsets in combos
    """
    def __init__(
        self,
        shift_explainer: BaseShiftExplainer,
        num_obs: int,
        num_p: int,
        combos: list,
        detailed_lst: list[str] = [COND_COV_STR, COND_OUTCOME_STR],
        do_aggregate: bool = False,
    ):
        self.shift_explainer = shift_explainer
        self.num_p = num_p
        self.num_obs = num_obs
        self.do_aggregate = do_aggregate
        self.detailed_lst = detailed_lst
        self.combos = combos

        self.agg_res_w = {}
        self.agg_res_x = {}
        self.agg_res_y = {}

        self.detailed_cond_cov_lst = []
        self.detailed_cond_outcome_lst = []

    def do_decomposition(self):
        if self.do_aggregate:
            (
                self.agg_res_w,
                self.agg_res_x,
                self.agg_res_y,
            ) = self.shift_explainer.get_aggregate_terms()
        if COND_COV_STR in self.detailed_lst:
            self.detailed_cond_cov_lst = []
            self.detailed_cond_cov_dict = {"numer": [], "denom": []}
            for subgroup_mask in self.combos:
                logging.info("COND COV SUBGROUP %s", subgroup_mask)
                ratio_inf_res = self.shift_explainer.get_cond_cov_decomp_term(subgroup_mask)
                self.detailed_cond_cov_lst.append(ratio_inf_res["explained_ratio"])
                self.detailed_cond_cov_dict["numer"].append(ratio_inf_res["numer"])
                self.detailed_cond_cov_dict["denom"].append(ratio_inf_res["denom"])
        if COND_OUTCOME_STR in self.detailed_lst:
            self.detailed_cond_outcome_lst = []
            self.detailed_cond_outcome_dict = {"numer": [], "denom": []}
            for subgroup_mask in self.combos:
                logging.info("COND OUTCOME SUBGROUP %s", subgroup_mask)
                ratio_inf_res = self.shift_explainer.get_cond_outcome_decomp_term(
                    subgroup_mask
                )
                self.detailed_cond_outcome_lst.append(ratio_inf_res["explained_ratio"])
                self.detailed_cond_outcome_dict["numer"].append(ratio_inf_res["numer"])
                self.detailed_cond_outcome_dict["denom"].append(ratio_inf_res["denom"])

    def _get_ci_summary(
        self,
        estim,
        se,
        component: str,
        level: str,
        decomp: str,
        vars: list[str],
        est: str,
        ci_level=0.95,
    ) -> pd.DataFrame:
        ## get alpha from the level
        a = (1 - ci_level) / 2.0
        a = np.array([a, 1 - a])
        ## calculate the quantiles
        fac = norm.ppf(a)

        return pd.DataFrame(
            {
                "value": estim,
                "se": se,
                "ci_lower": estim + fac[0] * se,
                "ci_upper": estim + fac[1] * se,
                "ci_widths": fac[1] * se,
                "component": component,
                "level": level,
                "decomp": decomp,
                "vars": vars,
                "est": est,
            }
        )

    def get_aggregate_res_w(self, plot_key: str = "onestep", ci_level=0.95):
        inf_res = self.agg_res_w[plot_key]
        source_var = np.var(inf_res.ic[:self.shift_explainer.test_source_n]) / self.shift_explainer.test_source_n * np.power(self.shift_explainer.source_prevalence, 2)
        target_var = np.var(inf_res.ic[self.shift_explainer.test_source_n:]) / self.shift_explainer.test_target_n * np.power(self.shift_explainer.target_prevalence, 2)
        se = np.sqrt(source_var + target_var)
        return self._get_ci_summary(
            np.array([inf_res.estim]),
            np.array([se]),
            component="agg",
            decomp="agg",
            level="agg",
            vars="W",
            est=plot_key,
            ci_level=ci_level,
        )

    def get_aggregate_res_x(self, plot_key: str = "onestep", ci_level=0.95):
        inf_res = self.agg_res_x[plot_key]
        source_var = np.var(inf_res.ic[:self.shift_explainer.test_source_n]) / self.shift_explainer.test_source_n * np.power(self.shift_explainer.source_prevalence, 2)
        target_var = np.var(inf_res.ic[self.shift_explainer.test_source_n:]) / self.shift_explainer.test_target_n * np.power(self.shift_explainer.target_prevalence, 2)
        se = np.sqrt(source_var + target_var)
        return self._get_ci_summary(
            np.array([inf_res.estim]),
            np.array([se]),
            component="agg",
            decomp="agg",
            level="agg",
            vars="X",
            est=plot_key,
            ci_level=ci_level,
        )

    def get_aggregate_res_y(self, plot_key: str = "onestep", ci_level=0.95):
        inf_res = self.agg_res_y[plot_key]
        source_var = np.var(inf_res.ic[:self.shift_explainer.test_source_n]) / self.shift_explainer.test_source_n * np.power(self.shift_explainer.source_prevalence, 2)
        target_var = np.var(inf_res.ic[self.shift_explainer.test_source_n:]) / self.shift_explainer.test_target_n * np.power(self.shift_explainer.target_prevalence, 2)
        se = np.sqrt(source_var + target_var)
        return self._get_ci_summary(
            np.array([inf_res.estim]),
            np.array([se]),
            component="agg",
            decomp="agg",
            level="agg",
            vars="Y",
            est=plot_key,
            ci_level=ci_level,
        )

    def get_detailed_res_cond_outcome(
        self, plot_key: str = "onestep", ci_level=0.95
    ) -> pd.DataFrame:
        if COND_OUTCOME_STR not in self.detailed_lst:
            return None

        estims = [
            res[plot_key].estim for res in self.detailed_cond_outcome_dict["numer"]
        ]
        ses = np.sqrt(
            [
                np.var(res[plot_key].ic) / res[plot_key].ic.shape[0]
                for res in self.detailed_cond_outcome_dict["numer"]
            ]
        )
        numer_inf_res = self._get_ci_summary(
            estims,
            ses,
            ci_level=ci_level,
            component="numer",
            level="detail",
            decomp=COND_OUTCOME_STR,
            vars=[str(tuple(combo)) for combo in self.combos],
            est=plot_key,
        )

        estims = [res[plot_key].estim for res in self.detailed_cond_outcome_lst]
        ses = np.sqrt(
            [
                np.var(res[plot_key].ic) / res[plot_key].ic.shape[0]
                for res in self.detailed_cond_outcome_lst
            ]
        )
        ratio_inf_res = self._get_ci_summary(
            estims,
            ses,
            ci_level=ci_level,
            component="explained_ratio",
            level="detail",
            decomp=COND_OUTCOME_STR,
            vars=[str(tuple(combo)) for combo in self.combos],
            est=plot_key,
        )

        estims = [
            res[plot_key].estim for res in self.detailed_cond_outcome_dict["denom"]
        ]
        ses = np.sqrt(
            [
                np.var(res[plot_key].ic) / res[plot_key].ic.shape[0]
                for res in self.detailed_cond_outcome_dict["denom"]
            ]
        )
        denom_inf_res = self._get_ci_summary(
            estims,
            ses,
            ci_level=ci_level,
            component="denom",
            level="detail",
            decomp=COND_OUTCOME_STR,
            vars=[str(tuple(combo)) for combo in self.combos],
            est=plot_key,
        )
        return pd.concat([ratio_inf_res, numer_inf_res, denom_inf_res]).reset_index()

    def get_detailed_res_cond_cov(
        self, plot_key: str = "onestep", ci_level=0.95
    ) -> dict:
        if COND_COV_STR not in self.detailed_lst:
            return None

        estims = [res[plot_key].estim for res in self.detailed_cond_cov_lst]
        ses = np.sqrt(
            [
                np.var(res[plot_key].ic) / res[plot_key].ic.shape[0]
                for res in self.detailed_cond_cov_lst
            ]
        )
        ratio_inf_res = self._get_ci_summary(
            estims,
            ses,
            ci_level=ci_level,
            component="explained_ratio",
            level="detail",
            decomp=COND_COV_STR,
            vars=[str(tuple(combo)) for combo in self.combos],
            est=plot_key,
        )

        estims = [res[plot_key].estim for res in self.detailed_cond_cov_dict["numer"]]
        ses = np.sqrt(
            [
                np.var(res[plot_key].ic) / res[plot_key].ic.shape[0]
                for res in self.detailed_cond_cov_dict["numer"]
            ]
        )
        numer_inf_res = self._get_ci_summary(
            estims,
            ses,
            ci_level=ci_level,
            component="numer",
            level="detail",
            decomp=COND_COV_STR,
            vars=[str(tuple(combo)) for combo in self.combos],
            est=plot_key,
        )

        estims = [res[plot_key].estim for res in self.detailed_cond_cov_dict["denom"]]
        ses = np.sqrt(
            [
                np.var(res[plot_key].ic) / res[plot_key].ic.shape[0]
                for res in self.detailed_cond_cov_dict["denom"]
            ]
        )
        denom_inf_res = self._get_ci_summary(
            estims,
            ses,
            ci_level=ci_level,
            component="denom",
            level="detail",
            decomp=COND_COV_STR,
            vars=[str(tuple(combo)) for combo in self.combos],
            est=plot_key,
        )
        return pd.concat([ratio_inf_res, numer_inf_res, denom_inf_res]).reset_index()

    def summary(self, ci_level: float = 0.95) -> pd.DataFrame:
        df = None
        if self.do_aggregate:
            # Collate aggregate results
            for agg_plot_key in self.agg_res_y.keys():
                agg_res_w = self.get_aggregate_res_w(agg_plot_key, ci_level)
                agg_res_x = self.get_aggregate_res_x(agg_plot_key, ci_level)
                agg_res_y = self.get_aggregate_res_y(agg_plot_key, ci_level)
                agg_df = pd.concat([agg_res_w, agg_res_x, agg_res_y])
                df = pd.concat([df, agg_df]) if df is not None else agg_df

        # Collate detailed results
        if COND_OUTCOME_STR in self.detailed_lst:
            detail_plot_keys = self.detailed_cond_outcome_lst[0].keys()
            for detail_plot_key in detail_plot_keys:
                res_cond_outcome_df = self.get_detailed_res_cond_outcome(
                    detail_plot_key, ci_level
                )
                df = (
                    pd.concat([df, res_cond_outcome_df])
                    if df is not None
                    else res_cond_outcome_df
                )

        if COND_COV_STR in self.detailed_lst:
            detail_plot_keys = self.detailed_cond_cov_lst[0].keys()
            for detail_plot_key in detail_plot_keys:
                res_cond_cov_df = self.get_detailed_res_cond_cov(
                    detail_plot_key, ci_level
                )
                df = (
                    pd.concat([df, res_cond_cov_df])
                    if df is not None
                    else res_cond_cov_df
                )
        return df.reset_index(drop=True)


class ExplainerShapInference(ExplainerInference):
    """Runs aggregate and detailed decomposition for all subsets, excluding those that have baseline variables,
    containing point estimates and confidence intervals
    """
    def __init__(
        self,
        shift_explainer,
        num_obs: int,
        num_p: int,
        detailed_lst: list[str] = [COND_OUTCOME_STR, COND_COV_STR],
        do_aggregate: bool = False,
        gamma: float = 1,
    ):
        self.shift_explainer = shift_explainer
        self.num_p = num_p
        self.num_obs = num_obs
        self.w_mask = shift_explainer.w_mask
        self.num_w = self.w_mask.sum()
        self.do_aggregate = do_aggregate
        self.detailed_lst = detailed_lst
        self.gamma = gamma

        self.agg_res_w = {}
        self.agg_res_x = {}
        self.agg_res_y = {}
        self.detailed_cond_cov_shapler = ShapleyInference(
            self.num_obs, self.num_p - self.num_w, self.get_cond_cov, self.gamma
        )
        self.detailed_cond_outcome_shapler = ShapleyInference(
            self.num_obs, self.num_p - self.num_w, self.get_cond_outcome, self.gamma
        )

    def get_cond_cov(self, s):
        st_time = time.time()
        subgroup_mask_with_w = np.ones(self.num_p, dtype=bool)
        subgroup_mask_with_w[~self.w_mask] = s
        res = self.shift_explainer.get_cond_cov_decomp_term(subgroup_mask_with_w)["explained_ratio"]
        logging.info("get_cond_cov time %s %f", s, time.time() - st_time)
        return res

    def get_cond_outcome(self, s):
        st_time = time.time()
        subgroup_mask_with_w = np.ones(self.num_p, dtype=bool)
        subgroup_mask_with_w[~self.w_mask] = s
        res = self.shift_explainer.get_cond_outcome_decomp_term(subgroup_mask_with_w)["explained_ratio"]
        logging.info("get_cond_outcome time %s %f", s, time.time() - st_time)
        return res

    def do_decomposition(self):
        if self.do_aggregate:
            (
                self.agg_res_w,
                self.agg_res_x,
                self.agg_res_y,
            ) = self.shift_explainer.get_aggregate_terms()

        for detailed_key in self.detailed_lst:
            print("DETAILED DECOMP", detailed_key)
            if detailed_key == COND_COV_STR:
                self.detailed_cond_cov_ests = (
                    self.detailed_cond_cov_shapler.get_point_est()
                )
                self.detailed_cond_cov_ses = self.detailed_cond_cov_shapler.get_ses()
            else:
                self.detailed_cond_outcome_ests = (
                    self.detailed_cond_outcome_shapler.get_point_est()
                )
                self.detailed_cond_outcome_ses = (
                    self.detailed_cond_outcome_shapler.get_ses()
                )

    def get_detailed_res_cond_outcome(self, plot_key: str = "onestep", ci_level=0.95):
        if COND_OUTCOME_STR not in self.detailed_lst:
            return None

        values = self.detailed_cond_outcome_ests[plot_key].flatten()
        ci_widths = self.detailed_cond_outcome_ses[plot_key].flatten()
        var_names = [f"X{i}" for i in range(1, self.num_p - self.w_mask.sum() + 1)]
        return self._get_ci_summary(
            values,
            ci_widths,
            ci_level=ci_level,
            component="explained_ratio",
            level="detail",
            decomp=COND_OUTCOME_STR,
            vars=["Source_bin"] + var_names,
            est=plot_key,
        )

    def get_detailed_res_cond_cov(self, plot_key: str = "onestep", ci_level=0.95):
        values = self.detailed_cond_cov_ests[plot_key].flatten()
        ci_widths = self.detailed_cond_cov_ses[plot_key].flatten()
        var_names = [f"X{i}" for i in range(1, self.num_p - self.w_mask.sum() + 1)]
        return self._get_ci_summary(
            values[1:],
            ci_widths[1:],
            ci_level=ci_level,
            component="explained_ratio",
            level="detail",
            decomp=COND_COV_STR,
            vars=var_names,
            est=plot_key,
        )

    def summary(
        self, ci_level: float = 0.95
    ) -> pd.DataFrame:
        # Collate aggregate results
        df = None
        if self.do_aggregate:
            for plot_key in ["plugin", "onestep"]:
                agg_res_w = self.get_aggregate_res_w(plot_key, ci_level)
                agg_res_x = self.get_aggregate_res_x(plot_key, ci_level)
                agg_res_y = self.get_aggregate_res_y(plot_key, ci_level)
                agg_df = pd.concat([agg_res_w, agg_res_x, agg_res_y])
                logging.info(agg_df)
                df = pd.concat([df, agg_df]) if df is not None else agg_df

        # Collate detailed results
        if COND_OUTCOME_STR in self.detailed_lst:
            for plot_key in ["plugin", "onestep"]:
                det_res_cond_outcome = self.get_detailed_res_cond_outcome(plot_key, ci_level)
                logging.info(det_res_cond_outcome)
                df = (
                    pd.concat([df, det_res_cond_outcome])
                    if df is not None
                    else det_res_cond_outcome
                )

        if COND_COV_STR in self.detailed_lst:
            for plot_key in ["plugin", "onestep"]:
                det_res_cond_cov = self.get_detailed_res_cond_cov(plot_key, ci_level)
                logging.info(det_res_cond_cov)
                df = (
                    pd.concat([df, det_res_cond_cov])
                    if df is not None
                    else det_res_cond_cov
                )
        return df.reset_index(drop=True)
