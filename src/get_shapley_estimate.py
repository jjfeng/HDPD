"""
Code for computing aggregate and detailed decomposition by Shapley, both estimates and confidence interval
"""

import os
import pickle
import logging
import argparse
import itertools
import scipy
import csv

import pandas as pd
import numpy as np

from data_loader import DataLoader
from decomp_explainer import ExplainerShapInference, ExplainerInference
from estimate_datashifter import EstimateShiftExplainerBaseVariables
from common import read_csv, LossEvaluator, COND_OUTCOME_STR, COND_COV_STR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain performance differences across contexts"
    )
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx for data replicates",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help="random seed offset from job idx",
    )
    parser.add_argument(
        "--is-oracle",
        action="store_true",
        help="whether to use an oracle model for the outcome model",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.9,
        help="significance level of confidence intervals",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=['accuracy', 'brier'],
        help="loss function",
    )
    parser.add_argument(
        "--w-indices",
        type=str,
        nargs="?",
        const="",
        help="comma separated list of indices of baseline variables, 0-indexed"
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        default="",
        choices=["", COND_OUTCOME_STR, COND_COV_STR, f"{COND_COV_STR}+{COND_OUTCOME_STR}"],
        help="type of detailed decomposition, either conditional covariate or outcome or or both",
    )
    parser.add_argument(
        "--source-data-template",
        type=str,
        default="_output/source_dataJOB.csv",
        help="source data",
    )
    parser.add_argument(
        "--target-data-template",
        type=str,
        default="_output/target_dataJOB.csv",
        help="target data",
    )
    parser.add_argument(
        "--mdl-file",
        type=str,
        default="_output/mdl.pkl",
        help="trained model which is to be explained",
    )
    parser.add_argument(
        "--do-aggregate",
        action="store_true",
        help="compute aggregate decomposition",
    )
    parser.add_argument(
        "--do-grid-search",
        action="store_true",
        help="do grid search in outcome and density ratio modeling",
    )
    parser.add_argument(
        "--do-clipping",
        action="store_true",
        help="clip density ratios for ustatistics at a threshold",
    )
    parser.add_argument(
        "--reps-ustatistics",
        type=int,
        default=2000,
        help="number of ustatistic samples for one-step estimator",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1,
        help="param to control number of subset samples in shapley approximation, higher value means more subsets",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="number of bins for binning source outcome probability",
    )
    parser.add_argument(
        "--combos",
        type=str,
        help="list of subset masks split by + sign, each mask separated by , sign",
    )
    parser.add_argument(
        "--gridsearch-polynom-lr",
        action="store_true",
        help="whether to add logistic regression with polynomial features in grid search",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="_output/resultestimateJOB.csv",
        help="results file to store value estimates, confidence intervals",
    )
    parser.add_argument(
        "--explainer-out-file",
        type=str,
        default="_output/shapler.pkl",
        help="shapley explanations output file",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logJOB.txt",
        help="log file",
    )
    args = parser.parse_args()
    args.source_data = args.source_data_template.replace("JOB",
            str(args.job_idx))
    args.target_data = args.target_data_template.replace("JOB",
            str(args.job_idx))
    args.result_file = args.result_file.replace("JOB",
            str(args.job_idx))
    args.log_file = args.log_file_template.replace("JOB",
            str(args.job_idx))
    args.explainer_out_file = args.explainer_out_file.replace("JOB",
            str(args.job_idx))
    args.w_indices = np.array(list(map(int, args.w_indices.split(",")))) if args.w_indices!="" else np.array([])
    args.decomposition = args.decomposition.split("+")
    assert (args.w_indices.size!=0) or (COND_COV_STR not in args.decomposition), "w_indices should be non-empty in conditional covariate"
    if args.combos is not None:
        combos = args.combos.split("+")
        args.combos = [np.array(list(map(int, combo.split(","))), dtype=bool) for combo in combos]
        print(args.combos)
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    
    sourceXW, sourceY = read_csv(args.source_data)
    sourceXW, sourceY = sourceXW.to_numpy(), sourceY.to_numpy()
    targetXW, targetY = read_csv(args.target_data)
    targetXW, targetY = targetXW.to_numpy(), targetY.to_numpy()

    source_loader = DataLoader(sourceXW, sourceY, args.w_indices)
    target_loader = DataLoader(targetXW, targetY, args.w_indices)
    logging.info("NUM P %d", source_loader.num_p)

    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)

    loss_func = LossEvaluator(args.loss, ml_mdl).get_loss
    
    source_loss = loss_func(source_loader._get_XW(), source_loader._get_Y()).mean()
    target_loss = loss_func(target_loader._get_XW(), target_loader._get_Y()).mean()
    logging.info("TOTAL LOSS DIFF: %f - %f = %f", target_loss, source_loss, target_loss - source_loss)

    # Get aggregate decomposition
    shift_explainer = EstimateShiftExplainerBaseVariables(
        source_loader=source_loader,
        target_loader=target_loader,
        loss_func=loss_func,
        ml_mdl=ml_mdl,
        num_bins=args.num_bins,
        do_grid_search=args.do_grid_search,
        do_clipping=args.do_clipping,
        gridsearch_polynom_lr=args.gridsearch_polynom_lr,
        is_oracle=args.is_oracle,
        reps_ustatistics=args.reps_ustatistics)

    if args.combos is not None:
        # Evaluate certain variable subsets
        explainer = ExplainerInference(
            num_obs = source_loader.num_n,
            num_p = source_loader.num_p,
            shift_explainer=shift_explainer,
            detailed_lst=args.decomposition,
            combos=args.combos,
            do_aggregate=args.do_aggregate
        )
    else:
        # Do full decomposition with shapley values, estimates and confidence intervals
        explainer = ExplainerShapInference(
            num_obs = source_loader.num_n,
            num_p = source_loader.num_p,
            shift_explainer=shift_explainer,
            do_aggregate=args.do_aggregate,
            detailed_lst=args.decomposition,
            gamma = args.gamma,
        )
    explainer.do_decomposition()

    res_df = explainer.summary(ci_level=args.ci_level)
    res_df["job"] = args.job_idx
    res_df["mdl"] = args.mdl_file
    res_df["nsource"] = source_loader.num_n
    res_df["ntarget"] = target_loader.num_n
    res_df.to_csv(args.result_file, index=False)

    with open(args.explainer_out_file, 'wb') as f:
        pickle.dump(explainer, f)

if __name__ == "__main__":
    main()
