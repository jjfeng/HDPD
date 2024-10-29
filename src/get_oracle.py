import os
import pickle
import logging
import argparse
import itertools
import scipy
import csv

import pandas as pd
import numpy as np

from decomp_explainer import ExplainerShapInference, ExplainerInference
from oracle_datashifter import OracleShiftExplainerBaseVariables
from common import LossEvaluator, COND_OUTCOME_STR, COND_COV_STR

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
        "--num-obs",
        type=int,
        default=10000,
        help="number of observations for oracle",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1,
        help="param to control number of subset samples in shapley approximation, higher value means more subsets",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=['accuracy', 'brier'],
        help="loss function",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        default="",
        choices=["", COND_OUTCOME_STR, COND_COV_STR, f"{COND_COV_STR}+{COND_OUTCOME_STR}"],
        help="type of detailed decompositions, either conditional covariate or outcome or both",
    )
    parser.add_argument(
        "--do-aggregate",
        action="store_true",
        help="compute aggregate decomposition",
    )
    parser.add_argument(
        "--source-data-generator",
        type=str,
        help="source data generator in a pickle file",
    )
    parser.add_argument(
        "--target-data-generator",
        type=str,
        help="target data generator in a pickle file",
    )
    parser.add_argument(
        "--combos",
        type=str,
        help="list of subset masks split by + sign, each mask separated by , sign",
    )
    parser.add_argument(
        "--mdl-file",
        type=str,
        default="_output/mdl.pkl",
        help="trained model which is to be explained",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="number of bins for binning source outcome probability",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="_output/resultoracle.csv",
        help="results file to store value estimates, confidence intervals",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="_output/log.txt",
        help="log file",
    )
    args = parser.parse_args()
    args.decomposition = args.decomposition.split("+")
    args.source_data_generator = args.source_data_generator.replace("JOB", str(args.job_idx))
    args.target_data_generator = args.target_data_generator.replace("JOB", str(args.job_idx))
    args.result_file = args.result_file.replace("JOB", str(args.job_idx))
    args.log_file = args.log_file.replace("JOB", str(args.job_idx))
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

    with open(args.target_data_generator, "rb") as f:
        target_dg = pickle.load(f)
        print("TARGET X", target_dg.x_mean)
        print(target_dg.beta)
    with open(args.source_data_generator, "rb") as f:
        source_dg = pickle.load(f)
        print("SOURCE X", source_dg.x_mean)
        print(source_dg.beta)
    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)
    
    loss_func = LossEvaluator(args.loss, ml_mdl).get_loss

    sourceX, sourceY = source_dg.generate(args.num_obs * 2)
    targetY = target_dg._generate_Y(sourceX)
    sourceloss = loss_func(sourceX, sourceY).mean()
    targetloss = loss_func(sourceX, targetY).mean()
    logging.info("conditional LOSS DIFF: %f - %f = %f", targetloss.mean(), sourceloss.mean(), targetloss.mean() - sourceloss.mean())

    targetX = target_dg._generate_X(args.num_obs * 2)
    sourceY_targetX = source_dg._generate_Y(targetX)
    sourceloss = loss_func(sourceX, sourceY).mean()
    targetloss = loss_func(targetX, sourceY_targetX).mean()
    logging.info("marginal LOSS DIFF: %f - %f = %f", targetloss.mean(), sourceloss.mean(), targetloss.mean() - sourceloss.mean())

    sourceX, sourceY = source_dg.generate(args.num_obs * 2)
    targetX, targetY = target_dg.generate(args.num_obs * 2)
    sourceloss = loss_func(sourceX, sourceY).mean()
    targetloss = loss_func(targetX, targetY).mean()
    logging.info("TOTAL LOSS DIFF: %f - %f = %f", targetloss.mean(), sourceloss.mean(), targetloss.mean() - sourceloss.mean())

    shift_explainer = OracleShiftExplainerBaseVariables(source_generator=source_dg, target_generator=target_dg, loss_func=loss_func, ml_mdl=ml_mdl, num_bins=args.num_bins, num_obs=args.num_obs)
    if args.combos is not None:
        # Evaluate certain variable subsets
        explainer = ExplainerInference(
            num_obs = args.num_obs,
            num_p = source_dg.num_p,
            shift_explainer=shift_explainer,
            detailed_lst=args.decomposition,
            combos=args.combos,
            do_aggregate=args.do_aggregate,
        )
    else:
        # Do full decomposition with shapley values, estimates and confidence intervals
        explainer = ExplainerShapInference(
            num_obs = args.num_obs,
            num_p = source_dg.num_p,
            shift_explainer=shift_explainer,
            do_aggregate=args.do_aggregate,
            detailed_lst=args.decomposition,
            gamma = args.gamma
        )
    explainer.do_decomposition()

    res_df = explainer.summary(ci_level=0.95)
    res_df["mdl"] = args.mdl_file
    res_df.to_csv(args.result_file, index=False)
    
if __name__ == "__main__":
    main()
