"""
Run comparators on data files
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
from comparators import *
from common import read_csv, LossEvaluator

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
        "--decomposition",
        type=str,
        choices=['Cond_Outcome', 'Cond_Cov'],
        help="type of detailed decomposition for comparators, outcome or covariate",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=['accuracy', 'brier'],
        help="loss function",
    )
    parser.add_argument(
        "--do-grid-search",
        action="store_true",
        help="do grid search in outcome and density ratio modeling"
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
        "--w-indices",
        type=str,
        nargs="?",
        const="",
        help="comma separated list of indices of baseline variables, 0-indexed",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="gamma for shapley subset sampling",
    )
    parser.add_argument(
        "--mdl-file",
        type=str,
        default="_output/mdl.pkl",
        help="trained model which is to be explained",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="_output/resultcomparatorJOB.csv",
        help="results file to store importances from comparators",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logcomparatorJOB.txt",
        help="log file",
    )
    args = parser.parse_args()
    args.source_data = args.source_data_template.replace("JOB",
            str(args.job_idx))
    args.target_data = args.target_data_template.replace("JOB",
            str(args.job_idx))
    args.source_data_generator = args.source_data_generator.replace("JOB", 
            str(args.job_idx)) if args.source_data_generator is not None else None
    args.target_data_generator = args.target_data_generator.replace("JOB", 
            str(args.job_idx)) if args.target_data_generator is not None else None
    args.result_file = args.result_file.replace("JOB",
            str(args.job_idx))
    args.log_file = args.log_file_template.replace("JOB",
            str(args.job_idx))
    args.w_indices = np.array(list(map(int, args.w_indices.split(",")))) if args.w_indices!="" else np.array([])
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

    target_dg = None
    source_dg = None
    if args.target_data_generator:
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
    
    source_loss = loss_func(source_loader._get_XW(), source_loader._get_Y()).mean()
    target_loss = loss_func(target_loader._get_XW(), target_loader._get_Y()).mean()
    logging.info("TOTAL LOSS DIFF: %f - %f = %f", target_loss, source_loss, target_loss - source_loss)

    if args.decomposition == "Cond_Cov":
        COMPARATORS = [
            WuShiftExplain,
            MeanChange
        ]
        if args.target_data_generator:
            COMPARATORS += [OaxacaBlinderExplanation]
    elif args.decomposition == "Cond_Outcome":
        COMPARATORS = [
            ParametricChangeExplanation,
            ParametricAccExplanation,
            RandomForestAccExplanation,
            GBTAccExplanation,
            # RandomForestExplanation,
            OaxacaBlinderExplanation,
        ]

    # Get comparator explanations
    res_df = None
    for Comparator in COMPARATORS:
        print("COMPARATOR", Comparator)
        if Comparator is WuShiftExplain:
            explainer = WuShiftExplain(
                ml_mdl=ml_mdl,
                source_data_loader=source_loader,
                target_data_loader=target_loader,
                source_generator=source_dg,
                target_generator=target_dg,
                loss_func=loss_func,
                do_grid_search=args.do_grid_search,
                gamma=args.gamma
            )
        else:
            explainer = Comparator(
                ml_mdl=ml_mdl,
                source_data_loader=source_loader,
                target_data_loader=target_loader,
                source_generator=source_dg,
                target_generator=target_dg,
                loss_func=loss_func
            )
        explainer.do_decomposition()

        res_df = pd.concat(
            [res_df, explainer.summary()]
        )

    res_df["job"] = args.job_idx
    res_df["mdl"] = args.mdl_file
    res_df["nsource"] = source_loader.num_n
    res_df["ntarget"] = target_loader.num_n
    res_df.to_csv(args.result_file, index=False)

if __name__ == "__main__":
    main()
