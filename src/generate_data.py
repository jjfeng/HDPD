import os
import pickle
import logging
import argparse

import pandas as pd
import numpy as np

from data_generator import DataGenerator, DataGeneratorMultiNorm, DataGeneratorSeqNorm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate data for testing")
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
        help="random seed offset",
    )
    parser.add_argument(
        "--x-scale",
        type=float,
        default=2,
        help="scale of normal distributed additive error in sequentially generated x-dist setting",
    )
    parser.add_argument(
        "--x-dist",
        type=str,
        default="unif",
        choices=["unif", "norm", "seq_norm"],
        help="type of distribution for x",
    )
    parser.add_argument(
        "--x-mean",
        type=str,
        help="x mean",
    )
    parser.add_argument(
        "--beta",
        type=str,
        help="comma separated list of coefficients for y logit"
    )
    parser.add_argument(
        "--w-indices",
        type=str,
        nargs="?",
        const="",
        help="comma separated list of indices of baseline variables, 0-indexed"
    )
    parser.add_argument(
        "--nonlinear",
        action="store_true",
        help="introduce non-linearity in outcome function by taking absolute value of first feature before computing outcome probability"
    )
    parser.add_argument(
        "--num-obs",
        type=int,
        default=100,
        help="number of observations",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/data_logJOB.txt",
        help="log file",
    )
    parser.add_argument(
        "--out-data-gen-file",
        type=str,
        default="_output/datagenJOB.pkl",
        help="output data generator file in pickle format",
    )
    parser.add_argument(
        "--out-file-template",
        type=str,
        default="_output/dataJOB.csv",
        help="output data file in csv format",
    )
    args = parser.parse_args()
    args.beta = np.array(list(map(float, args.beta.split(","))))
    args.x_mean = np.array(list(map(float, args.x_mean.split(","))))
    # NOTE: w_indices cannot be empty string ""
    args.w_indices = np.array(list(map(int, args.w_indices.split(",")))) if args.w_indices!="" else np.array([])
    args.log_file = args.log_file_template.replace("JOB",
            str(args.job_idx))
    args.out_file = args.out_file_template.replace("JOB",
            str(args.job_idx))
    args.out_data_gen_file = args.out_data_gen_file.replace("JOB",
            str(args.job_idx))
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    if args.x_dist == "unif":
        dg = DataGenerator(beta = args.beta, intercept=0, x_mean=args.x_mean, w_indices=args.w_indices, nonlinear=args.nonlinear)
    elif args.x_dist == "norm":
        dg = DataGeneratorMultiNorm(beta = args.beta, intercept=0, x_mean=args.x_mean, w_indices=args.w_indices, nonlinear=args.nonlinear)
    elif args.x_dist == "seq_norm":
        dg = DataGeneratorSeqNorm(beta = args.beta, intercept=0, x_mean=args.x_mean, w_indices=args.w_indices, nonlinear=args.nonlinear, scale=args.x_scale)

    X, y = dg.generate(args.num_obs)
    df = pd.DataFrame(X)
    df["y"] = y
    df.to_csv(args.out_file, index=False)

    if args.out_data_gen_file:
        with open(args.out_data_gen_file, "wb") as f:
            pickle.dump(dg, f)

if __name__ == "__main__":
    main()
