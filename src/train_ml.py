import os
import argparse
import pickle
import logging
import json

import pandas as pd
import numpy as np

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from common import get_n_jobs, read_csv


def parse_args():
    parser = argparse.ArgumentParser(description="train a ML algorithm")
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LogisticRegression",
        choices=[
            "RandomForestClassifier",
            "LogisticRegression",
            "LogisticRegressionLasso",
            "GradientBoostingClassifier",
            "MLPClassifier",
        ],
        help="model to train",
    )
    parser.add_argument("--param-dict-file", type=str, default="model_dict.json", help="model hyperparams in json format")
    parser.add_argument(
        "--calib-method", type=str, choices=["sigmoid", "isotonic"], default="sigmoid", help="calibration method"
    )
    parser.add_argument(
        "--do-class-weight",
        action="store_true",
        help="use sample weights for imbalanced classes while fitting",
    )
    parser.add_argument(
        "--dataset-template",
        type=str,
        default="_output/dataJOB.csv",
        help="input data",
    )
    parser.add_argument(
        "--mdl-file-template",
        type=str,
        default="_output/mdlJOB.pkl",
        help="model output in pickle format",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logJOB.txt",
        help="log file",
    )
    args = parser.parse_args()
    args.dataset_file = args.dataset_template.replace("JOB", str(args.job_idx))
    args.log_file = args.log_file_template.replace("JOB", str(args.job_idx))
    args.mdl_file = args.mdl_file_template.replace("JOB", str(args.job_idx))
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    # Generate training data
    X, y = read_csv(args.dataset_file)

    if args.do_class_weight:
        sample_weight = compute_sample_weight("balanced", y)
    else:
        sample_weight = None

    with open(args.param_dict_file, "r") as f:
        full_param_dict = json.load(f)
        param_dict = full_param_dict[args.model_type]
        # Convert 'none' from json to None
        if ('penalty' in param_dict) and ('none' in param_dict['penalty']):
            param_dict['penalty'].remove('none')
            param_dict['penalty'].append(None)

    # Train the original ML model
    n_jobs = get_n_jobs()
    print("n_jobs", n_jobs)
    if args.model_type == "GradientBoostingClassifier":
        base_mdl = GradientBoostingClassifier()
    elif args.model_type == "RandomForestClassifier":
        base_mdl = RandomForestClassifier(n_jobs=n_jobs)
    elif args.model_type == "LogisticRegression":
        base_mdl = LogisticRegression(penalty=None, max_iter=10000)
    elif args.model_type == "LogisticRegressionLasso":
        base_mdl = LogisticRegression(penalty="l1", solver="saga", max_iter=10000)
    elif args.model_type == "MLPClassifier":
        base_mdl = MLPClassifier()
    else:
        raise NotImplementedError("model type not implemented")
    if max([len(a) for a in param_dict.values()]) > 1:
        # If there is tuning to do
        grid_cv = GridSearchCV(
            estimator=base_mdl, param_grid=param_dict, cv=3, n_jobs=1, verbose=4
        )
        grid_cv.fit(
            X,
            y,
            sample_weight=sample_weight
        )
        logging.info("CV BEST SCORE %f", grid_cv.best_score_)
        logging.info("CV BEST PARAMS %s", grid_cv.best_params_)
        print(grid_cv.best_params_)
        base_mdl.set_params(**grid_cv.best_params_)
    else:
        param_dict0 = {k: v[0] for k, v in param_dict.items()}
        base_mdl.set_params(**param_dict0)
        print(base_mdl)
        logging.info(base_mdl)
    # print("MODEL OOB", mdl.oob_score_)

    if args.model_type not in ["LogisticRegression","LogisticRegressionLasso"]:
        mdl = CalibratedClassifierCV(base_mdl, cv=5, method=args.calib_method)
    else:
        mdl = base_mdl

    logging.info("training data %s", X.shape)
    mdl.fit(
        X.to_numpy(),
        y,
        sample_weight=sample_weight
    )
    if args.model_type in ["LogisticRegression","LogisticRegressionLasso"]:
        logging.info(mdl.coef_)
        logging.info(mdl.intercept_)

    with open(args.mdl_file, "wb") as f:
        pickle.dump(mdl, f)


if __name__ == "__main__":
    main()
