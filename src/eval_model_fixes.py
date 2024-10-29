import logging
import os
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data_loader import DataLoader
from common import read_csv, LossEvaluator, COND_OUTCOME_STR, COND_COV_STR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model fixes per explanations"
    )
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx",
    )
    parser.add_argument(
        "--mdl-file",
        type=str,
        help="trained model which is to be explained",
    )
    parser.add_argument(
        "--num-top",
        type=str,
        help="list of features in decreasing order of importance to include in revised model, comma separated",
    )
    parser.add_argument(
        "--num-weights-combined-comparator",
        type=int,
        help="number of weights between 0 and 1 to train a model on combined source and target data",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        choices=['Cond_Outcome', 'Cond_Cov'],
        help="type of detailed decomposition, either conditional outcome or covariate",
    )
    parser.add_argument(
        "--source-data-generator",
        type=str,
        help="source data generator in pickle file",
    )
    parser.add_argument(
        "--target-data-generator",
        type=str,
        help="target data generator in pickle file",
    )
    parser.add_argument(
        "--source-data-file",
        type=str,
        help="source data in csv file",
    )
    parser.add_argument(
        "--target-data-file",
        type=str,
        help="target data in csv file",
    )
    parser.add_argument(
        "--comparator-file",
        type=str,
        help="output of importance of comparators",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=2000,
        help="max number of observations to train revised model",
    )
    parser.add_argument(
        "--num-obs",
        type=int,
        default=20000,
        help="number of observations for revised model train and test",
    )
    parser.add_argument(
        "--decomp-file",
        type=str,
        help="output of aggregate and detailed decompositions",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="_output/eval_model_perf.txt",
        help="results file",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="_output/log.txt",
        help="log file",
    )
    args = parser.parse_args()
    args.num_top = list(map(int, args.num_top.split(",")))
    args.comparator_file = args.comparator_file.replace("JOB", 
            str(args.job_idx))
    args.decomp_file = args.decomp_file.replace("JOB", 
            str(args.job_idx))
    if args.source_data_file is not None:
        args.source_data_file = args.source_data_file.replace("JOB", 
            str(args.job_idx))
        args.target_data_file = args.target_data_file.replace("JOB", 
            str(args.job_idx))
    if args.source_data_generator is not None:
        args.source_data_generator = args.source_data_generator.replace("JOB", 
            str(args.job_idx))
        args.target_data_generator = args.target_data_generator.replace("JOB", 
            str(args.job_idx))
    args.result_file = args.result_file.replace("JOB",
            str(args.job_idx))
    return args

def retrain_ml_on_combined_data(ml_mdl, train_sourceX, train_sourceY, train_targetX, train_targetY, test_targetX, test_targetY, weight_source_data=0.0):
    sample_weight = np.concatenate([
        weight_source_data * np.ones(train_sourceX.shape[0]),
        (1-weight_source_data) * np.ones(train_targetX.shape[0])
    ])
    # Train same model as ml_mdl
    combined_ml_mdl = clone(ml_mdl)
    combined_ml_mdl.fit(
        np.concatenate([train_sourceX, train_targetX]),
        np.concatenate([train_sourceY, train_targetY]),
        sample_weight=sample_weight
    )

    true_test_Y = test_targetY.flatten().astype(int)
    pred_prob = combined_ml_mdl.predict_proba(test_targetX)[:,1]
    auc = roc_auc_score(true_test_Y, pred_prob)
    pred_y = combined_ml_mdl.predict(test_targetX).flatten().astype(int)
    acc = np.mean(pred_y == true_test_Y)
    log_lik = np.mean(true_test_Y * np.log(pred_prob) + (1 - true_test_Y) * np.log(1-pred_prob))
    return {'auc': auc, 'acc': acc}

def retrain_ml_with_top_explanation(feature_values, w_mask, ml_mdl, train_targetX, train_targetY, test_targetX, test_targetY, num_top_feats=1):
    # This assumes the W indices are at the beginning of the feature list
    top_feat_idxs = np.argsort(np.abs(feature_values))[-num_top_feats:] + w_mask.sum()
    train_pred_prob = ml_mdl.predict_proba(train_targetX)[:,1:]
    print("TOP FEAT", top_feat_idxs)
    retrain_feats = np.concatenate([
        np.log(train_pred_prob/(1 - train_pred_prob)),
        train_targetX[:,w_mask],
        train_targetX[:,top_feat_idxs],
    ], axis=1)
    test_pred_prob = ml_mdl.predict_proba(test_targetX)[:,1:]
    retest_feats = np.concatenate([
        np.log(test_pred_prob/(1 - test_pred_prob)),
        test_targetX[:,w_mask],
        test_targetX[:, top_feat_idxs],
    ], axis=1)

    rf = clone(ml_mdl)
    rf.fit(retrain_feats, train_targetY)

    true_test_Y = test_targetY.flatten().astype(int)
    pred_prob = rf.predict_proba(retest_feats)[:,1]
    auc = roc_auc_score(true_test_Y, pred_prob)
    pred_y = rf.predict(retest_feats).flatten().astype(int)
    acc = np.mean(pred_y == true_test_Y)
    log_lik = np.mean(true_test_Y * np.log(pred_prob) + (1 - true_test_Y) * np.log(1-pred_prob))
    return {'auc': auc, 'acc': acc}

def reweight_ml_with_top_explanation(feature_values, w_mask, ml_mdl, train_sourceX, train_sourceY, train_targetX, train_targetY, test_targetX, test_targetY, num_top_feats, max_train):
    print(feature_values)
    top_feat_idxs = np.argsort(np.abs(feature_values))[-num_top_feats:] + w_mask.sum()
    mask = w_mask.copy()
    for idx in top_feat_idxs:
        mask[idx] = True
    print("MASK", mask)
    
    # train density ratio
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
    pipeline = Pipeline([
            ('clf', RandomForestClassifier())
        ])
    kernel_model = Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('linear', LogisticRegression(penalty='l1', fit_intercept=True, solver='saga', max_iter=20000))])
    parameters = [{
        'clf': [RandomForestClassifier(n_estimators=1000, criterion="log_loss", n_jobs=-1)],
        'clf__max_depth': [6]
    },
    {
        'clf': [kernel_model],
        'clf__linear__C': [100,10,1,0.1,0.001],
    }]
    density_ratio = GridSearchCV(
        pipeline, #RandomForestClassifier(criterion="log_loss", n_estimators=1000, n_jobs=-1),
        param_grid=parameters, #{'max_depth': [4,8,10]},
        cv=3,
        scoring="neg_log_loss")
    density_ratio.fit(
        np.concatenate([train_sourceX[:, mask], train_targetX[:, mask]]),
        np.concatenate([np.zeros(train_sourceX.shape[0]), np.ones(train_sourceX.shape[0])])
    )
    print("CV RES", density_ratio.cv_results_)
    print("DENSITY RATIO", density_ratio.best_params_, train_sourceX.shape)
    pred_ratio = density_ratio.predict_proba(train_sourceX[:, mask])[:,1]/density_ratio.predict_proba(train_sourceX[:, mask])[:,0]
    # plt.hist(pred_ratio)
    # plt.show()

    # retrain model
    print("RATIO MIN MAX", pred_ratio.max(), pred_ratio.min())
    # LogisticRegressionCV(penalty='l2', cv=3, n_jobs=-1)
    ml_mdl_revised = GridSearchCV(
        RandomForestClassifier(criterion="log_loss", n_estimators=200, n_jobs=-1),
        param_grid={'max_depth': [6,8,10]},
        cv=3)
    ml_mdl_revised.fit(train_sourceX[:max_train], train_sourceY[:max_train], sample_weight=pred_ratio[:max_train])
    # print("LogisticRegressionCV COEF", ml_mdl_revised.coef_)
    # print("REWEIGHTED", ml_mdl_revised.best_params_)
    
    # evaluate model
    true_test_Y = test_targetY.flatten().astype(int)
    pred_prob = ml_mdl_revised.predict_proba(test_targetX)[:,1]
    auc = roc_auc_score(true_test_Y, pred_prob)
    pred_y = ml_mdl_revised.predict(test_targetX).flatten().astype(int)
    acc = np.mean(pred_y == true_test_Y)
    log_lik = np.mean(true_test_Y * np.log(pred_prob) + (1 - true_test_Y) * np.log(1-pred_prob))
    return {'auc': auc, 'acc': acc}

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    np.random.seed(11563)

    if args.source_data_generator is None:
        sourceXW, sourceY = read_csv(args.source_data)
        sourceXW, sourceY = sourceXW.to_numpy(), sourceY.to_numpy()
        targetXW, targetY = read_csv(args.target_data)
        targetXW, targetY = targetXW.to_numpy(), targetY.to_numpy()

        source_loader = DataLoader(sourceXW, sourceY, args.w_indices)
        target_loader = DataLoader(targetXW, targetY, args.w_indices)
        w_mask = source_loader.w_mask
        logging.info("NUM P %d", source_loader.num_p)
        raise NotImplementedError("havnet implemented evaluator for non-data generators")
    else:
        with open(args.target_data_generator, "rb") as f:
            target_dg = pickle.load(f)
        with open(args.source_data_generator, "rb") as f:
            source_dg = pickle.load(f)
        train_sourceX, train_sourceY = source_dg.generate(args.num_obs)
        train_targetX, train_targetY = target_dg.generate(args.num_obs)
        test_targetX, test_targetY = target_dg.generate(args.num_obs * 10)
        w_mask = source_dg.w_mask
        num_z = source_dg.num_p - source_dg.num_w
    
    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)

    df_comp = pd.read_csv(args.comparator_file)
    df_comp = df_comp[df_comp.component.isin([
        "RandomForestAccExplanation",
        "ParametricChangeExplanation",
        "ParametricAccExplanation",
        "OaxacaBlinderExplanation",
        "WuShiftExplain",
        "MeanChange"
    ])]

    source_pred_y = ml_mdl.predict(train_sourceX).flatten().astype(int)
    source_acc = np.mean(source_pred_y == train_sourceY.flatten())
    target_pred_y = ml_mdl.predict(train_targetX).flatten().astype(int)
    target_acc = np.mean(target_pred_y == train_targetY.flatten())
    print("SOURCE TO TARGET ACC", source_acc, target_acc)

    res_list = []
    for comparator in df_comp.component.unique():
        aucs = []
        for num_top in args.num_top:
            print(comparator)
            feature_values = df_comp.value[df_comp.component == comparator].to_numpy()[-num_z:]
            if args.decomposition == COND_OUTCOME_STR:
                aucs.append(retrain_ml_with_top_explanation(
                    feature_values,
                    w_mask,
                    ml_mdl,
                    train_targetX,
                    train_targetY,
                    test_targetX,
                    test_targetY,
                    num_top_feats=num_top,
                    ))
            else:
                aucs.append(reweight_ml_with_top_explanation(
                    feature_values,
                    w_mask,
                    ml_mdl,
                    train_sourceX,
                    train_sourceY,
                    train_targetX,
                    train_targetY,
                    test_targetX,
                    test_targetY,
                    num_top_feats=num_top,
                    max_train=args.max_train,
                    # source_dg=source_dg,
                    # target_dg=target_dg
                    ))

        res_df = pd.DataFrame({f"{k}-{i}": [v] for i, res_dict in zip(args.num_top, aucs) for k,v in res_dict.items()})
        res_df.insert(0, 'Method', comparator)
        res_list.append(res_df)
        print(comparator, res_df)
    print(pd.concat(res_list))
    print(pd.concat(res_list).to_latex(float_format="%.2f", index=False))

    print("PROPOSED")
    df_proposed = pd.read_csv(args.decomp_file)
    df_proposed = df_proposed[df_proposed.est == "plugin"]
    print(df_proposed)
    aucs = []
    for num_top in args.num_top:
        feat_vals = df_proposed.value[df_proposed.component == "explained_ratio"].to_numpy()[-num_z:]
        print(feat_vals)
        if args.decomposition == COND_OUTCOME_STR:
            aucs.append(retrain_ml_with_top_explanation(
                feat_vals,
                w_mask,
                ml_mdl,
                train_targetX,
                train_targetY,
                test_targetX,
                test_targetY,
                num_top_feats=num_top,
                ))
        else:
            aucs.append(reweight_ml_with_top_explanation(
                feat_vals,
                w_mask,
                ml_mdl,
                train_sourceX,
                train_sourceY,
                train_targetX,
                train_targetY,
                test_targetX,
                test_targetY,
                num_top_feats=num_top,
                max_train=args.max_train,
                # source_dg=source_dg,
                # target_dg=target_dg
                ))
    res_df = pd.DataFrame({f"{k}-{i}": [v] for i, res_dict in zip(args.num_top, aucs) for k,v in res_dict.items()})
    res_df.insert(0, 'Method', "proposed")
    res_list.append(res_df)

    print(pd.concat(res_list))
    print(pd.concat(res_list).to_latex(float_format="%.2f", index=False))

    print("RETRAIN ON REWEIGHTED SOURCE AND TARGET")
    weights = np.linspace(0, 1, args.num_weights_combined_comparator, endpoint=False)
    for weight in weights:
        res_dict = retrain_ml_on_combined_data(
            ml_mdl,
            train_sourceX,
            train_sourceY,
            train_targetX,
            train_targetY,
            test_targetX,
            test_targetY,
            weight_source_data=weight
        )

        res_df = pd.DataFrame({f"{k}-1": [v] for k,v in res_dict.items()})
        if weight==0:
            res_df.insert(0, 'Method', f"target model on Z,W ({weight:.2f})")
        else:
            res_df.insert(0, 'Method', f"weighted source-target model on Z,W ({weight:.2f})")
        res_list.append(res_df)
    print(pd.concat(res_list))
    print(pd.concat(res_list).to_latex(float_format="%.2f", index=False))
    with open(args.result_file, "w") as fw:
        fw.write(pd.concat(res_list).to_latex(float_format="%.2f", index=False))
        
    
if __name__ == "__main__":
    main()
