import os
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import COND_COV_STR, COND_OUTCOME_STR
from common import min_max_scale

MARKERS = ['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'V']

def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain performance differences across contexts"
    )
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx",
    )
    parser.add_argument(
        "--oracle-file",
        type=str,
        help="oracle results file",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        choices=['Cond_Outcome', 'Cond_Cov'],
        help="type of detailed decomposition, either conditional outcome or covariate",
    )
    parser.add_argument(
        "--plot-aggregate",
        action="store_true",
        help="plot aggregate decomposition",
    )
    parser.add_argument(
        "--source-data-generator",
        type=str,
        help="source data generator in pickle file",
    )
    parser.add_argument(
        "--summary-files-our",
        type=str,
        help="output of aggregate and detailed decompositions",
    )
    parser.add_argument(
        "--summary-files-comparators",
        type=str,
        help="output of importance of comparators",
    )
    parser.add_argument(
        "--feature-names-file",
        type=str,
        help="feature mappings file e.g. X1 to feature name, csv file",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default="_output/vi_ci.png",
        help="plot of aggregate decomposition and variable importance from detailed decompositions, proposed approach and comparators",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="_output/vi_ci.csv",
        help="output file of plot data",
    )
    args = parser.parse_args()
    print("SUMMARY FILES", args.summary_files_our, args.summary_files_comparators)
    args.summary_files_our = args.summary_files_our.split(",")
    args.summary_files_comparators = args.summary_files_comparators.split(",")
    args.plot_file = args.plot_file.replace("JOB",
            str(args.job_idx))
    args.result_file = args.result_file.replace("JOB",
            str(args.job_idx))
    args.source_data_generator = args.source_data_generator.replace("JOB", 
            str(args.job_idx))
    return args

def concat_files_to_df(files, num_jobs):
    all_res = []
    for file in files:
        if num_jobs is not None:
            for job_idx in range(num_jobs):
                file_jobidx = file.replace("JOB", str(job_idx+1))
                if os.path.exists(file_jobidx):
                    all_res.append(pd.read_csv(file_jobidx, delimiter=',', quotechar='"'))
                else:
                    print("FILE MISSING", file_jobidx)
        else:
            all_res.append(pd.read_csv(file, delimiter=',', quotechar='"'))
    return pd.concat(all_res)

def plot_est_and_ci(df, ax):
    g = sns.pointplot(
        y='vars', x='value_scale', hue='component', data=df, 
        markers=MARKERS[:len(df['component'].unique())], scale=2,
        join=False, orient='h', ax=ax
        )
    level_str = df.level.unique()[0]
    if level_str == "agg":
        ax.set_title("Aggregate")
        ax.set_xlabel("Decomposition Value", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
    else: 
        decomp_str = df.decomp.unique()[0]
        if decomp_str == COND_COV_STR:
            ax.set_title("Detailed: Conditional covariate")
        else:
            ax.set_title("Detailed: Conditional outcome")
        
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        g.legend(fontsize=10)
        ax.set_xlabel("Attribution", fontsize=12)
        ax.set_ylabel("Variable", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(0,1)
    # ax.set_ylabel(df.y_axis_name.iloc[0], fontsize=12)

def main():
    args = parse_args()

    feature_names_df = pd.read_csv(args.feature_names_file)

    with open(args.source_data_generator, "rb") as f:
        source_dg = pickle.load(f)

    w_vars = ["X%d" % (i+1) for i in source_dg.w_indices.tolist()]
    x_vars = ["X%d" % (i+1) for i in np.where(~source_dg.w_mask)[0].tolist()]
    x_vars_map = dict(zip(
        x_vars,
        ["Z%d" % (i+1) for i in range(sum(~source_dg.w_mask))]
    ))
    print("X VARS MAP", x_vars_map)

    df_comp = concat_files_to_df(args.summary_files_comparators, 1)
    df_comp = df_comp[df_comp.component.isin([
        "RandomForestAccExplanation",
        "ParametricChangeExplanation",
        "ParametricAccExplanation",
        "OaxacaBlinderExplanation",
        "WuShiftExplain",
        "MeanChange"
    ])]
    print(df_comp)
    # Remove baseline variables
    df_comp = df_comp[~df_comp['vars'].isin(w_vars)]

    # Remap X variable names to the ones in detailed decomp
    df_comp['vars'] = df_comp['vars'].map(x_vars_map)
    print(df_comp)

    df_our = concat_files_to_df(args.summary_files_our, 1)
    df_our = df_our[
        df_our["est"]=="plugin" # only use plugin because it is close to one-step
    ]
    df_our['vars'] = df_our.vars.map({
        f'X{i+1}':f'Z{i+1}' for i in range(sum(~source_dg.w_mask))
    })
    df = pd.concat([
        df_our, df_comp
    ])
    df = df.merge(feature_names_df, on="vars")
    df = df[df.decomp == args.decomposition]
    
    # scale decompositions from other methods
    df['value'] = df.value.abs()
    normalizer_df = df.groupby(['component','level','decomp'])['value'].sum().reset_index().rename({"value": "normalizer"}, axis=1)
    df = df.merge(normalizer_df, on=['component','level','decomp'])
    df['value_scale'] = df.value/df.normalizer

    df['component'] = df.component.replace({
        "explained_ratio": "HDPD",
        "ParametricChangeExplanation": "ParametricChange",
        "ParametricAccExplanation": "ParametricAcc",
        "RandomForestAccExplanation": "RandomForestAcc",
        "OaxacaBlinderExplanation": "Oaxaca-Blinder",
        "WuShiftExplain": "WuShift",
    })
    
    # Creating plot
    uniq_subplots = df[['level', 'decomp']].drop_duplicates()
    num_subplots = uniq_subplots.shape[0]
    print("num_subplots", num_subplots)
    
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots * 3))
    sns.set_context('paper', font_scale=2)
    for i in range(num_subplots):
        plot_key = uniq_subplots.iloc[i]
        mask = (df.level == plot_key.level) & (df.decomp == plot_key.decomp)
        print("MASK", plot_key.level, plot_key.decomp)
        plot_est_and_ci(
            df[mask],
            axs[i] if num_subplots > 1 else axs
        )

    plt.tight_layout()
    plt.savefig(args.plot_file)
    print(args.plot_file)

    df.to_csv(args.result_file, index=False)

if __name__ == "__main__":
    main()
