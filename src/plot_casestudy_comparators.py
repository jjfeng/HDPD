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
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        choices=['Cond_Outcome', 'Cond_Cov']
    )
    parser.add_argument(
        "--plot-aggregate",
        action="store_true",
    )
    parser.add_argument(
        "--w-indices",
        type=str,
    )
    parser.add_argument(
        "--summary-files-our",
        type=str,
    )
    parser.add_argument(
        "--summary-files-comparators",
        type=str,
    )
    parser.add_argument(
        "--feature-names-file",
        type=str,
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default="_output/vi_ci.png",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="_output/vi_ci.csv",
    )
    args = parser.parse_args()
    print("SUMMARY FILES", args.summary_files_our, args.summary_files_comparators)
    args.summary_files_our = args.summary_files_our.split(",")
    args.summary_files_comparators = args.summary_files_comparators.split(",")
    args.plot_file = args.plot_file.replace("JOB",
            str(args.job_idx))
    args.result_file = args.result_file.replace("JOB",
            str(args.job_idx))
    args.w_indices = np.array(list(map(int, args.w_indices.split(","))))
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
    df = df.sort_values(by=['component', 'rank'])
    g = sns.pointplot(
        y='vars_name', x='value_scale', hue='component', data=df, 
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
    num_p = (feature_names_df.y_axis_name == "Variable").sum()
    num_w = len(args.w_indices)
    w_mask = np.zeros(num_p + num_w, dtype=bool)
    w_mask[args.w_indices] = True

    df_comp = concat_files_to_df(args.summary_files_comparators, 1)
    x_vars = ["X%d" % (i+1) for i in np.where(~w_mask)[0].tolist()]
    x_vars_map = dict(zip(
        x_vars,
        ["X%d" % (i+1) for i in range(sum(~w_mask))]
    ))

    # Remap X variable names to the ones in detailed decomp
    df_comp['vars'] = df_comp['vars'].map(x_vars_map)
    df_comp['rank'] = df_comp.value.rank(ascending=False)
    print(df_comp)

    df_our = concat_files_to_df(args.summary_files_our, 1)
    df_our = df_our[
        (df_our["est"]=="onestep") & (df_our["level"]=="detail")
    ]
    df_our['value'] = df_our.value.clip(lower=0)
    df_our['rank'] = df_our.value.rank(ascending=False)
    df = pd.concat([
        df_comp, df_our
    ])
    print(df)
    df = df.merge(feature_names_df, on="vars")
    df = df[df.decomp == args.decomposition]
    
    # scale decompositions from other methods
    df['value'] = df.value.abs()
    normalizer_df = df.groupby(['component','level','decomp'])['value'].sum().reset_index().rename({"value": "normalizer"}, axis=1)
    df = df.merge(normalizer_df, on=['component','level','decomp'])
    df['value_scale'] = df.value/df.normalizer

    df['component'] = df.component.replace({"explained_ratio": "Proposed detailed decomp"})

    # Creating plot
    uniq_subplots = df[['level', 'decomp']].drop_duplicates()
    num_subplots = uniq_subplots.shape[0]
    print("num_subplots", num_subplots)
    
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots * 8))
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
