import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import COND_COV_STR, COND_OUTCOME_STR

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
        "--plot-components",
        type=str,
        help="list of aggregate, detailed decompositions, and comparators to plot, comma separated",
    )
    parser.add_argument(
        "--plot-aggregate",
        action="store_true",
        help="plot aggregate decomposition",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        help="output of aggregate and detailed decompositions",
    )
    parser.add_argument(
        "--feature-names-file",
        type=str,
        help="feature mappings file e.g. X1 to feature name, csv file",
    )
    parser.add_argument(
        "--plot-cond-out-file",
        type=str,
        help="plot file of conditional outcome detailed decomposition",
    )
    parser.add_argument(
        "--plot-cond-cov-file",
        type=str,
        help="plot file of conditional covariate detailed decomposition",
    )
    parser.add_argument(
        "--plot-agg-file",
        type=str,
        help="plot file of aggregate decomposition",
    )
    args = parser.parse_args()
    args.plot_components = args.plot_components.split(",")
    args.summary_file = args.summary_file.replace("JOB",
            str(args.job_idx))
    args.plot_cond_out_file = args.plot_cond_out_file.replace("JOB",
            str(args.job_idx))
    args.plot_cond_cov_file = args.plot_cond_cov_file.replace("JOB",
            str(args.job_idx))
    args.plot_agg_file = args.plot_agg_file.replace("JOB",
            str(args.job_idx))
    return args

def plot_est_and_ci(df, ax, x_col_name="value", color='black'):
    for _, df_row in df.iterrows():
        ax.errorbar(x=df_row[x_col_name], y=df_row.vars_name, xerr=df_row.ci_widths, fmt='none', color=color, capsize=8, elinewidth=3)

def main():
    args = parse_args()

    feature_names_df = pd.read_csv(args.feature_names_file)

    df = pd.read_csv(args.summary_file)
    df = df.merge(feature_names_df, on="vars")
    df['est'] = df.est.map({
        'onestep': 'Proposed',
        'plugin': 'Plugin'
    })

    if args.oracle_file:
        oracle_df = pd.read_csv(args.oracle_file)
        oracle_df = oracle_df[oracle_df.est == "onestep"]
        oracle_df = oracle_df.merge(feature_names_df, on="vars")
        oracle_df['est'] = "Oracle"

        # oracle_df = oracle_df[["vars", "component", "level", "decomp", "value"]]
        # oracle_df = oracle_df.rename({"value": "oracle"}, axis=1)
        # df = df.merge(oracle_df, on=["vars", "component", "level", "decomp"])
        df = pd.concat([df, oracle_df]).reset_index()

    # df = df.rename({"est": "Estimator"}, axis=1)
    if args.plot_components:
        df = df[df.component.isin(args.plot_components)]
    
    # Creating plot
    uniq_subplots = df[['level', 'decomp', 'component']].drop_duplicates()
    num_subplots = uniq_subplots.shape[0]
    print("num_subplots", num_subplots)
    
    colors = sns.color_palette(n_colors=3)
    sns.set_context('paper', font_scale=2)
    for decomp_str in [COND_COV_STR, COND_OUTCOME_STR]:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,3))
        mask = (df.level == "detail") & (df.decomp == decomp_str)
        print(decomp_str, mask)
        if mask.sum() == 0:
            continue
        detail_df = df[mask]
        print(detail_df)
        detail_df['value'] = detail_df.value.clip(lower=0)
        detail_df = pd.concat([
            detail_df[detail_df.est == 'Proposed'].sort_values(by=['value'], ascending=False),
            detail_df[detail_df.est != 'Proposed']
        ])
        detail_df = pd.concat([
            detail_df[detail_df.vars_name == '{X1}'],
            detail_df[detail_df.vars_name == '{X2}'],
            detail_df[detail_df.vars_name == '{X3}'],
        ])
        print(detail_df)
        ax = sns.pointplot(
            y='vars_name',
            x='value',
            hue='est',
            data=detail_df,
            join=False,
            capsize=0.1,
            orient='h',
            ax=ax,
            markers=["o","o", "x"],
            scale=2,
            palette=colors,
            legend='auto')
        ax.set_xlim((-0.001,1))
        plot_est_and_ci(
            df[mask & (df.est == "Proposed")],
            ax,
            color=colors[0],
        )
        plot_est_and_ci(
            df[mask & (df.est == "Plugin")],
            ax,
            color=colors[1],
        )
        

        ax.legend().set_title('')
        ax.set_xlabel("Attribution Value", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylabel(detail_df.y_axis_name.iloc[0], fontsize=12)
        plt.tight_layout()
        if decomp_str == COND_COV_STR:
            ax.set_title("Detailed: Conditional covariate")
            plt.savefig(args.plot_cond_cov_file, bbox_inches='tight')
            print(args.plot_cond_cov_file)
        else:
            ax.set_title("Detailed: Conditional outcome")
            plt.savefig(args.plot_cond_out_file, bbox_inches='tight')
            print(args.plot_cond_out_file)

    plt.clf()
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,3))
    mask = df.level == "agg"
    agg_df = df[mask]
    agg_df = pd.concat([
        agg_df[agg_df.est == 'Proposed'].sort_values(by=['value'], ascending=False),
        agg_df[agg_df.est != 'Proposed']
    ])
    agg_df = pd.concat([
        agg_df[agg_df.vars_name == 'W'],
        agg_df[agg_df.vars_name == 'Z'],
        agg_df[agg_df.vars_name == 'Y'],
    ])
    print(agg_df)
    sns.pointplot(
        y='vars_name',
        x='value',
        hue='est',
        data=agg_df,
        join=False,
        capsize=0.1,
        markers=["o","o", "x"],
        scale=2,
        orient='h',
        ax=ax,
        palette=colors,
        legend='auto')
    plot_est_and_ci(
        agg_df[agg_df.est == "Proposed"],
        ax,
        color=colors[0],
    )
    plot_est_and_ci(
        agg_df[agg_df.est == "Plugin"],
        ax,
        color=colors[1],
    )
    ax.set_title("Aggregate")
    ax.legend().set_title('')
    ax.set_xlabel("Decomposition Value", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel(df.y_axis_name.iloc[0], fontsize=12)

    plt.tight_layout()
    plt.savefig(args.plot_agg_file, bbox_inches='tight')
    print(args.plot_agg_file)

if __name__ == "__main__":
    main()
