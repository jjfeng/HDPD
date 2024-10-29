"""Script to plot aggregate decomposition and variable importance
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import COND_COV_STR, COND_OUTCOME_STR

MARKERS = ['o', '^', 'x', '+', '*', '8', 's', 'p', 'D', 'V']

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
        "--w-indices",
        type=str,
        nargs="?",
        const="",
        help="comma separated list of indices of baseline variables, 0-indexed",
    )
    parser.add_argument(
        "--plot-components",
        type=str,
        help="list of aggregate, detailed decompositions, and comparators to plot, comma separated",
    )
    parser.add_argument(
        "--keep-feats-plot",
        type=str,
        help="list of features to plot, 0-indexed and comma separated",
    )
    parser.add_argument(
        "--comparator-file",
        type=str,
        help="output file of comparator results",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        help="output file of aggregate and detailed decompositions",
    )
    parser.add_argument(
        "--feature-names-file",
        type=str,
        help="feature mappings file e.g. X1 to feature name, csv file",
    )
    parser.add_argument(
        "--plot-detail-file",
        type=str,
        default="_output/vi_ci_detail.png",
        help="plot of variable importance from detailed decompositions, proposed approach and comparators",
    )
    parser.add_argument(
        "--plot-agg-file",
        type=str,
        default="_output/agg_ci.png",
        help="aggregate decomposition output plot",
    )
    args = parser.parse_args()
    args.plot_components = args.plot_components.split(",")
    args.keep_feats_plot = list(map(int, args.keep_feats_plot.split(",")))
    args.comparator_file = args.comparator_file.replace("JOB",
            str(args.job_idx))
    args.summary_file = args.summary_file.replace("JOB",
            str(args.job_idx))
    args.plot_agg_file = args.plot_agg_file.replace("JOB",
            str(args.job_idx))
    args.plot_detail_file = args.plot_detail_file.replace("JOB",
            str(args.job_idx))
    args.w_indices = np.array(list(map(int, args.w_indices.split(",")))) if args.w_indices!="" else np.array([])
    return args

def plot_est_and_ci(df, ax, x_col_name="value", show_component=False, color='black', draw_ci=True, linewidth=3):
    for _, df_row in df.iterrows():
        ax.errorbar(x=df_row[x_col_name], y=df_row.vars_name, xerr=df_row.ci_widths, fmt='none', color=color, capsize=5, elinewidth=linewidth)

def load_comparators(args, feature_names_df):
    num_p = (feature_names_df.y_axis_name == "Variable").sum()
    num_w = len(args.w_indices)
    w_mask = np.zeros(num_p + num_w, dtype=bool)
    if args.w_indices.size!=0:
        w_mask[args.w_indices] = True

    df_comp = pd.read_csv(args.comparator_file)
    x_vars = ["X%d" % (i+1) for i in np.where(~w_mask)[0].tolist()]
    x_vars_map = dict(zip(
        x_vars,
        ["X%d" % (i+1) for i in range(sum(~w_mask))]
    ))

    # Remap X variable names to the ones in detailed decomp
    df_comp['vars'] = df_comp['vars'].map(x_vars_map)
    return df_comp

def main():
    args = parse_args()

    feature_names_df = pd.read_csv(args.feature_names_file)
    
    df = pd.read_csv(args.summary_file)
    df['est'] = df.est.map({
        'onestep': 'HDPD debiased',
        'plugin': 'HDPD plugin'
    })
    df_comp = load_comparators(args, feature_names_df)
    df_comp['est'] = df_comp.est.map({
        'WuShiftExplain': 'WuShift',
        'MeanChange': 'MeanChange',
        'ParametricChangeExplanation': 'ParametricChange',
        'ParametricAccExplanation': 'ParametricAcc',
        'RandomForestAccExplanation': 'RandomForestAcc',
        'OaxacaBlinderExplanation': 'OaxacaBlinder',
    })
    df = pd.concat([df, df_comp])
    df = df.merge(feature_names_df, on="vars")

    if args.plot_components:
        df = df[df.component.isin(args.plot_components)]
    
    print(df[['level', 'decomp','est']])

    # Creating detail plot
    colors = sns.color_palette(n_colors=5)
    sns.set_context('paper', font_scale=2)
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
    #fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(13,5))
    mask = df.level == "detail"
    detail_df = df[mask]
    detail_df['value'] = detail_df.value.clip(lower=0)
    detail_df = pd.concat([
        detail_df[detail_df.est == 'HDPD debiased'].sort_values(by=['value'], ascending=False),
        detail_df[detail_df.est != 'HDPD debiased']
    ])
    vars_s = detail_df.vars_name.iloc[np.r_[args.keep_feats_plot]].tolist()
    vars_mask = df.vars_name.isin(vars_s)
    detail_df = detail_df[detail_df.vars_name.isin(vars_s)]

    ax = sns.pointplot(
        y='vars_name',
        x='value',
        hue='est',
        hue_order=sorted(detail_df['est'].unique()),
        markers=MARKERS[:len(detail_df['est'].unique())],
        data=detail_df,
        join=False,
        capsize=0.1,
        orient='h',
        ax=ax,
        palette=colors,
        scale=2,
        legend='auto')
    ax.set_xlim((-0.001,1.04))
    plot_est_and_ci(
        df[mask & (df.est == "HDPD debiased") & vars_mask],
        ax,
        color='black',
        show_component=args.plot_components is None,
        draw_ci=True,
        linewidth=2,
    )
    decomp_str = df[mask & vars_mask].decomp.unique()[0]
    if decomp_str == COND_COV_STR:
        ax.set_title("Detailed: Conditional covariate")
    else:
        ax.set_title("Detailed: Conditional outcome")
    
    ax.legend().set_title("")
    ax.set_xlabel("Attribution Value", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel(detail_df.y_axis_name.iloc[0], fontsize=12)
    plt.tight_layout()
    plt.savefig(args.plot_detail_file, bbox_inches='tight')
    print(args.plot_detail_file)

    plt.clf()
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,3))
    mask = df.level == "agg"
    print(df[mask])
    plot_est_and_ci(
        df[mask & (df.est == "HDPD debiased")],
        ax,
        color='black',
        show_component=args.plot_components is None,
        draw_ci=True,
        linewidth=3,
    )
    ax = sns.pointplot(
        y='vars_name',
        x='value',
        hue='est',
        hue_order=sorted(df[mask]['est'].unique()),
        markers = MARKERS[:len(df[mask]['est'].unique())],
        data=df[mask],
        join=False,
        capsize=0.1,
        orient='h',
        ax=ax,
        palette=colors,
        scale=2,
        legend='auto')
    ax.legend().set_title("")
    ax.set_title("Aggregate")
    ax.set_xlabel("Decomposition Value", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel(df.y_axis_name.iloc[0], fontsize=12)

    plt.tight_layout()
    plt.savefig(args.plot_agg_file, bbox_inches='tight')
    print(args.plot_agg_file)

if __name__ == "__main__":
    main()
