import os
import logging
import argparse

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(
        description="concatenate csvs"
    )
    parser.add_argument(
        "--result-files-oracle",
        type=str,
        help="oracle results file",
    )
    parser.add_argument(
        "--result-files-estimate",
        type=str,
        help="output of aggregate and detailed decompositions",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        help="significance level of confidence intervals",
    )
    parser.add_argument(
        "--plot-components",
        type=str,
        help="list of aggregate, detailed decompositions, and comparators to plot, comma separated",
        default="agg,explained_ratio"
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="whether to have legend",
    )
    parser.add_argument(
        "--feature-names-file",
        type=str,
        help="feature mappings file e.g. X1 to feature name, csv file",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        help="number of jobs to read all estimates from data replicates",
    )
    parser.add_argument(
        "--scale-bias",
        help="scale the bias plot by root n",
        action="store_true",
    )
    parser.add_argument(
        "--summary-csv-file",
        type=str,
        default="_output/summary.csv",
        help="output file of aggregate and detailed decompositions",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="_output/log.txt",
        help="log file",
    )
    parser.add_argument(
        "--csv-file-oracle",
        type=str,
        help="output file of oracle result",
    )
    parser.add_argument(
        "--csv-file-estimate",
        type=str,
        help="output file of estimates from all data replicates",
    )
    parser.add_argument(
        "--biasplot-file-agg",
        type=str,
        default="_output/plot_agg.pdf",
        help="bias plot for aggregate decompositions",
    )
    parser.add_argument(
        "--biasplot-file-detail",
        type=str,
        default="_output/plot_detail.pdf",
        help="bias plot for detailed decompositions",
    )
    parser.add_argument(
        "--varplot-file-agg",
        type=str,
        default="_output/plot_agg.pdf",
        help="variance plot for aggregate decompositions",
    )
    parser.add_argument(
        "--varplot-file-detail",
        type=str,
        default="_output/plot_detail.pdf",
        help="variance plot for detailed decompositions",
    )
    parser.add_argument(
        "--coverage-detail-file",
        type=str,
        help="coverage plot for detailed decompositions",
    )
    parser.add_argument(
        "--coverage-agg-file",
        type=str,
        help="coverage plot for aggregate decompositions",
    )
    args = parser.parse_args()
    args.plot_components = args.plot_components.split(",")
    args.result_files_oracle = args.result_files_oracle.split(",")
    args.result_files_estimate = args.result_files_estimate.split(",")
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

def plot_coverage_vs_n(data, oracle_res, ci_level, plotfile, show_component=False):
    sns.set_context('paper', font_scale=4)
    plt.clf()
    print("DATA AMOUNT", len(data))
    data["est_estimate"] = data["est_estimate"].str.replace("eif", "one-step correct")
    if show_component:
        data["decomp_component"] = data["decomp"] + data["component"]
    else:
        data["decomp_component"] = data["decomp"]
    oracle_res = dict(oracle_res[["decomp", "value"]].values)

    facet_height = 5  # height of each facet in inches
    aspect_ratio = 1.5  # width = aspect_ratio * height
    ax = sns.relplot(
        data=data,
        x="nsource",
        y="covered",
        hue="est_estimate",
        row="vars_name",
        kind="line",
        markers=True,
        linewidth=3,
        legend=False,
        height=facet_height,
        aspect=aspect_ratio,
        facet_kws={'sharex': True, 'sharey': True}
    )
    print("CI LEVEL", ci_level)
    custom_ticks = data.nsource.unique()
    ax.set(ylim=(0, 1), xticks=custom_ticks)
    ax.set_xticklabels(labels=custom_ticks, rotation=45)

    for axis in ax.axes.flat:
        axis.axhline(ci_level, ls="--", color="black")
        
    ax.set_titles(row_template="Subset = {row_name}") 
    
    # update to pretty axes and titles
    ax.set_axis_labels(
        "n",
        "Coverage"
    )
    ax.fig.tight_layout()
    plt.savefig(plotfile, bbox_inches="tight")
    print("coverage plot ", plotfile)

def plot_bias_vs_n(data, oracle_res, plotfile, scale_bias=True):
    plt.clf()
    plt.figure(figsize=(12,10))
    sns.set_context('paper', font_scale=2)
    data["est_estimate"] = data["est_estimate"].str.replace("eif", "one-step correct")
    data["decomp_component"] = data["decomp"] + data["component"]
    oracle_res = dict(oracle_res[["decomp", "value"]].values)
    ax = sns.relplot(
        data=data,
        x="nsource",
        y="bias_scaled" if scale_bias else "bias",
        hue="est_estimate",
        col="decomp_component",
        row="vars",
        kind="line",
        markers=True,
        linewidth=3,
        facet_kws={'sharey': False}
    )
    ax.set(xscale="log")
    # Horizontal line at bias 0
    for axis in ax.axes.flatten():
        axis.axhline(0, ls="--", color="black", alpha=0.8)
    # ax.fig.suptitle(
    #     "Decomposition: base %0.3f marg %0.3f cond %0.3f" % (oracle_res["base"],oracle_res["marg"],oracle_res["cond"])
    # )
    ax.fig.subplots_adjust(top=0.8)
    # update to pretty axes and titles
    ax.set_axis_labels(
        "n",
        "sqrt(n) bias" if scale_bias else "bias"
    )._legend.set_title("Estimator")
    ax._legend.set_bbox_to_anchor([1.04,0.5])
    ax._legend._loc = 3
    ax.fig.tight_layout()
    plt.savefig(plotfile, bbox_inches="tight")

def plot_var_vs_n(data, oracle_res, plotfile):
    plt.clf()
    plt.figure(figsize=(10,5))
    sns.set_context('paper', font_scale=2)
    data["est_estimate"] = data["est_estimate"].str.replace("eif", "one-step correct")
    data["decomp_component"] = data["decomp"] + data["component"]
    ax = sns.relplot(
        data=data,
        x="nsource",
        y="variance_scaled_estimate",
        hue="est_estimate",
        col="decomp_component",
        row="vars",
        kind="line",
        markers=True,
        linewidth=3,
    )
    ax.set(xscale="log")
    for decomp, axis in ax.axes_dict.items():
        print(oracle_res[oracle_res["decomp"]==decomp])
        # TODO: query correct variance value for different vars
        axis.axhline(
            oracle_res[oracle_res["decomp"]==decomp]["variance"].unique(),
            ls="--", color="black", alpha=0.8
        )
    # update to pretty axes and titles
    ax.set_axis_labels(
        "n",
        "sqrt(n) * variance"
    )._legend.set_title("Estimator")
    plt.savefig(plotfile, bbox_inches="tight")

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    feature_names_df = pd.read_csv(args.feature_names_file)


    all_res_oracle = concat_files_to_df(args.result_files_oracle, None)
    all_res_oracle = all_res_oracle[all_res_oracle.est == "onestep"]
    if args.plot_components:
        print(args.plot_components)
        all_res_oracle = all_res_oracle[all_res_oracle.component.isin(args.plot_components)]
    print("all_res_oracle", all_res_oracle)
    all_res_oracle = all_res_oracle.reset_index(drop=True)
    all_res_oracle.to_csv(args.csv_file_oracle, index=False, na_rep=np.nan)
    
    all_res_estimate = concat_files_to_df(args.result_files_estimate, args.num_jobs)
    all_res_estimate = all_res_estimate.reset_index(drop=True)
    all_res_estimate.to_csv(args.csv_file_estimate, index=False, na_rep=np.nan)
    
    print("all_res_estimate", all_res_estimate[['level','decomp','vars', 'component']])
    print("ORACLE", all_res_oracle[['level','decomp','vars', 'component']])

    # Get std deviation of oracle across oracle runs
    all_res_estimate = all_res_oracle.merge(all_res_estimate, on=['level','decomp','vars', 'component'], suffixes=('_oracle', '_estimate'))
    all_res_estimate = all_res_estimate.merge(feature_names_df, on="vars")
    print(all_res_estimate)
    # Get bias (scaled) of estimates
    all_res_estimate['bias'] = (all_res_estimate['value_estimate'] - all_res_estimate['value_oracle'])
    all_res_estimate['bias_scaled'] = all_res_estimate['bias'] * np.sqrt(all_res_estimate['nsource'] * 0.5)
    # Get variance of estimates
    all_res_estimate['variance_estimate'] = all_res_estimate.groupby(
        ['nsource','ntarget','level','decomp','est_estimate','vars'], dropna=False
    )['value_estimate'].transform('std', ddof=1)**2
    all_res_estimate['variance_scaled_estimate'] = all_res_estimate['variance_estimate'] * np.sqrt(all_res_estimate['nsource'] * 0.5)
    print(all_res_estimate.columns)
    # Get coverage of CI
    all_res_estimate['covered'] = np.logical_and(
        all_res_estimate['value_oracle'] >= all_res_estimate['ci_lower_estimate'],
        all_res_estimate['value_oracle'] <= all_res_estimate['ci_upper_estimate'],
    )
    print("all_res_estimate.est_estimate", all_res_estimate.est_estimate.unique())

    all_res_estimate.to_csv(args.csv_file_estimate, index=False, na_rep=np.nan)
    
    summary_df = all_res_estimate[['level','decomp','vars', 'component', 'bias', 'bias_scaled','value_estimate','value_oracle','est_estimate','covered', 'nsource']].groupby(['level','decomp','vars','component', 'est_estimate', 'nsource']).mean()
    summary_df.to_csv(args.summary_csv_file)

    oracle_decomp = all_res_oracle.groupby(['level','decomp','vars', 'component'], dropna=False)['value'].mean().reset_index()
    
    if (args.biasplot_file_agg is not None) and ('agg' in args.plot_components):
        plot_res = all_res_estimate[
            (all_res_estimate["level"] == "agg")
        ]
        oracle_res = oracle_decomp[
            (oracle_decomp["level"] == "agg")
        ]
        print(plot_res)
        if len(plot_res)!=0:
            plot_bias_vs_n(plot_res, oracle_res, args.biasplot_file_agg, args.scale_bias)

    if args.biasplot_file_detail is not None:
        plot_res = all_res_estimate[
            (all_res_estimate["level"] == "detail")
        ]
        oracle_res = oracle_decomp[
            (oracle_decomp["level"] == "detail")
        ]
        print(plot_res)
        if len(plot_res)!=0:
            plot_bias_vs_n(plot_res, oracle_res, args.biasplot_file_detail, args.scale_bias)
    
    if (args.coverage_detail_file is not None) and len(all_res_estimate) and ('explained_ratio' in args.plot_components):
        print("PLOT COVERAGE")
        plot_coverage_vs_n(all_res_estimate[all_res_estimate.level == "detail"], oracle_decomp[oracle_decomp.level == "detail"], args.ci_level, args.coverage_detail_file)
    if (args.coverage_agg_file is not None) and len(all_res_estimate) and ('agg' in args.plot_components):
        print("PLOT COVERAGE")
        plot_coverage_vs_n(all_res_estimate[all_res_estimate.level == "agg"], oracle_decomp[oracle_decomp.level == "agg"], args.ci_level, args.coverage_agg_file)
    
    # if args.varplot_file_agg is not None:
    #     plot_res = all_res_estimate[
    #         (all_res_estimate["level"] == "agg")
    #     ]
    #     oracle_res = variance_oracle[
    #         (variance_oracle["level"] == "agg")
    #     ]
    #     plot_var_vs_n(plot_res, oracle_res, args.varplot_file_agg)
    
    # if args.varplot_file_detail is not None:
    #     plot_res = all_res_estimate[
    #         (all_res_estimate["level"] == "detail")
    #     ]
    #     oracle_res = variance_oracle[
    #         (variance_oracle["level"] == "detail")
    #     ]
    #     if len(plot_res)!=0:
    #         plot_var_vs_n(plot_res, oracle_res, args.varplot_file_detail)

if __name__ == "__main__":
    main()
