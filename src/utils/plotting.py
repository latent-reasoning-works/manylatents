import matplotlib.pyplot as plt
import scprep


# Generate PHATE plots
def plot_phate_results(phate_embs_list, 
                       metadata, 
                       ts, 
                       param_values, 
                       param_name, 
                       cmap, 
                       output_dir):
    num_vals = len(phate_embs_list)
    num_t = len(phate_embs_list[0])

    # Ensure axes are always treated as 2D arrays
    fig, ax = plt.subplots(
        figsize=(10 * num_t, 10 * num_vals),
        nrows=num_vals,
        ncols=num_t,
        gridspec_kw={'wspace': 0.08},
        squeeze=False  # Ensure ax is always 2D
    )

    for i, embs in enumerate(phate_embs_list):
        for j, emb in enumerate(embs):
            scprep.plot.scatter2d(
                emb,
                s=5,
                ax=ax[i, j],
                c=metadata['Population'].values,
                cmap=cmap,
                xticks=False,
                yticks=False,
                legend=False,
                legend_loc='lower center',
                legend_anchor=(0.5, -0.35),
                legend_ncol=8,
                label_prefix="PHATE ",
                fontsize=8
            )
            ax[i, j].set_title('t={} {}={}'.format(ts[j], param_name, param_values[i]), fontsize=30)
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()
