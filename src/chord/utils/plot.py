import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
from matplotlib.collections import PathCollection

def pretty_boxplot(ax, data, labels, title, pval, *,
                   colors=('tab:blue', 'tab:orange'),
                   ylim=(-0.2, 1),
                   font_family='Arial',
                   base_fontsize=12):

    plt.rcParams.update({
        'font.weight'      : 'normal',
        'axes.titleweight' : 'normal',
        'axes.labelweight' : 'normal', 
        'font.family': font_family,
        'font.size': base_fontsize,
        'axes.titlesize': base_fontsize + 2,
        'axes.labelsize': base_fontsize,
        'xtick.labelsize': base_fontsize - 1,
        'ytick.labelsize': base_fontsize - 1,
    })
    ax.tick_params(axis='both', which='both', width=1.0) 

    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=1.6, color="black"),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2)
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.grid(False)

    ax.set_title(f"{title}", pad=8, fontweight='normal',  fontsize=12)
    ax.text(0.5, 0.95, f"p = {pval:.1e}",
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=10, fontweight='normal')
    ax.set_ylim(*ylim)


def plot_enrichment_barchart(
    df,
    top_n: int = 12,
    p_value_col: str = 'Adjusted P-value',
    term_col: str = 'Term_Name',
    overlap_col: str = 'Overlap',
    go_id_col: str = 'Term',
    title: str = 'Top Enriched GO Terms',
    color: str = "#97BEE9", 
    output_file: str = None
    ):
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    mpl.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="ticks")

    d = df.copy()
    d['-log10(AdjP)'] = -np.log10(d[p_value_col])
    def _get_gene_count(x):
        try:
            return int(str(x).split('/')[0])
        except Exception:
            return 0
    d['Gene_Count'] = d[overlap_col].apply(_get_gene_count)
    
    d_plot = d.sort_values(by='-log10(AdjP)', ascending=False).head(top_n)
    d_plot = d_plot.sort_values(by='-log10(AdjP)', ascending=True).reset_index(drop=True)

    def _truncate(name, maxlen=50):
        s = str(name)
        return (s[:maxlen] + '...') if len(s) > maxlen else s
    if go_id_col in d_plot.columns:
        y_labels = [
            f"{_truncate(name)} ({go_id})"
            for name, go_id in zip(d_plot[term_col], d_plot[go_id_col])
        ]
    else:
        y_labels = [_truncate(name) for name in d_plot[term_col]]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    barplot = sns.barplot(
        x='-log10(AdjP)', y=term_col, data=d_plot, ax=ax,
        color=color, edgecolor='white',
        width=0.9
    )
    
    ax.margins(y=0.01)              
    ax.set_ylim(-0.5, len(d_plot)-0.5)
    ax.tick_params(axis='y', pad=0.5)
    ax.set_yticklabels(y_labels, fontsize=8, linespacing=0.9)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"-log10({p_value_col.replace('_', ' ')})", fontsize=6)
    ax.set_ylabel("")
    
    ax.grid(False)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_yticks(range(len(d_plot)))

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    plt.show()