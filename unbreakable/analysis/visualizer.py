import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ptitprince as pt
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.ticker import MaxNLocator
import mapclassify as mc
from mycolorpy import colorlist as mcp
# import contextily as ctx


def raincloud_plot(outcomes: pd.DataFrame, savefig: bool, color_palette: str = 'Set2', sharex: bool = True):
    '''Visualize the outcomes using a raincloud plot.

    Args:
        outcomes (pd.DataFrame): The outcomes dataframe.
        savefig (bool): Whether to save the figure or not.
        color_palette (str, optional): The color palette to use. Defaults to 'Set2'.
        sharex (bool, optional): Whether to share the x-axis or not. Defaults to True.

    Returns:
        None
    '''

    regions = outcomes['region'].unique().tolist()
    n_regions = len(regions)
    colors = sns.color_palette(color_palette, n_colors=len(regions))

    x_columns = [
        'n_aff_people',
        'n_new_poor_increase_pp',
        'n_new_poor',
        'annual_avg_consum_loss_pct',
        'r',
        'new_poverty_gap_initial',
        'new_poverty_gap_all',
    ]

    x_titles = [
        'Affected People',
        'New Poor Increase (p.p.)',
        'New Poor',
        f'Wt. Ann. Avg. Consump. Loss p.c. (%)',
        'Socio-Economic Resilience',
        'New Poverty Gap Initial Poor',
        'New Poverty Gap All Poor']

    for x_column, x_title in zip(x_columns, x_titles):
        fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(
            4 * n_regions / 3, 3 * n_regions / 3), sharex=sharex)

        for region in regions:
            # Select the region
            df = outcomes[outcomes['region'] == region].copy()

            # Convert to float
            df[x_column] = df[x_column].astype(float)

            # Make a half violin plot
            pt.half_violinplot(x=x_column,
                               y='policy',
                               data=df,
                               color=colors[regions.index(region)],
                               bw=.2,
                               cut=0.,
                               scale="area",
                               width=.6,
                               inner=None,
                               ax=ax[regions.index(region) // 3, regions.index(region) % 3])

            # Add stripplot
            sns.stripplot(x=x_column,
                          y='policy',
                          data=df,
                          color=colors[regions.index(region)],
                          edgecolor='white',
                          size=3,
                          jitter=1,
                          zorder=0,
                          orient='h',
                          ax=ax[regions.index(region) // 3, regions.index(region) % 3])

            # Add boxplot
            sns.boxplot(x=x_column,
                        y='policy',
                        data=df,
                        color="black",
                        width=.15,
                        zorder=10,
                        showcaps=True,
                        boxprops={'facecolor': 'none', "zorder": 10},
                        showfliers=True,
                        whiskerprops={'linewidth': 2, "zorder": 10},
                        saturation=1,
                        orient='h',
                        ax=ax[regions.index(region) // 3, regions.index(region) % 3])

            # Set title
            title = region
            ax[regions.index(region) // 3,
               regions.index(region) % 3].set_title(title)
            ax[regions.index(region) // 3,
               regions.index(region) % 3].set_ylabel('')
            ax[regions.index(region) // 3,
               regions.index(region) % 3].set_xlabel(x_title)

            # Remove y ticks and labels
            ax[regions.index(region) // 3,
               regions.index(region) % 3].set_yticklabels([])
            ax[regions.index(region) // 3,
               regions.index(region) % 3].set_yticks([])

            # Do not display floats in the x-axis
            ax[regions.index(region) // 3, regions.index(region) %
               3].xaxis.set_major_locator(MaxNLocator(integer=True))

            # Add text close to the boxplot's median
            ax[regions.index(region) // 3, regions.index(region) % 3].text(df[x_column].median(), 0.2,
                                                                           f'M={df[x_column].median():.2f}',
                                                                           horizontalalignment='left', size='small', color='black')
        # Remove 2 last subplots
        ax[3, 1].set_visible(False)
        ax[3, 2].set_visible(False)
        fig.tight_layout()
        if savefig:
            plt.savefig(
                f'../reports/figures/analysis/{x_column}.png', dpi=500, bbox_inches='tight')
            plt.savefig(
                f'../reports/figures/analysis/{x_column}.pgf', bbox_inches='tight')


def bivariate_choropleth(data: gpd.GeoDataFrame, x_name: str, y_name: str, x_label: str, y_label: str, scheme: str, figsize: tuple, return_table: bool) -> None:
    '''Create a bivariate choropleth map.

    Args:
        data (gpd.GeoDataFrame): Outcomes data frame.
        x_name (str): The name of the first variable.
        y_name (str): The name of the second variable.
        x_label (str): The label of the first variable.
        y_label (str): The label of the second variable.
        scheme (str): The scheme to use for binning the data.
        figsize (tuple): The size of the figure.
        return_table (bool): Whether to return the data frame or not.

    Returns:
        None
    '''

    fig, ax = plt.subplots(figsize=figsize)

    # TODO: Allow for 5 classes
    # Bin the data
    data = bin_data(data, x_name, y_name, scheme, print_statistics=False)

    # Get colors
    all_colors, available_colors = get_colors(data)
    cmap = matplotlib.colors.ListedColormap(available_colors)

    # Step 1: Draw the map
    # border = gpd.read_file(f'../data/processed/boundaries/{city}/city.json')
    border = gpd.read_file(
        '../data/raw/shapefiles/Saint Lucia/gadm36_LCA_shp/gadm36_LCA_0.shp')
    data.plot(ax=ax,
              edgecolor='black',
              linewidth=.1,
              column='Bi_Class',  # variable that is going to be used to color the map
              cmap=cmap,  # newly defined bivariate cmap
              categorical=True,  # bivariate choropleth has to be colored as categorical map
              legend=False)  # we're going to draw the legend ourselves
    # add the basemap
    # ctx.add_basemap(ax=ax, source=ctx.providers.CartoDB.Positron)
    border.plot(ax=ax, facecolor='none',
                edgecolor='black', alpha=.5)  # city border
    for idx, row in data.iterrows():
        ax.annotate(text=row['NAME_1'], xy=row['geometry'].centroid.coords[0],
                    ha='center', fontsize=8, color='white')

    plt.tight_layout()  # "tighten" two figures map and basemap
    plt.axis('off')  # we don't need axis with coordinates
    # ax.set_title('Bivariate Choropleth Amsterdam')

    # Step 2: draw the legend

    # We're drawing a 3x3 "box" as 3 columns
    # The xmin and xmax arguments axvspan are defined to create equally sized small boxes

    img2 = fig  # refer to the main figure
    # add new axes to place the legend there
    ax2 = fig.add_axes([0.15, 0.25, 0.1, 0.1])
    # and specify its location
    alpha = 1  # alpha argument to make it more/less transparent

    # Column 1
    # All colors to create a complete legend
    # all_colors = ['#e8e8e8', '#b0d5df', '#64acbe', '#e4acac', '#ad9ea5', '#627f8c', '#c85a5a', '#985356', '#574249']

    ax2.axvspan(xmin=0, xmax=0.33, ymin=0, ymax=0.33,
                alpha=alpha, color=all_colors[0])
    ax2.axvspan(xmin=0, xmax=0.33, ymin=0.33, ymax=0.66,
                alpha=alpha, color=all_colors[1])
    ax2.axvspan(xmin=0, xmax=0.33, ymin=0.66, ymax=1,
                alpha=alpha, color=all_colors[2])

    # Column 2
    ax2.axvspan(xmin=0.33, xmax=0.66, ymin=0, ymax=0.33,
                alpha=alpha, color=all_colors[3])
    ax2.axvspan(xmin=0.33, xmax=0.66, ymin=0.33, ymax=0.66,
                alpha=alpha, color=all_colors[4])
    ax2.axvspan(xmin=0.33, xmax=0.66, ymin=0.66, ymax=1,
                alpha=alpha, color=all_colors[5])

    # Column 3
    ax2.axvspan(xmin=0.66, xmax=1, ymin=0, ymax=0.33,
                alpha=alpha, color=all_colors[6])
    ax2.axvspan(xmin=0.66, xmax=1, ymin=0.33, ymax=0.66,
                alpha=alpha, color=all_colors[7])
    ax2.axvspan(xmin=0.66, xmax=1, ymin=0.66, ymax=1,
                alpha=alpha, color=all_colors[8])

    # Step 3: annoate the legend
    # remove ticks from the big box
    ax2.tick_params(axis='both', which='both', length=0)
    ax2.axis('off')  # turn off its axis
    ax2.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(
        arrowstyle="->", lw=1, color='black'))  # draw arrow for x
    ax2.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(
        arrowstyle="->", lw=1, color='black'))  # draw arrow for y
    ax2.text(s=x_label, x=0.1, y=-0.25, fontsize=8)  # annotate x axis
    ax2.text(s=y_label, x=-0.25, y=0.1, rotation=90,
             fontsize=8)  # annotate y axis
    # plt.savefig('bivariate_choropleth.png', dpi=300)

    if return_table:
        return data


def nine_quadrants_plot(data: pd.DataFrame, x_name: str, y_name: str, scale: bool = True) -> None:
    '''Create a nine quadrants plot.

    Args:
        data (pd.DataFrame): Outcomes data frame.
        x_name (str): The name of the first variable.
        y_name (str): The name of the second variable.
        scale (bool, optional): Whether to scale the data or not. Defaults to True.

    Returns: 
        None
    '''
    _, ax = plt.subplots(figsize=(6, 5))

    if scale:
        scaler = MinMaxScaler()
        data[x_name] = scaler.fit_transform(data[x_name].values.reshape(-1, 1))
        data[y_name] = scaler.fit_transform(data[y_name].values.reshape(-1, 1))

    data.plot.scatter(x_name, y_name, s=20, ax=ax, c='black', zorder=2)

    # Iterate over each row and annotate the points
    for idx, row in data.iterrows():
        ax.annotate(text=row['NAME_1'], xy=(row[x_name], row[y_name]),
                    ha='center', fontsize=10, color='black')

    # Annotate with Spearman's rho
    # rho, p = spearmanr(data[x_name], data[y_name])
    # ax.text(0.05, 0.95, f'$\\rho$ = {round(rho, 2)}', transform=ax.transAxes,
    #         verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', alpha=1))

    ax.axvline(0.33, color='black', alpha=.33, lw=1)
    ax.axvline(0.66, color='black', alpha=.33, lw=1)
    ax.axhline(0.33, color='black', alpha=.33, lw=1)
    ax.axhline(0.66, color='black', alpha=.33, lw=1)

    alpha = 1

    all_colors = {'1A': '#dddddd',
                  '1B': '#dd7c8a',
                  '1C': '#cc0024',
                  '2A': '#7bb3d1',
                  '2B': '#8d6c8f',
                  '2C': '#8a274a',
                  '3A': '#016eae',
                  '3B': '#4a4779',
                  '3C': '#4b264d'}

    # Column 1
    c = all_colors['1A']
    ax.axvspan(xmin=0, xmax=0.33, ymin=0 + 0.025,
               ymax=0.345, alpha=alpha, color=c)

    c = all_colors['1B']
    ax.axvspan(xmin=0, xmax=0.33, ymin=0.33 + 0.015,
               ymax=0.66 - 0.015, alpha=alpha,  color=c)

    c = all_colors['1C']
    ax.axvspan(xmin=0, xmax=0.33, ymin=0.66 - 0.015,
               ymax=1 - 0.05, alpha=alpha, color=c)

    # Column 2
    c = all_colors['2A']
    ax.axvspan(xmin=0.33, xmax=0.66, ymin=0 + 0.025,
               ymax=0.345, alpha=alpha,  color=c)

    c = all_colors['2B']
    ax.axvspan(xmin=0.33, xmax=0.66, ymin=0.345,
               ymax=0.645, alpha=alpha,  color=c)

    c = all_colors['2C']
    ax.axvspan(xmin=0.33, xmax=0.66, ymin=0.649,
               ymax=1 - 0.05, alpha=alpha, color=c)

    # Column 3
    c = all_colors['3A']
    ax.axvspan(xmin=0.66, xmax=1, ymin=0.025, ymax=0.345, alpha=alpha, color=c)

    c = all_colors['3B']
    ax.axvspan(xmin=0.66, xmax=1, ymin=0.345,
               ymax=0.645, alpha=alpha,  color=c)

    c = all_colors['3C']
    ax.axvspan(xmin=0.66, xmax=1, ymin=0.649,
               ymax=1 - 0.05, alpha=alpha, color=c)

    ax.set_xlim(-.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add regression line
    # x = data[x_name]
    # y = data[y_name]
    # m, b = np.polyfit(x, y, 1)
    # ax.plot(x, m * x + b, color='black', alpha=0.5, zorder=1)


def get_colors(data: pd.DataFrame) -> list:
    '''Get colors for the bivariate choropleth map.'''

    # colors = ['#e8e8e8', # 1A
    #           '#b0d5df', # 1B
    #           # '#64acbe', # 1C
    #           '#e4acac', # 2A
    #           '#ad9ea5', # 2B
    #           '#627f8c', # 2C
    #           '#c85a5a', # 3A
    #           '#985356'] # , # 3B
    #           # '#574249'] # 3C

    all_colors = {'1A': '#e8e8e8',
                  '1B': '#b0d5df',
                  '1C': '#64acbe',
                  '2A': '#e4acac',
                  '2B': '#ad9ea5',
                  '2C': '#627f8c',
                  '3A': '#c85a5a',
                  '3B': '#985356',
                  '3C': '#574249'}

    all_colors = {'1A': '#dddddd',
                  '1B': '#dd7c8a',
                  '1C': '#cc0024',
                  '2A': '#7bb3d1',
                  '2B': '#8d6c8f',
                  '2C': '#8a274a',
                  '3A': '#016eae',
                  '3B': '#4a4779',
                  '3C': '#4b264d'}

    # Set of colors matching the elements of Bi_Class
    # We have to exclude those that did not come up in the data
    available_classes = data['Bi_Class'].value_counts().sort_index().index
    available_colors = [all_colors[i] for i in available_classes]
    return list(all_colors.values()), available_colors


def bin_data(data: gpd.GeoDataFrame, x_name: str, y_name: str, scheme: str = 'fisher_jenks', print_statistics: bool = True) -> gpd.GeoDataFrame:
    '''Bin the data for the bivariate choropleth map.

    Args:
        data (gpd.GeoDataFrame): Outcomes data frame.
        x_name (str): The name of the first variable.
        y_name (str): The name of the second variable.
        scheme (str): The scheme to use for binning the data.
        print_statistics (bool, optional): Whether to print statistics or not. Defaults to True.

    Returns:
        data (gpd.GeoDataFrame): The outcomes data frame with the binned data.
    '''
    if scheme == 'fisher_jenks':
        x_classifier = mc.FisherJenks(data[x_name], k=3)
        # x_bin_edges = x_classifier.bins
        x_bin_labels = x_classifier.yb

        y_classifier = mc.FisherJenks(data[y_name], k=3)
        # y_bin_edges = y_classifier.bins
        y_bin_labels = y_classifier.yb

    # Bin the first variable - x
    data['Var1_Class'] = x_bin_labels
    data['Var1_Class'] = data['Var1_Class'].astype('str')

    # Bin the second variable - y
    data['Var2_Class'] = y_bin_labels
    data['Var2_Class'] = data['Var2_Class'].astype('str')

    # Code created x bins to 1, 2, 3
    d = {'0': '1', '1': '2', '2': '3'}
    data['Var1_Class'] = data['Var1_Class'].replace(d)

    # Code created y bins to A, B, C
    d = {'0': 'A', '1': 'B', '2': 'C'}
    data['Var2_Class'] = data['Var2_Class'].replace(d)

    # Combine x and y codes to create Bi_Class
    data['Bi_Class'] = data['Var1_Class'].astype('str') + data['Var2_Class']

    if print_statistics:
        print('Number of unique elements in Var1_Class =',
              len(data['Var1_Class'].unique()))
        print('Number of unique elements in Var2_Class =',
              len(data['Var2_Class'].unique()))
        print('Number of unique elements in Bi_Class =',
              len(data['Bi_Class'].unique()))
    return data


def annotated_hist(outcomes: pd.DataFrame, annotate: bool) -> None:
    '''Create an annotated histogram of the annual average consumption loss.

    Args:
        outcomes (pd.DataFrame): Outcomes data frame.
        annotate (bool): Whether to annotate the plot or not.

    Returns:
        None
    '''
    sns.histplot(outcomes['annual_avg_consum_loss_pct'],)
    plt.xlabel('Annual Average Consumption Loss PC (%)')
    plt.ylabel('Run count')

    plt.axvline(outcomes['annual_avg_consum_loss_pct'].min(
    ), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(outcomes['annual_avg_consum_loss_pct'].max(
    ), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(outcomes['annual_avg_consum_loss_pct'].median(
    ), color='black', linestyle='dashed', linewidth=1)

    if annotate:
        plt.annotate(f"{outcomes['annual_avg_consum_loss_pct'].min():.2f}%",
                     xy=(outcomes['annual_avg_consum_loss_pct'].min(), 0),
                     xytext=(
                         outcomes['annual_avg_consum_loss_pct'].min() - 5, 100),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='top')
        plt.annotate(f"{outcomes['annual_avg_consum_loss_pct'].max():.2f}%",
                     xy=(outcomes['annual_avg_consum_loss_pct'].max(), 0),
                     xytext=(
                         outcomes['annual_avg_consum_loss_pct'].max() + 5, 100),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='left', verticalalignment='top')
        plt.annotate(f"{outcomes['annual_avg_consum_loss_pct'].median():.2f}%",
                     xy=(outcomes['annual_avg_consum_loss_pct'].median(), 0),
                     xytext=(
                         outcomes['annual_avg_consum_loss_pct'].median() + 5, 100),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='left', verticalalignment='top')
    sns.despine()
    plt.tight_layout()


def coloured_density_plots(outcomes: pd.DataFrame, scheme: str, k: int, cmap: str = "OrRd", legend: bool = True) -> None:
    '''Make colored density plots for each region. Color here matches the color of the choropleth map.

    Args:
        outcomes (pd.DataFrame): Outcomes data frame.
        scheme (str, optional): The scheme to use for binning the data.
        k (int, optional): The number of bins.
        cmap (str, optional): The name of the colormap to use. Defaults to "OrRd".

    Returns:
        None
    '''
    # Choropleth map uses median values to classify the regions, we're going to do the same
    median_outcomes = outcomes.groupby('region').median(numeric_only=True)[
        ['annual_avg_consum_loss_pct']]

    # The map used equalinterval scheme, but it would be beneficial to allow for other schemes
    if scheme == 'equal_intervals':
        classifier = mc.EqualInterval(median_outcomes.values, k=k)
    elif scheme == 'fisher_jenks':
        classifier = mc.FisherJenks(median_outcomes.values, k=k)
    else:
        raise ValueError(
            'Invalid scheme. Please use `equal_intervals` or `fisher_jenks`')

    # Get the bin edges and labels
    bin_edges = classifier.bins
    bin_labels = classifier.yb

    # Map the region to the bin label
    region_to_label_mapper = dict(zip(median_outcomes.index, bin_labels))

    # Map the bin label to a color
    colors = mcp.gen_color(cmap=cmap, n=k)
    label_color_mapper = dict(zip(np.arange(0, k), colors))

    regions = outcomes['region'].unique().tolist()
    fig, ax = plt.subplots()

    # Get regions with min and max values
    descr = outcomes.iloc[:, 2:-1].groupby('region').describe()[
        ['annual_avg_consum_loss_pct']]
    min_distr = descr['annual_avg_consum_loss_pct']['50%'].idxmin()
    max_distr = descr['annual_avg_consum_loss_pct']['50%'].idxmax()

    # Make the density plots
    for region in regions:
        df = outcomes[outcomes['region'] == region]
        region_label = region_to_label_mapper[region]
        label_color = label_color_mapper[region_label]
        color = label_color

        # Make the line thicker for regions with min and max values
        if region in [min_distr, max_distr]:
            linewidth = 5
        else:
            linewidth = 1
        sns.kdeplot(data=df, x='annual_avg_consum_loss_pct',
                    ax=ax, color=color, linewidth=linewidth, alpha=1)

        ax.set_xlabel('Annual Average Consumption Loss PC (%)')
        ax.set_ylabel('Run density')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_facecolor('lightgray')

    if legend:
        # Move legend outside the plot
        ax.legend(regions, title='Region', frameon=False)
        ax.get_legend().set_bbox_to_anchor((1, 1))
