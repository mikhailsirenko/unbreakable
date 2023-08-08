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
# import contextily as ctx


def rainclouds(outcomes: pd.DataFrame, savefigs: bool,  x_columns: list = [], x_titles: list = [], plot_years_in_poverty: bool = False, color_palette: str = 'Set2', sharex: bool = True):
    districts = outcomes['district'].unique().tolist()
    n_districts = len(districts)
    colors = sns.color_palette(color_palette, n_colors=len(districts))

    if len(x_columns) == 0:
        x_columns = [
            'n_affected_people',
            'n_new_poor_increase_pct',
            'n_new_poor',
            'annual_average_consumption_loss_pct',
            'r',
            'new_poverty_gap',
            # 'one_year_in_poverty',
            # 'two_years_in_poverty',
            # 'three_years_in_poverty',
            # 'four_years_in_poverty',
            # 'five_years_in_poverty',
            # 'six_years_in_poverty',
            # 'seven_years_in_poverty',
            # 'eight_years_in_poverty',
            # 'nine_years_in_poverty',
            # 'ten_years_in_poverty'
        ]

    if len(x_titles) == 0:
        x_titles = [
            'Affected People',
            'New Poor Increase (%)',
            'New Poor',
            'Wt. Ann. Avg. Consump. Loss p.c. (%)',
            'Socio-Economic Resilience',
            'Poverty Gap',
            # 'One year in poverty',
            # 'Two years in poverty',
            # 'Three years in poverty',
            # 'Four years in poverty',
            # 'Five years in poverty',
            # 'Six years in poverty',
            # 'Seven years in poverty',
            # 'Eight years in poverty',
            # 'Nine years in poverty',
            # 'Ten years in poverty'
        ]

    is_years_in_poverty = False

    for x_column, x_title in zip(x_columns, x_titles):
        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(
            4 * n_districts / 3, 3 * n_districts / 3), sharex=sharex)

        for district in districts:
            df = outcomes[outcomes['district'] == district].copy()

            # Calculate an increase in new poor in respect to the total population
            # df = df.assign(one_year_in_poverty = df['years_in_poverty'].apply(lambda x: x[0]))
            # df = df.assign(two_years_in_poverty = df['years_in_poverty'].apply(lambda x: x[1]))
            # df = df.assign(three_years_in_poverty = df['years_in_poverty'].apply(lambda x: x[2]))
            # df = df.assign(four_years_in_poverty = df['years_in_poverty'].apply(lambda x: x[3]))
            # df = df.assign(five_years_in_poverty = df['years_in_poverty'].apply(lambda x: x[4]))
            # df = df.assign(six_years_in_poverty = df['years_in_poverty'].apply(lambda x: x[5]))
            # df = df.assign(seven_years_in_poverty = df['years_in_poverty'].apply(lambda x: x[6]))
            # df = df.assign(eight_years_in_poverty = df['years_in_poverty'].apply(lambda x: x[7]))
            # df = df.assign(nine_years_in_poverty = df['years_in_poverty'].apply(lambda x: x[8]))
            # df = df.assign(ten_years_in_poverty = df['years_in_poverty'].apply(lambda x: x[9]))

            df[x_column] = df[x_column].astype(float)

            # Make a half violin plot
            pt.half_violinplot(x=x_column,
                               y='policy',  # hue='scenario',
                               data=df,
                               color=colors[districts.index(district)],
                               bw=.2,
                               cut=0.,
                               scale="area",
                               width=.6,
                               inner=None,
                               ax=ax[districts.index(district) // 3, districts.index(district) % 3])

            # Add stripplot
            sns.stripplot(x=x_column,
                          y='policy',  # hue='scenario',
                          data=df,
                          color=colors[districts.index(district)],
                          edgecolor='white',
                          size=3,
                          jitter=1,
                          zorder=0,
                          orient='h',
                          ax=ax[districts.index(district) // 3, districts.index(district) % 3])

            # Add boxplot
            sns.boxplot(x=x_column,
                        y='policy',  # hue='scenario',
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
                        ax=ax[districts.index(district) // 3, districts.index(district) % 3])

            if is_years_in_poverty:
                title = district + ', E = ' + \
                    f'{round(df[x_column].mean())}'
            else:
                title = district
            ax[districts.index(district) // 3,
               districts.index(district) % 3].set_title(title)
            ax[districts.index(district) // 3,
               districts.index(district) % 3].set_ylabel('')
            ax[districts.index(district) // 3,
               districts.index(district) % 3].set_xlabel(x_title)

            # Remove y ticks and labels
            ax[districts.index(district) // 3,
               districts.index(district) % 3].set_yticklabels([])
            ax[districts.index(district) // 3,
               districts.index(district) % 3].set_yticks([])

            # Do not display floats in the x-axis
            ax[districts.index(district) // 3, districts.index(district) %
               3].xaxis.set_major_locator(MaxNLocator(integer=True))

            # Plot the median
            # ax[districts.index(district) // 3, districts.index(district) % 3].axvline(df[x_column].median(), color='black', linestyle='--', linewidth=1)

            # Add text close to the boxplot's median
            ax[districts.index(district) // 3, districts.index(district) % 3].text(df[x_column].median(), 0.2,
                                                                                   f'M={df[x_column].median():.2f}',
                                                                                   horizontalalignment='left', size='small', color='black')

            # # Add text close to the boxplot's min and max
            # ax[districts.index(district) // 3, districts.index(district) % 3].text(df[x_column].min(), 0.3,
            #                                                                        f'min={df[x_column].min():.2f}',
            #                                                                        horizontalalignment='left', size='small', color='black')
            # ax[districts.index(district) // 3, districts.index(district) % 3].text(df[x_column].max(), 0.4,
            #                                                                        f'max={df[x_column].max():.2f}',
            #                                                                        horizontalalignment='left', size='small', color='black')

            initial_poverty_gap = df['initial_poverty_gap'].iloc[0]

            # Add initial poverty gap as in the legend to the plot
            if x_column == 'new_poverty_gap':
                ax[districts.index(district) // 3, districts.index(district) % 3].text(0.025, 0.9,
                                                                                       f'Poverty gap before disaster={initial_poverty_gap:.2f}',
                                                                                       horizontalalignment='left', size='small', color='black',
                                                                                       transform=ax[districts.index(district) // 3, districts.index(district) % 3].transAxes)

        # Add a super title
        # fig.suptitle(x_title, fontsize=16)
        fig.tight_layout()
        if savefigs:
            plt.savefig(
                f'../figures/analysis/{x_column}.png', dpi=500, bbox_inches='tight')


def bivariate_choropleth(data, x_name, y_name, x_label, y_label, scheme, figsize, return_table):
    fig, ax = plt.subplots(figsize=figsize)

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


def nine_quadrants_plot(data, x_name, y_name, scale=True):
    _, ax = plt.subplots(figsize=(6, 5))

    if scale:
        scaler = MinMaxScaler()
        # data[x_name] = scaler.fit_transform(data[x_name].values.reshape(-1, 1))
        # data[y_name] = scaler.fit_transform(data[y_name].values.reshape(-1, 1))
        # Scale data between 0 and 1
        data[x_name] = (data[x_name] - data[x_name].min()) / \
            (data[x_name].max() - data[x_name].min())
        data[y_name] = (data[y_name] - data[y_name].min()) / \
            (data[y_name].max() - data[y_name].min())

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


def get_colors(data):

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


def bin_data(data, x_name, y_name, scheme:'fisher_jenks', print_statistics=True):
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
    d = {'0' : '1', '1': '2', '2': '3'}
    data['Var1_Class'] = data['Var1_Class'].replace(d)

    # Code created y bins to A, B, C
    d = {'0' : 'A', '1': 'B', '2': 'C'}
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
