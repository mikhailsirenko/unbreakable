import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# import contextily as ctx


def bivariate_choropleth(data, x_name, y_name, x_label, y_label, scale, figsize):
    fig, ax = plt.subplots(figsize=figsize)

    # Bin the data
    data = bin_data(data, x_name, y_name, scale, print_statistics=False)

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


def bin_data(data, x_name, y_name, scale, print_statistics=True):
    if scale:
        # Scale the data to be between 0 and 1
        # data[x_name] = (data[x_name] - data[x_name].min()) / \
        #     (data[x_name].max() - data[x_name].min())
        # data[y_name] = (data[y_name] - data[y_name].min()) / \
        #     (data[y_name].max() - data[y_name].min())
        # Scale the data with MinMaxScaler
        scaler = MinMaxScaler()
        data[x_name] = scaler.fit_transform(data[x_name].values.reshape(-1, 1))
        data[y_name] = scaler.fit_transform(data[y_name].values.reshape(-1, 1))

    # Define the bins
    bins = [0, 0.33, 0.66, 1]

    # Bin the first variable - x
    data['Var1_Class'] = pd.cut(data[x_name], bins=bins, include_lowest=True)
    data['Var1_Class'] = data['Var1_Class'].astype('str')

    # Bin the second variable - y
    data['Var2_Class'] = pd.cut(data[y_name], bins=bins, include_lowest=True)
    data['Var2_Class'] = data['Var2_Class'].astype('str')

    # Code created x bins to 1, 2, 3
    x_class_codes = np.arange(1, len(bins))
    d = dict(
        zip(data['Var1_Class'].value_counts().sort_index().index, x_class_codes))
    data['Var1_Class'] = data['Var1_Class'].replace(d)

    # Code created y bins to A, B, C
    y_class_codes = ['A', 'B', 'C']
    d = dict(
        zip(data['Var2_Class'].value_counts().sort_index().index, y_class_codes))
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
