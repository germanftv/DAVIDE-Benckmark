import os
import pandas as pd
import numpy as np
import seaborn as sns

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# default root folder of the experiments
exp_root = os.path.join(os.getenv('RESULTS_ROOT'), 'E1_depth_impact')

# list of dictionaries with the experiment information
exp_info =[
    {
        'exp_folder': '000_test_shiftnet_baseRGB_in01_out01_seed10',
        'T': 1,
        'data_input': 'RGB',
        'seed': 10
    },
    {
        'exp_folder': '000_test_shiftnet_baseRGB_in01_out01_seed13',
        'T': 1,
        'data_input': 'RGB',
        'seed': 13
    },
    {
        'exp_folder': '000_test_shiftnet_baseRGB_in01_out01_seed17',
        'T': 1,
        'data_input': 'RGB',
        'seed': 17
    },
    {
        'exp_folder': '001_test_shiftnet_RGBD_in01_out01_seed10',
        'T': 1,
        'data_input': 'RGBD',
        'seed': 10
    },
    {
        'exp_folder': '001_test_shiftnet_RGBD_in01_out01_seed13',
        'T': 1,
        'data_input': 'RGBD',
        'seed': 13

    },
    {
        'exp_folder': '001_test_shiftnet_RGBD_in01_out01_seed17',
        'T': 1,
        'data_input': 'RGBD',
        'seed': 17

    },
    {
        'exp_folder': '002_test_shiftnet_baseRGB_in03_out01_seed10',
        'T': 3,
        'data_input': 'RGB',
        'seed': 10

    },
    {
        'exp_folder': '002_test_shiftnet_baseRGB_in03_out01_seed13',
        'T': 3,
        'data_input': 'RGB',
        'seed': 13

    },
    {
        'exp_folder': '002_test_shiftnet_baseRGB_in03_out01_seed17',
        'T': 3,
        'data_input': 'RGB',
        'seed': 17

    },
    {
        'exp_folder': '003_test_shiftnet_RGBD_in03_out01_seed10',
        'T': 3,
        'data_input': 'RGBD',
        'seed': 10

    },
    {
        'exp_folder': '003_test_shiftnet_RGBD_in03_out01_seed13',
        'T': 3,
        'data_input': 'RGBD',
        'seed': 13

    },
    {
        'exp_folder': '003_test_shiftnet_RGBD_in03_out01_seed17',
        'T': 3,
        'data_input': 'RGBD',
        'seed': 17

    },
    {
        'exp_folder': '004_test_shiftnet_baseRGB_in05_out03_seed10',
        'T': 5,
        'data_input': 'RGB',
        'seed': 10

    },
    {
        'exp_folder': '004_test_shiftnet_baseRGB_in05_out03_seed13',
        'T': 5,
        'data_input': 'RGB',
        'seed': 13

    },
    {
        'exp_folder': '004_test_shiftnet_baseRGB_in05_out03_seed17',
        'T': 5,
        'data_input': 'RGB',
        'seed': 17

    },
    {
        'exp_folder': '005_test_shiftnet_RGBD_in05_out03_seed10',
        'T': 5,
        'data_input': 'RGBD',
        'seed': 10

    },
    {
        'exp_folder': '005_test_shiftnet_RGBD_in05_out03_seed13',
        'T': 5,
        'data_input': 'RGBD',
        'seed': 13

    },
    {
        'exp_folder': '005_test_shiftnet_RGBD_in05_out03_seed17',
        'T': 5,
        'data_input': 'RGBD',
        'seed': 17

    },
    {
        'exp_folder': '006_test_shiftnet_baseRGB_in07_out05_seed10',
        'T': 7,
        'data_input': 'RGB',
        'seed': 10

    },
    {
        'exp_folder': '006_test_shiftnet_baseRGB_in07_out05_seed13',
        'T': 7,
        'data_input': 'RGB',
        'seed': 13

    },
    {
        'exp_folder': '006_test_shiftnet_baseRGB_in07_out05_seed17',
        'T': 7,
        'data_input': 'RGB',
        'seed': 17

    },
    {
        'exp_folder': '007_test_shiftnet_RGBD_in07_out05_seed10',
        'T': 7,
        'data_input': 'RGBD',
        'seed': 10

    },
    {
        'exp_folder': '007_test_shiftnet_RGBD_in07_out05_seed13',
        'T': 7,
        'data_input': 'RGBD',
        'seed': 13

    },
    {
        'exp_folder': '007_test_shiftnet_RGBD_in07_out05_seed17',
        'T': 7,
        'data_input': 'RGBD',
        'seed': 17

    },
    {
        'exp_folder': '008_test_shiftnet_baseRGB_in09_out07_seed10',
        'T': 9,
        'data_input': 'RGB',
        'seed': 10

    },
    {
        'exp_folder': '008_test_shiftnet_baseRGB_in09_out07_seed13',
        'T': 9,
        'data_input': 'RGB',
        'seed': 13

    },
    {
        'exp_folder': '008_test_shiftnet_baseRGB_in09_out07_seed17',
        'T': 9,
        'data_input': 'RGB',
        'seed': 17

    },
    {
        'exp_folder': '009_test_shiftnet_RGBD_in09_out07_seed10',
        'T': 9,
        'data_input': 'RGBD',
        'seed': 10

    },
    {
        'exp_folder': '009_test_shiftnet_RGBD_in09_out07_seed13',
        'T': 9,
        'data_input': 'RGBD',
        'seed': 13

    },
    {
        'exp_folder': '009_test_shiftnet_RGBD_in09_out07_seed17',
        'T': 9,
        'data_input': 'RGBD',
        'seed': 17

    },
    {
        'exp_folder': '010_test_shiftnet_baseRGB_in11_out09_seed10',
        'T': 11,
        'data_input': 'RGB',
        'seed': 10

    },
    {
        'exp_folder': '010_test_shiftnet_baseRGB_in11_out09_seed13',
        'T': 11,
        'data_input': 'RGB',
        'seed': 13

    },
    {
        'exp_folder': '010_test_shiftnet_baseRGB_in11_out09_seed17',
        'T': 11,
        'data_input': 'RGB',
        'seed': 17

    },
    {
        'exp_folder': '011_test_shiftnet_RGBD_in11_out09_seed10',
        'T': 11,
        'data_input': 'RGBD',
        'seed': 10

    },
    {
        'exp_folder': '011_test_shiftnet_RGBD_in11_out09_seed13',
        'T': 11,
        'data_input': 'RGBD',
        'seed': 13

    },
    {
        'exp_folder': '011_test_shiftnet_RGBD_in11_out09_seed17',
        'T': 11,
        'data_input': 'RGBD',
        'seed': 17

    }

]

# define the dtypes for the attributes and proximity dataframes
dtypes_env_motion = {
    'clip': str,
    'frame': str,
    'indoors': bool,
    'outdoors': bool,
    'CM': bool,     # camera motion
    'CM+MO': bool,  # camera motion + moving objects
}

dtypes_proximity = {
    'clip': str,
    'frame': str,
    'close': bool,
    'mid': bool,
    'far': bool,
}


def parse_args():
    """Parse the arguments for the script."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--env_motion_csv', type=str, default='../../dataset/annotations/test_env_motion.csv')
    parser.add_argument('--proximity_csv', type=str, default='../../dataset/annotations/test_proximity.csv')
    parser.add_argument('--avg_conf_csv', type=str, default='../../dataset/annotations/test_avg_conf_depth.csv')
    parser.add_argument('--depth_impact_fig', type=str, default='./Figures/fig_E1_depth_impact_attributes.pdf')
    parser.add_argument('--confidence_fig', type=str, default='./Figures/fig_E1_confidence_attributes.pdf')
    return parser.parse_args()


def read_csv_results(exp_root, exp_info):
    """
    Read the csv files in the results folder of the experiments.

    Args:
        exp_root (str): root folder of the experiments
        exp_info (list): list of dictionaries with the experiment information

    Returns:
        results (list): list of dictionaries with the results of the experiments
    """
    results = []
    for exp in exp_info:
        exp_folder = exp['exp_folder']
        T = exp['T']
        data_input = exp['data_input']
        seed = exp['seed']

        exp_path = os.path.join(exp_root, exp_folder, 'results')
        files = os.listdir(exp_path)
        files = [f for f in files if f.endswith('.csv')]
        if len(files) == 0:
            print(f'No files found in {exp_path}')
            continue
        if len(files) > 0:
            # take the most recent file
            files.sort(key=lambda x: os.path.getmtime(os.path.join(exp_path, x)), reverse=True)
            df = pd.read_csv(os.path.join(exp_path, files[0]))
            # rename "clip_name" to "clip"
            df = df.rename(columns={'clip_name': 'clip'})
            # remove ".png" from "frame"
            df['frame'] = df['frame'].str.replace('.png', '')
            # index by clip and frame
            df = df.set_index(['clip', 'frame'])
            results.append({
                'exp_folder': exp_folder,
                'T': T,
                'data_input': data_input,
                'seed': seed,
                'df': df
            })
    return results


def compute_avg_metrics_per_attribute(df, attributes_df):
    """
    Compute the average metrics for each attribute in the attributes dataframe.

    Args:
        df (pd.DataFrame): dataframe with the metrics
        attributes_df (pd.DataFrame): dataframe with the attributes
    """
    avg_metrics = {}
    # get attribute columns
    attribute_columns = [c for c in attributes_df.columns if c not in ['clip', 'frame']]
    for attribute in attribute_columns:
        #get the indices of rows that have the attribute
        indices = attributes_df[attributes_df[attribute] == True].index
        avg_metrics[attribute] = df.loc[indices][['psnr', 'ssim']].mean()
    return avg_metrics


def compute_avg_metrics_all(results, attributes_df):
    """
    Compute the average metrics for each attribute for all the experiments.
    """
    avg_metrics_all = []
    for result in results:
        avg_metrics = compute_avg_metrics_per_attribute(result['df'], attributes_df)
        avg_metrics_all.append({
            'T': result['T'],                       # temporal length of the context window
            'data_input': result['data_input'],
            'seed': result['seed'],
            'avg_metrics': avg_metrics
        })
    return avg_metrics_all


def avg_metrics_to_df(avg_metrics_all, attributes_df):
    """
    Convert the list of average metrics to a pandas dataframe.
    """
    res = []
    # get attribute columns
    attribute_columns = [c for c in attributes_df.columns if c not in ['clip', 'frame']]
    for avg_metrics in avg_metrics_all:
        for attribute in attribute_columns:
            res.append({
                'T': avg_metrics['T'],                      # temporal length of the context window
                'data_input': avg_metrics['data_input'],
                'seed': avg_metrics['seed'],
                'attribute': attribute,
                'psnr': avg_metrics['avg_metrics'][attribute]['psnr'],
                'ssim': avg_metrics['avg_metrics'][attribute]['ssim']
            })
    df = pd.DataFrame(res)
    return df


def process_data(results, attributes_df):
    """
    Process the data to get the average PSNR for each attribute.

    Args:
        results (dict): dictionary with the results of the experiments
        attributes_df (pd.DataFrame): dataframe with the attributes

    Returns:
        df_psnr (pd.DataFrame): dataframe with the average PSNR for each attribute
    """
    # compute the average metrics for each attribute
    avg_metrics_all = compute_avg_metrics_all(results, attributes_df)
    # convert the list of average metrics to a pandas dataframe
    df_avg_metrics = avg_metrics_to_df(avg_metrics_all, attributes_df)

    # compute the average PSNR and SSIM and group by "T", "data_input" and "attribute"
    df_avg_metrics_grouped = df_avg_metrics.groupby(['T', 'data_input', 'attribute']).mean().reset_index()
    # drop seed column
    df_avg_metrics_grouped = df_avg_metrics_grouped.drop(columns=['seed'])

    # compute PSNR and SSIM difference between RGB and RGBD, grouped by "attribute" and "T": (RGBD - RGB)
    df_avg_metrics_grouped['psnr_diff'] = df_avg_metrics_grouped.groupby(['attribute', 'T'])['psnr'].diff()
    df_avg_metrics_grouped['ssim_diff'] = df_avg_metrics_grouped.groupby(['attribute', 'T'])['ssim'].diff()

    # present the PSNR results as a table with "attribute" as columns and "T" as rows
    df_psnr = df_avg_metrics_grouped.pivot_table(index='T', columns=['attribute'], values='psnr_diff')

    # sort attributes as: indoors, outdoors, close, mid, far, CM, CM+MO
    df_psnr = df_psnr[['CM', 'CM+MO', 'close', 'mid', 'far', 'indoors', 'outdoors']]
    # rename attributes with capital letter in the beginning
    df_psnr.columns = [c.capitalize() if c not in ['CM', 'CM+MO'] else c for c in df_psnr.columns]

    # get frames 1,3,5, and 11
    df_psnr = df_psnr.loc[[1,3,5,11]]
    return df_psnr


def radar_chart(df, depth_impact_fig):
    """
    Create a radar chart with the average PSNR for each attribute.

    Args:
        df (pd.DataFrame): dataframe with the average metrics for each attribute
        depth_impact_fig (str): path to save the figure
    """

    # Define a color palette
    colors = px.colors.qualitative.Plotly
    colors = [color.lstrip('#') for color in colors]        # transform the HEX colors to RGBA with 20% transparency
    colors = [f'rgba{tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))+(0.15,)}' for h in colors]
    
    # Create a radar chart
    fig = go.Figure()

    # Add a trace for each row in the dataframe
    for i, (index, row) in enumerate(df.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=row.values,
            theta=df.columns,
            name=index,
            fill='toself',
            fillcolor=colors[i % len(colors)],  # Use the color from the list
        ))

    # Update the layout of the figure
    fig.update_layout(
        autosize=False,
        width=750,
        height=600,
        font=dict(
            family="sans-serif",  # Set the font family to "DejaVu Sans"
            size=28,
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1],
                tickfont=dict(  # Set the font for the radial ticks
                    family="sans-serif",
                    size=25
                )
            ),
            angularaxis=dict(  # Set the font for the angular ticks
                tickfont=dict(
                    family="sans-serif",
                    size=28
                )
            )
        ),
        showlegend=True,
        legend=dict(
            title=dict(
                text="T",
                font=dict(
                    family="sans-serif",  # Set the font for the legend
                    size=28,
                )
            ),
            itemsizing='constant'  # Make the legend items the same size
        )
    )
    pio.write_image(fig, depth_impact_fig)  # Save the figure as a PDF


def confidence_per_attribute(df, confidence_fig):
    """
    Create a bar plot with the average depth confidence for each attribute.

    Args:
        df (pd.DataFrame): dataframe with the average depth confidence for each attribute
        confidence_fig (str): path to save the figure
    """

    import matplotlib.pyplot as plt

    attr_lut = {
        'indoors': 'Indoors',
        'outdoors': 'Outdoors',
        'CM': 'CM',
        'CM+MO': 'CM+MO',
        'close': 'Close',
        'mid': 'Mid',
        'far': 'Far',
    }

    rows = []
    for col in df.columns:
        if col == 'mean_conf':
            continue
        avg_conf = df.groupby(col)['mean_conf'].mean()
        tmp = {
            'Attributes': attr_lut[col],
            'Mean confidence': avg_conf[True],
        }
        rows.append(tmp)

    df_conf_attr = pd.DataFrame(rows)
    print(df_conf_attr)

    sns.set_theme()                     # set the default seaborn theme
    sns.set_context('poster')           # set the default seaborn context 
    plt.rcParams['text.usetex'] = True  # enable LaTeX text rendering
    plt.rcParams['font.size'] = 28      # set global font size

    # wider figure
    plt.figure(figsize=(10, 5))
    # pastel palette (flare)
    palette = sns.color_palette('hls', 7)


    plot = sns.barplot(df_conf_attr, x='Attributes', y='Mean confidence', palette=palette)

    # Iterate over the bars and annotate them with their height values
    for p in plot.patches:
        height = p.get_height()
        plot.annotate(f'${height:.2f}$', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    fontsize=20)

    # update x-axis ticks
    plt.xticks(rotation=45)              # rotate the x-axis ticks

    # update y-axis range
    plt.ylim(0, 90)                     # set y-axis range

    # ticks font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # figure adjustments
    plt.ylabel('Average depth confidence (\%)')   # set y-axis label

    # save the plot to pdf
    plot.get_figure().savefig(confidence_fig, bbox_inches='tight')  # save the figure as a PDF


def main():
    # parse the arguments
    args = parse_args()
    # read the csv files
    results = read_csv_results(args.exp_root, exp_info)

    # load the environment-motion, proximity, and confidence dataframes
    env_motion_df = pd.read_csv(args.env_motion_csv, dtype=dtypes_env_motion)
    proximity_df = pd.read_csv(args.proximity_csv, dtype=dtypes_proximity)
    avg_conf_df = pd.read_csv(args.avg_conf_csv)

    # concatenate columns of proximity_df to env_motion_df except for the first two columns
    proximity_df = proximity_df.drop(columns=['frame', 'clip'])
    attributes_df = pd.concat([env_motion_df, proximity_df], axis=1)

    # index the attributes_df by clip and frame
    attributes_df = attributes_df.set_index(['clip', 'frame'])

    # process the data
    df_psnr = process_data(results, attributes_df)

    # radar chart
    os.makedirs(os.path.dirname(args.depth_impact_fig), exist_ok=True)
    radar_chart(df_psnr, args.depth_impact_fig)
    print('Figure saved to', args.depth_impact_fig)

    # reset the index of attributes_df, make clip and frame columns
    attributes_df = attributes_df.reset_index()

    # concatenate columns of avg_conf_df to attributes_df except for the first two columns
    avg_conf_df = avg_conf_df.drop(columns=['frame', 'clip'])
    attributes_df = pd.concat([attributes_df, avg_conf_df], axis=1)

    # index the attributes_df by clip and frame
    attributes_df = attributes_df.set_index(['clip', 'frame'])

    # confidence per attribute
    os.makedirs(os.path.dirname(args.confidence_fig), exist_ok=True)
    confidence_per_attribute(attributes_df, args.confidence_fig)
    print('Figure saved to', args.confidence_fig)


if __name__ == '__main__':
    main()