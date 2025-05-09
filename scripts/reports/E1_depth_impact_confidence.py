import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.text import Text
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def parse_args():
    """Parse the arguments for the script."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--avg_conf_csv', type=str, default='../../dataset/annotations/test_avg_conf_depth.csv')
    parser.add_argument('--psnr_figure', type=str, default='./Figures/fig_E1_depth_impact_confidence.pdf')
    parser.add_argument('--hist_figure', type=str, default='./Figures/fig_E1_confidence_histogram.pdf')
    return parser.parse_args()


def read_csv_results(exp_root, exp_info):
    """
    Read the csv files in the results folder of the experiments.

    Args:
        exp_root (str): root folder of the experiments
        exp_info (list): list of dictionaries with the experiment information
    """
    results = []
    for exp in exp_info:
        exp_folder = exp['exp_folder']
        T = exp['T']                        # temporal length of the context window
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
            results.append({
                'exp_folder': exp_folder,
                'T': T,
                'data_input': data_input,
                'seed': seed,
                'df': df
            })
    return results


def read_avg_confidence(csv_file):
    """
    Read the csv file with the average depth confidence values.

    Args:
        csv_file (str): path to the csv file
    """
    # read avg. confidence data
    avg_conf = pd.read_csv(csv_file)
    # transform the 'clip_name' column to match the one in the results
    avg_conf['frame'] = avg_conf['frame'].apply(lambda x: f'{x:08d}.png')
    # rename the columns
    avg_conf.columns = ['clip_name', 'frame', 'avg_conf']
    return avg_conf


def confiddence_histogram(df, output_file):
    """
    Generate the histogram of the confidence values.

    Args:
        df (pd.DataFrame): dataframe with the confidence values
        output_file (str): path to save the output figure
    """
    # set figure settings
    sns.set_theme()                     # set the default seaborn theme
    sns.set_context('poster')           # set the default seaborn context 
    plt.figure(figsize=(5.5, 6))
    plt.rcParams['text.usetex'] = True  # enable LaTeX text rendering
    plt.rcParams['font.size'] = 24      # set global font size

    # Create the plot
    plot = sns.histplot(df['avg_conf'], binwidth=10, stat='count', binrange=(0, 100))

    # figure adjustments
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Average depth confidence (\%)')          # set x-axis label
    plt.ylabel('Frames')   # set y-axis label

    # save the plot to pdf
    plot.get_figure().savefig(output_file, bbox_inches='tight')

    print(f'Figure saved to {output_file}')


def associate_confidence_with_results(results, avg_conf):
    """
    Associate the average depth confidence with the results.

    Args:
        results (list): list of dictionaries with the experiment information
        avg_conf (pd.DataFrame): dataframe with the average depth confidence values
    """
    T_psnr = {
        'RGB': None,
        'RGBD': None
    }
    df_psnr = {
        1: T_psnr.copy(),
        3: T_psnr.copy(),
        5: T_psnr.copy(),
        11: T_psnr.copy()
    }

    # average the PSNR values of different seeds
    for T in [1, 3, 5, 11]:
        for input_data in ['RGB', 'RGBD']:
            df_list = []
            for seed in [10, 13, 17]:
                df = None
                for r in results:
                    if r['data_input'] == input_data and r['seed'] == seed and r['T'] == T:
                        df = r['df']
                        break
                if df is None:
                    print(f'No data found for {input_data} and seed {seed}')
                    continue
                df_list.append(df[['clip_name', 'frame', 'psnr']])
            df_mean = pd.concat(df_list).groupby(['clip_name', 'frame']).mean().reset_index()
            df_psnr[T][input_data] = df_mean

    # associate the average confidence with the PSNR values
    clip_names = df_psnr[1]['RGB']['clip_name']
    frames = df_psnr[1]['RGB']['frame']
    rows = []
    with tqdm(total=len(clip_names) * 2 * 4) as pbar:
        for clip, frame in zip(clip_names, frames):
            for T in [1, 3, 5, 11]:
                for data_input in ['RGB', 'RGBD']:
                    psnr = df_psnr[T][data_input].loc[df_psnr[T][data_input]['clip_name'] == clip].loc[df_psnr[T][data_input]['frame'] == frame]['psnr'].values
                    avg_confidence = avg_conf.loc[avg_conf['clip_name'] == clip].loc[avg_conf['frame'] == frame]['avg_conf'].values
                    rows.append({
                        'clip_name': clip,
                        'frame': frame,
                        'T': T,
                        'data_input': data_input,
                        'psnr': psnr[0],
                        'avg_conf': avg_confidence[0]
                    })
                    pbar.update(1)
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(rows)
    return df


def aggregate_results_by_confidence(df):
    """
    Aggregate the results by the average depth confidence.

    Args:
        df (pd.DataFrame): dataframe with the results
    """

    # create bins: 0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100
    bins = np.arange(0, 110, 10)
    bin_labels = [f'{l}-{r}' for l, r in zip(bins[:-1], bins[1:])]

    # compute the PSNR gain for each bin
    rows = []
    for bin in bin_labels:
        for T in [1, 3, 5, 11]:
            l, r = bin.split('-')
            l, r = float(l), float(r)
            psnr_gain = df.loc[df['avg_conf'] >= l].loc[df['avg_conf'] < r].loc[df['data_input'] == 'RGBD'].loc[df['T'] == T]['psnr'].mean() - df.loc[df['avg_conf'] >= l].loc[df['avg_conf'] < r].loc[df['data_input'] == 'RGB'].loc[df['T'] == T]['psnr'].mean()
            rows.append({
                'psnr_gain': psnr_gain,
                'avg_conf_bin': bin,
                'T': T,
            })

    df_bins = pd.DataFrame(rows)
    return df_bins, bin_labels


def generate_confidence_psnr_curve_per_confidence_bin(df_bins, bin_labels, output_file):
    """
    Generate the PSNR gain curve per confidence bin.

    Args:
        df (pd.DataFrame): dataframe with the PSNR gain per confidence bin
        output_file (str): path to save the output figure
    """

    # set figure settings
    sns.set_theme()                     # set the default seaborn theme
    sns.set_context('poster')           # set the default seaborn context 
    plt.rcParams['text.usetex'] = True  # enable LaTeX text rendering
    plt.rcParams['font.size'] = 24      # set global font size
    plt.figure(figsize=(11, 6))         # set the figure size
    palette = sns.color_palette('rocket_r', n_colors=4)  # set the color palette

    # Create bar plot
    plot = sns.barplot(df_bins, x='avg_conf_bin', y='psnr_gain', hue='T', palette=palette)

    # get current x-axis ticks
    ticks = plt.xticks()[0]
    # create Text objects for the x-axis labels
    labels = bin_labels
    labels = [Text(0, 0, l) for l in labels]  # create Text objects
    # update ticks labels
    plt.xticks(ticks, labels)
    plt.xticks(rotation=45)              # rotate the x-axis ticks

    # ticks font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=20)

    # figure adjustments
    legend = plot.get_legend()          # remove the title from the legend
    legend.set_title('$T$ (frames)')
    plt.xlabel('Average depth confidence (\%)')          # set x-axis label
    plt.ylabel(r'$\Delta$ PSNR (dB)')   # set y-axis label

    # save the plot to pdf
    plot.get_figure().savefig(output_file, bbox_inches='tight')
    print(f'Figure saved to {output_file}')


def main():
    # parse the arguments
    args = parse_args()
    # read the average depth confidence values
    avg_conf = read_avg_confidence(args.avg_conf_csv)
    # generate the histogram of the confidence values
    confiddence_histogram(avg_conf, args.hist_figure)

    # read the csv files
    results = read_csv_results(args.exp_root, exp_info)
    # associate the average depth confidence with the results
    df = associate_confidence_with_results(results, avg_conf)
    # aggregate the results by the average depth confidence
    df_bins, bin_labels = aggregate_results_by_confidence(df)

    # generate the PSNR gain curve per confidence bin
    generate_confidence_psnr_curve_per_confidence_bin(df_bins, bin_labels, args.psnr_figure)


if __name__ == '__main__':
    main()
