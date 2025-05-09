import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    parser.add_argument('--output_table', type=str, default='./Tables/table_E1_depth_impact.csv')
    parser.add_argument('--output_figure', type=str, default='./Figures/fig_E1_depth_impact.pdf')
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


def get_mean(results, metrics=['psnr']):
    """
    Get the mean of the metrics for each experiment.
    """
    data = []
    for res in results:
        exp_folder = res['exp_folder']
        T = res['T']
        data_input = res['data_input']
        seed = res['seed']
        df = res['df']
        row = {
            'exp_folder': exp_folder,
            'T': T,
            'data_input': data_input,
            'seed': seed
        }
        for metric in metrics:
            row[metric] = df[metric].mean()
        data.append(row)
    return pd.DataFrame(data)


def generate_gain_psnr_curve(df, output_file):
    """
    Generate the gain PSNR curve to evaluate the impact of the depth 
    information on the quality of the deblurred images.

    Args:
        df (pd.DataFrame): dataframe with the mean PSNR values for each experiment
        output_file (str): path to save the output figure
    """

    # get mean PSNR for RGB
    df_mean_rgb = df[df['data_input'] == 'RGB'].groupby('T').agg({'psnr': 'mean'}).reset_index()
    # repeat the mean PSNR for seed and data_input
    df_mean_rgb = df_mean_rgb.loc[df_mean_rgb.index.repeat(6)].reset_index(drop=True)

    # substraction to get the improvement
    df_diff = df.copy()
    df_diff['psnr'] = df_diff['psnr'] - df_mean_rgb['psnr']

    # set figure settings
    sns.set_theme()                     # set the default seaborn theme
    sns.set_context('poster')           # set the default seaborn context 
    plt.rcParams['text.usetex'] = True  # enable LaTeX text rendering
    plt.rcParams['font.size'] = 22      # set global font size
    custom_ticks = np.arange(1, 12, 2)  # create custom ticks for the x-axis

    # Create the plot
    plot = sns.lineplot(data=df_diff, x="T", y="psnr", hue="data_input", style="data_input", markers=True)

    # figure adjustments
    legend = plot.get_legend()          # remove the title from the legend
    legend.set_title('')
    plt.xticks(custom_ticks)            # set custom ticks on the x-axis
    plt.xlabel('$T$ (frames)')          # set x-axis label
    plt.ylabel(r'$\Delta$ PSNR (dB)')   # set y-axis label

    # Save figure as pdf
    plt.savefig(output_file, bbox_inches='tight')


def generate_psnr_comparison_table(df, output_file):
    """
    Generate the table with the mean PSNR values for each experiment.

    Args:
        df (pd.DataFrame): dataframe with the mean PSNR values for each experiment
        output_file (str): path to save the output table
    """
    df_mean_rgb = df[df['data_input'] == 'RGB'].groupby('T').agg({'psnr': 'mean'}).reset_index()
    df_mean_rgbd = df[df['data_input'] == 'RGBD'].groupby('T').agg({'psnr': 'mean'}).reset_index()
    df_diff = df_mean_rgbd - df_mean_rgb

    # create the table with T vs PSNR for RGB and RGBD
    df_table = pd.DataFrame({
        'T (frames)': df_mean_rgb['T'],
        'RGB': df_mean_rgb['psnr'],
        'RGBD': df_mean_rgbd['psnr'],
        'Diff. ($\pm$dB)': df_diff['psnr']
    })

    # transpose the table
    df_table = df_table.set_index('T (frames)').T

    # use ".3f" to format psnr to 3 decimal places
    df_table = df_table.apply(lambda col: col.map(lambda x: f'{x:.3f}'))

    # add sign symbol to the difference
    df_table.loc['Diff. ($\pm$dB)'] = df_table.loc['Diff. ($\pm$dB)'].apply(lambda x: f'+{x}' if float(x) > 0 else f'-{x}')

    # save the table to a csv file
    df_table.to_csv(output_file)
    print(df_table)


def main():
    # parse the arguments
    args = parse_args()
    # read the csv files
    results = read_csv_results(args.exp_root, exp_info)
    # get the mean of the metrics
    mean_results = get_mean(results)

    # generate the PSNR gain curve
    os.makedirs(os.path.dirname(args.output_figure), exist_ok=True)
    generate_gain_psnr_curve(mean_results, args.output_figure)

    # generate the PSNR comparison table
    os.makedirs(os.path.dirname(args.output_table), exist_ok=True)
    generate_psnr_comparison_table(mean_results, args.output_table)


if __name__ == '__main__':
    main()
