import os
import pandas as pd
import numpy as np

# default root folder of the experiments
exp_root = os.path.join(os.getenv('RESULTS_ROOT'), 'E0_depth_quality')

# list of dictionaries with the experiment information  
exp_info=[
    {
        'exp_folder': '000_test_shiftnet_depth_quality_sensor',
        'quality': 'sensor'
    },
    {
        'exp_folder': '001_test_shiftnet_depth_quality_mono_blur',
        'quality': 'mono_blur'
    },
    {
        'exp_folder': '002_test_shiftnet_depth_quality_mono_sharp',
        'quality': 'mono_sharp'
    }
]

def parse_args():
    """Parse the arguments for the script."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--output_table', type=str, default='./Tables/table_E0_depth_quality.csv')
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
        quality = exp['quality']
        
        exp_path = os.path.join(exp_root, exp_folder, 'results')
        files = os.listdir(exp_path)
        files = [f for f in files if f.endswith('.csv')]
        if len(files) == 0:
            print(f'No csv files found in {exp_path}')
            continue
        if len(files) > 0:
            # take the most recent file
            files.sort(key=lambda x: os.path.getmtime(os.path.join(exp_path, x)), reverse=True)
            df = pd.read_csv(os.path.join(exp_path, files[0]))
            results.append({
                'quality': quality,
                'df': df
            })
    return results


def get_mean(results, metrics=['psnr', 'ssim']):
    """
    Get the mean of the metrics for each experiment.

    Args:
        results (list): list of dictionaries with the experiment information
        metrics (list): list of metrics to compute the mean
    """
    mean_results = []
    for result in results:
        quality = result['quality']
        df = result['df']
        mean = df[metrics].mean()
        mean_results.append({
            'quality': quality,
            'psnr': mean['psnr'],
            'ssim': mean['ssim']
        })
    return pd.DataFrame(mean_results)


def main():
    # parse the arguments
    args = parse_args()
    # read the csv files
    results = read_csv_results(args.exp_root, exp_info)
    # get the mean of the metrics
    mean_results = get_mean(results)
    
    # switch rows 0 and 1
    df_exp = mean_results.reindex([1, 0, 2])

    # use ".2f" to format psnr to 2 decimal places
    # use ".3f" to format ssim to 3 decimal places
    df_exp['psnr'] = df_exp['psnr'].apply(lambda x: f'{x:.2f}')
    df_exp['ssim'] = df_exp['ssim'].apply(lambda x: f'{x:.3f}')

    # transpose the dataframe mean_results and print in latex format as:
    # quality | mono_blur | sensor | mono_sharp
    # -----------------------------------------
    # PSNR    | X         | Y      | Z
    # SSIM    | X         | Y      | Z
    df_exp = df_exp.set_index('quality').T

    print(df_exp)
    # save the table to a csv file
    os.makedirs(os.path.dirname(args.output_table), exist_ok=True)
    df_exp.to_csv(args.output_table)


if __name__ == '__main__':
    main()