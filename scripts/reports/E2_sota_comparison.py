import os
import pandas as pd
import numpy as np
import math

# default root folder for experiments
exp_root = os.getenv('RESULTS_ROOT')

# dictionary with experiments information
exp_info = {
    'DGN (1 fr.)':{
        'folder': 'E2_SOTA_comparison/000_test_DGN_single_frame',
        # 'flops': 
        # 'params': 
        # 'training_time': 
    },
    'Shift-Net (1 fr)':{
        'folder': ['E1_depth_impact/000_test_shiftnet_baseRGB_in01_out01_seed10',
                   'E1_depth_impact/000_test_shiftnet_baseRGB_in01_out01_seed13',
                   'E1_depth_impact/000_test_shiftnet_baseRGB_in01_out01_seed17'],
        # 'flops': 
        # 'params': 
        # 'training_time': 
    },
    'Shift-Net+D (1 fr)':{
        'folder': ['E1_depth_impact/001_test_shiftnet_RGBD_in01_out01_seed10',
                   'E1_depth_impact/001_test_shiftnet_RGBD_in01_out01_seed13',
                   'E1_depth_impact/001_test_shiftnet_RGBD_in01_out01_seed17'],
        # 'flops': 
        # 'params': 
        # 'training_time': 
    },
    'DGN (5 fr.)':{
        'folder': 'E2_SOTA_comparison/000_test_DGN_multi_frame',
        # 'flops': 
        # 'params': 
        # 'training_time': 
    },
    'EDVR':{
        'folder': 'E2_SOTA_comparison/001_test_EDVR',
        # 'flops': 
        # 'params': 
        # 'training_time': 
    },
    'RVRT':{
        'folder': 'E2_SOTA_comparison/003_test_RVRT',
        # 'flops': 
        # 'params': 
        # 'training_time': 
    },
    'VRT':{
        'folder': 'E2_SOTA_comparison/002_test_VRT',
        # 'flops': 
        # 'params': 
        # 'training_time': 
    },
    'Shift-Net (11 fr)':{
        'folder': ['E1_depth_impact/010_test_shiftnet_baseRGB_in11_out09_seed10',
                   'E1_depth_impact/010_test_shiftnet_baseRGB_in11_out09_seed13',
                   'E1_depth_impact/010_test_shiftnet_baseRGB_in11_out09_seed17'],
        # 'flops': 
        # 'params': 
        # 'training_time': 
    },
    'Shift-Net+D (11 fr)':{
        'folder': ['E1_depth_impact/011_test_shiftnet_RGBD_in11_out09_seed10',
                   'E1_depth_impact/011_test_shiftnet_RGBD_in11_out09_seed13',
                   'E1_depth_impact/011_test_shiftnet_RGBD_in11_out09_seed17'],
        # 'flops': 
        # 'params': 
        # 'training_time': 
    },
}


def parse_args():
    """Parse the arguments for the script."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--output_table', type=str, default='./Tables/table_E2_sota_comparison.csv')
    return parser.parse_args()


def read_csv_results(exp_root, exp_info):
    """
    Read the csv files with the results of the experiments.

    Args:
        exp_root (str): root folder for the experiments
        exp_info (dict): dictionary with the information of the experiments

    Returns:
        results: dictionary with the results of the experiments
    """
    results = {}
    for exp_name, info in exp_info.items():
        results[exp_name] = {}
        # results[exp_name]['flops'] = info['flops']
        # results[exp_name]['params'] = info['params']
        # results[exp_name]['training_time'] = info['training_time']
        results[exp_name]['dfs'] = []
        
        if info['folder'] is None:
            print(f'No results found for {exp_name}')
            results[exp_name] = None
            continue
        if not isinstance(info['folder'], list):
            info['folder'] = [info['folder']]
        for folder in info['folder']:
            exp_path = os.path.join(exp_root, folder, 'results')
            files = os.listdir(exp_path)
            # files = [f for f in files if os.path.isfile(os.path.join(exp_path, f))]
            files = [f for f in files if f.endswith('.csv')]
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
                results[exp_name]['dfs'].append(df)
            else:
                print(f'No results found for {exp_name}')
                results[exp_name]['dfs'].append(None)
    return results


def get_mean_metrics(results):
    """
    Compute the mean PSNR and SSIM for each experiment.

    Args:
        results (dict): dictionary with the results of the experiments

    Returns:
        metrics: dictionary with all the performance metrics
    """
    metrics = {}
    for exp_name, info in results.items():
        if info is None:
            continue
        metrics[exp_name] = {}
        metrics[exp_name]['psnr'] = []
        metrics[exp_name]['ssim'] = []
        # metrics[exp_name]['flops'] = info['flops']
        # metrics[exp_name]['params'] = info['params']
        # metrics[exp_name]['training_time'] = info['training_time']
        for df in info['dfs']:
            if df is None:
                continue
            metrics[exp_name]['psnr'].append(df['psnr'].mean())
            metrics[exp_name]['ssim'].append(df['ssim'].mean())
        metrics[exp_name]['psnr'] = np.mean(metrics[exp_name]['psnr'])
        metrics[exp_name]['ssim'] = np.mean(metrics[exp_name]['ssim'])
    return metrics


def mean_metrics_to_df(metrics):
    """
    Convert metrics dictionary to a pandas DataFrame.

    Args:
        metrics (dict): dictionary with the performance metrics

    Returns:
        df: pandas DataFrame with the performance metrics
    """
    df = pd.DataFrame(metrics).T
    # df = df.rename(columns={'psnr': 'PSNR', 'ssim': 'SSIM', 'flops': 'FLOPs', 'params': 'Params', 'training_time': 'Training time'})
    df = df.rename(columns={'psnr': 'PSNR', 'ssim': 'SSIM'})

    # 2 decimal places for PSNR column
    df['PSNR'] = df['PSNR'].apply(lambda x: round(x, 2))
    # 4 decimal places for SSIM column
    df['SSIM'] = df['SSIM'].apply(lambda x: round(x, 4))
    # # Flops in G
    # df['FLOPs'] = df['FLOPs'] / 1e9
    # df['FLOPs'] = df['FLOPs'].apply(lambda x: round(x, 1))
    # # Params in M
    # df['Params'] = df['Params'] / 1e6
    # df['Params'] = df['Params'].apply(lambda x: round(x, 2))
    # # approx training time to second unit 
    # df['Training time'] = df['Training time'].apply(lambda x: math.ceil(x / 10) * 10)

    return df


def main():
    # parse the arguments
    args = parse_args()
    # read the csv files
    results = read_csv_results(args.exp_root, exp_info)
    # get the mean of the metrics
    metrics = get_mean_metrics(results)
    # convert the metrics to a DataFrame
    df = mean_metrics_to_df(metrics)
    # save the table to a csv file
    os.makedirs(os.path.dirname(args.output_table), exist_ok=True)
    df.to_csv(args.output_table)
    print(df)


if __name__ == '__main__':
    main()