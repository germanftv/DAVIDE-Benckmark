import os
import pandas as pd
import numpy as np

# default root folder of the experiments
exp_root = os.path.join(os.getenv('RESULTS_ROOT'), 'E0_arch_ablation')

# Experiment info
exp_info ={
    'CatConv_RGBD':{
        'exp_folders':['000_test_shiftnet_RGBD_concatconv_shiftdepth0_seed10', '001_test_shiftnet_RGBD_concatconv_shiftdepth0_seed13', '002_test_shiftnet_RGBD_concatconv_shiftdepth0_seed17'],
        'fusion_block': 'Concat+Conv',
        'GSS': False
    },
    'CatConv_GSS_RGBD':{
        'exp_folders':['003_test_shiftnet_RGBD_concatconv_shiftdepth1_seed10', '004_test_shiftnet_RGBD_concatconv_shiftdepth1_seed13', '005_test_shiftnet_RGBD_concatconv_shiftdepth1_seed17'],
        'fusion_block': 'Concat+Conv',
        'GSS': True
    },
    'SFT_RGBD':{
        'exp_folders':['006_test_shiftnet_RGBD_sft_shiftdepth0_seed10', '007_test_shiftnet_RGBD_sft_shiftdepth0_seed13', '008_test_shiftnet_RGBD_sft_shiftdepth0_seed17'],
        'fusion_block': 'SFT',
        'GSS': False
    },
    'SFT_GSS_RGBD':{
        'exp_folders':['009_test_shiftnet_RGBD_sft_shiftdepth1_seed10', '010_test_shiftnet_RGBD_sft_shiftdepth1_seed13', '011_test_shiftnet_RGBD_sft_shiftdepth1_seed17'],
        'fusion_block': 'SFT',
        'GSS': True
    },
    'DAM_RGBD':{
        'exp_folders':['012_test_shiftnet_RGBD_dam_shiftdepth0_seed10', '013_test_shiftnet_RGBD_dam_shiftdepth0_seed13', '014_test_shiftnet_RGBD_dam_shiftdepth0_seed17'],
        'fusion_block': 'DAM',
        'GSS': False
    },
    'DAM_GSS_RGBD':{
        'exp_folders':['015_test_shiftnet_RGBD_dam_shiftdepth1_seed10', '016_test_shiftnet_RGBD_dam_shiftdepth1_seed13', '017_test_shiftnet_RGBD_dam_shiftdepth1_seed17'],
        'fusion_block': 'DAM',
        'GSS': True
    },
    'DaT_RGBD':{
        'exp_folders':['018_test_shiftnet_RGBD_dat_shiftdepth0_seed10', '019_test_shiftnet_RGBD_dat_shiftdepth0_seed13', '020_test_shiftnet_RGBD_dat_shiftdepth0_seed17'],
        'fusion_block': 'DaT',
        'GSS': False
    },
    'DaT_GSS_RGBD':{
        'exp_folders':['021_test_shiftnet_RGBD_dat_shiftdepth1_seed10', '022_test_shiftnet_RGBD_dat_shiftdepth1_seed13', '023_test_shiftnet_RGBD_dat_shiftdepth1_seed17'],
        'fusion_block': 'DaT',
        'GSS': True
    },
}


def parse_args():
    """Parse the arguments for the script."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--output_table', type=str, default='./Tables/table_E0_arch_ablation.csv')
    return parser.parse_args()


def read_csv_results(exp_root, exp_info):
    """
    Read the csv files in the results folder of the experiments.

    Args:
        exp_root (str): root folder of the experiments
        exp_info (dict): dictionary with the experiment information
    """
    results = {}
    for exp_name, exp_data in exp_info.items():
        results[exp_name] = []
        for exp_folder in exp_data['exp_folders']:
            exp_path = os.path.join(exp_root, exp_folder, 'results')
            files = os.listdir(exp_path)
            files = [f for f in files if f.endswith('.csv')]
            if len(files) > 0:
                # take the most recent file
                files.sort(key=lambda x: os.path.getmtime(os.path.join(exp_path, x)), reverse=True)
                df = pd.read_csv(os.path.join(exp_path, files[0]))
                results[exp_name].append(df)
            else:
                print(f'No results found for {exp_folder}')
                results[exp_name].append(None)
    return results


def get_mean_std(results, metrics= ['psnr', 'ssim']):
    """
    Get the mean and standard deviation of the metrics for each experiment.

    Args:
        results (dict): dictionary with the experiment information
        metrics (list): list of metrics to compute the mean and standard deviation. Default: ['psnr', 'ssim']
    """
    mean_std = {}
    for exp_name, exp_results in results.items():
        mean_std[exp_name] = {}
        for metric in metrics:
            metric_values = []
            for df in exp_results:
                if df is not None:
                    metric_values.append(df[metric].mean())
            if len(metric_values) > 0:
                metric_values = np.array(metric_values)
                mean = np.mean(metric_values, axis=0)
                std = np.std(metric_values, axis=0)
                mean_std[exp_name][metric] = (mean, std)
    return mean_std


def main():
    # parse the arguments
    args = parse_args()
    # read the csv files
    results = read_csv_results(args.exp_root, exp_info)
    # get the mean and standard deviation
    mean_std = get_mean_std(results)

    # Create a pandas table
    df_exp = pd.DataFrame(columns=['Model','Fusion Block', 'GSS', 'PSNR', 'SSIM'])
    df_exp['Model'] = mean_std.keys()
    df_exp = df_exp.set_index('Model')
    # fill the table
    for model, data in mean_std.items():
        df_exp.loc[model, 'Fusion Block'] = exp_info[model]['fusion_block']
        df_exp.loc[model, 'GSS'] = exp_info[model]['GSS']
        df_exp.loc[model, 'PSNR'] = f"{data['psnr'][0]:.2f} ± {data['psnr'][1]:.2f}"
        df_exp.loc[model, 'SSIM'] = f"{data['ssim'][0]:.4f} ± {data['ssim'][1]:.4f}"

    # make index as a column
    df_exp.reset_index(inplace=True)
    # remove 'Model' column
    df_exp.drop(columns=['Model'], inplace=True)

    print(df_exp)
    # save the table
    os.makedirs(os.path.dirname(args.output_table), exist_ok=True)
    df_exp.to_csv(args.output_table, index=False)


if __name__ == '__main__':
    main()