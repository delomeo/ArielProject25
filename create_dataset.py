#pylint: disable-all
import os
import glob 

import cupy as cp
from cupy import cuda
import numpy as np
import pandas as pd

import yaml
from tqdm import tqdm
from data_preprocessing import (ADC_convert, mask_hot_dead, apply_linear_corr, clean_dark, get_cds, bin_obs, correct_flat_field) 

def get_index(files, chunk_size, interval):
    start, stop = interval[0], interval[1]
    idxs = []
    for f in files[start:stop]:
        name = os.path.basename(f)
        parts = name.split('_')
        if parts[:3] == ["AIRS-CH0","signal","0.parquet"]:
            idxs.append(int(os.path.basename(os.path.dirname(f))))
    idxs = cp.sort(cp.array(idxs))
    return cp.array_split(idxs, max(1, len(idxs)//chunk_size))





def load_data (file, chunk_size, nb_files) : 
    data0 = cp.load(file + '_0.npy')
    data_all = cp.zeros((nb_files*chunk_size, data0.shape[1], data0.shape[2], data0.shape[3]))
    data_all[:chunk_size] = data0
    for i in range (1, nb_files) : 
        data_all[i*chunk_size:(i+1)*chunk_size] = cp.load(file + '_{}.npy'.format(i))
    return data_all 

def load_config(name='config.yaml'):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    else:
        print(f"Directory {path} already exists. Skipping creation.")


def main():
    
    global path_folder, path_out, output_dir

    config = load_config()

    # Fetch paths from config
    path_folder = config['PATH_FOLDER']
    path_out = config['PATH_OUT']
    output_dir = config['OUTPUT_DIR']
    
    create_dir(path_out) # Create output directory if it doesn't exist

    # Fetch additional configurations
    global CHUNKS_SIZE, DO_MASK, DO_THE_NL_CORR, DO_DARK, DO_FLAT, TIME_BINNING
    CHUNKS_SIZE = config.get('CHUNKS_SIZE', 1)
    DO_MASK = config.get('DO_MASK', True)
    DO_THE_NL_CORR = config.get('DO_THE_NL_CORR', False)
    DO_DARK = config.get('DO_DARK', True)
    DO_FLAT = config.get('DO_FLAT', True)
    TIME_BINNING = config.get('TIME_BINNING', None) # TODO: If none, do not use time binning, else use the value provided (default is 30)


def main():
    ## START OF THE MAIN CODE ##
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        print(f"Directory {path_out} created.")
    else:
        print(f"Directory {path_out} already exists.")

    files = glob.glob(os.path.join(path_folder + 'train/', '*/*'))

    index = get_index(files, CHUNKS_SIZE, INTERVAL)  

    axis_info = pd.read_parquet(os.path.join(path_folder,'axis_info.parquet'))
    DO_MASK = True
    DO_THE_NL_CORR = False
    DO_DARK = True
    DO_FLAT = True
    TIME_BINNING = True

    cut_inf, cut_sup = 39, 321
    l = cut_sup - cut_inf

    for n, index_chunk in enumerate(tqdm(index)):
        AIRS_CH0_clean = cp.zeros((CHUNKS_SIZE, 11250, 32, l))
        FGS1_clean = cp.zeros((CHUNKS_SIZE, 135000, 32, 32))
        
        for i in range (CHUNKS_SIZE) : 
            df = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/AIRS-CH0_signal_0.parquet'))
            signal = df.values.astype(cp.float64).reshape((df.shape[0], 32, 356))

            signal = ADC_convert(signal,)
            dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
            dt_airs[1::2] += 0.1
            chopped_signal = signal[:, :, cut_inf:cut_sup]
            del signal, df
            
            # CLEANING THE DATA: AIRS
            flat = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(cp.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            dark = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/dark.parquet')).values.astype(cp.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            dead_airs = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/dead.parquet')).values.astype(cp.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            linear_corr = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/linear_corr.parquet')).values.astype(cp.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
            
            if DO_MASK:
                chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
                AIRS_CH0_clean[i] = chopped_signal
            else:
                AIRS_CH0_clean[i] = chopped_signal
                
            if DO_THE_NL_CORR: 
                linear_corr_signal = apply_linear_corr(linear_corr,AIRS_CH0_clean[i])
                AIRS_CH0_clean[i,:, :, :] = linear_corr_signal
            del linear_corr
            
            if DO_DARK: 
                cleaned_signal = clean_dark(AIRS_CH0_clean[i], dead_airs, dark, dt_airs)
                AIRS_CH0_clean[i] = cleaned_signal
            else: 
                pass
            del dark
            
            df = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/FGS1_signal_0.parquet'))
            fgs_signal = df.values.astype(cp.float64).reshape((df.shape[0], 32, 32))

            
            fgs_signal = ADC_convert(fgs_signal, )
            dt_fgs1 = cp.ones(len(fgs_signal))*0.1
            dt_fgs1[1::2] += 0.1
            chopped_FGS1 = fgs_signal
            del fgs_signal, df
            
            # CLEANING THE DATA: FGS1
            flat = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/FGS1_calibration_0/flat.parquet')).values.astype(cp.float64).reshape((32, 32))
            dark = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/FGS1_calibration_0/dark.parquet')).values.astype(cp.float64).reshape((32, 32))
            dead_fgs1 = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/FGS1_calibration_0/dead.parquet')).values.astype(cp.float64).reshape((32, 32))
            linear_corr = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/FGS1_calibration_0/linear_corr.parquet')).values.astype(cp.float64).reshape((6, 32, 32))
            
            if DO_MASK:
                chopped_FGS1 = mask_hot_dead(chopped_FGS1, dead_fgs1, dark)
                FGS1_clean[i] = chopped_FGS1
            else:
                FGS1_clean[i] = chopped_FGS1

            if DO_THE_NL_CORR: 
                linear_corr_signal = apply_linear_corr(linear_corr,FGS1_clean[i])
                FGS1_clean[i,:, :, :] = linear_corr_signal
            del linear_corr
            
            if DO_DARK: 
                cleaned_signal = clean_dark(FGS1_clean[i], dead_fgs1, dark,dt_fgs1)
                FGS1_clean[i] = cleaned_signal
            else: 
                pass
            del dark
            
        # SAVE DATA AND FREE SPACE
        AIRS_cds = get_cds(AIRS_CH0_clean)
        FGS1_cds = get_cds(FGS1_clean)
        
        del AIRS_CH0_clean, FGS1_clean
        
        ## (Optional) Time Binning to reduce space
        if TIME_BINNING:
            AIRS_cds_binned = bin_obs(AIRS_cds,binning=30)
            FGS1_cds_binned = bin_obs(FGS1_cds,binning=30*12)
        else:
            AIRS_cds = AIRS_cds.transpose(0,1,3,2) ## this is important to make it consistent for flat fielding, but you can always change it
            AIRS_cds_binned = AIRS_cds
            FGS1_cds = FGS1_cds.transpose(0,1,3,2)
            FGS1_cds_binned = FGS1_cds
        
        del AIRS_cds, FGS1_cds
        
        for i in range (CHUNKS_SIZE):
            flat_airs = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(cp.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            flat_fgs = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i]}/FGS1_calibration_0/flat.parquet')).values.astype(cp.float64).reshape((32, 32))
            if DO_FLAT:
                corrected_AIRS_cds_binned = correct_flat_field(flat_airs,dead_airs, AIRS_cds_binned[i])
                AIRS_cds_binned[i] = corrected_AIRS_cds_binned
                corrected_FGS1_cds_binned = correct_flat_field(flat_fgs,dead_fgs1, FGS1_cds_binned[i])
                FGS1_cds_binned[i] = corrected_FGS1_cds_binned
            else:
                pass

        ## save data
        cp.save(os.path.join(path_out, 'AIRS_clean_train_{}.npy'.format(n)), AIRS_cds_binned)
        cp.save(os.path.join(path_out, 'FGS1_train_{}.npy'.format(n)), FGS1_cds_binned)
        del AIRS_cds_binned
        del FGS1_cds_binned

if __name__ == "__main__":
    main()
    print("Dataset creation completed successfully.")