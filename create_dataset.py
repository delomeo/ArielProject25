#pylint: disable-all
import os
import glob 

from typing import List, Tuple, Optional, Dict
import cupy as cp
from cupy import cuda
import numpy as np
import pandas as pd

import yaml
from tqdm import tqdm
from data_preprocessing import (ADC_convert, mask_hot_dead, apply_linear_corr, clean_dark, get_cds, bin_obs, correct_flat_field) 

def get_index(files, chunk_size, interval) -> cp.ndarray:
    start, stop = interval[0], interval[1]
    
    # Extract unique parent folder numbers
    idxs = set()
    for f in files:
        parent_folder = os.path.basename(os.path.dirname(f))
        if parent_folder.isdigit():  # Ensure the folder name is numeric
            idxs.add(int(parent_folder))
    
    # Convert to a sorted list
    idxs = sorted(idxs)
    
    # Slice the indices based on the interval
    idxs = idxs[start:stop]
    
    # Split indices into chunks
    return cp.array_split(cp.array(idxs), max(1, len(idxs) // chunk_size))

def load_data (file, chunk_size, nb_files) -> cp.ndarray: 
    data0 = cp.load(file + '_0.npy')
    data_all = cp.zeros((nb_files*chunk_size, data0.shape[1], data0.shape[2], data0.shape[3]))
    data_all[:chunk_size] = data0
    for i in range (1, nb_files): 
        data_all[i*chunk_size:(i+1)*chunk_size] = cp.load(file + '_{}.npy'.format(i))
    return data_all 

def load_config(name='config.yaml') -> Dict:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_dir(path) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    else:
        print(f"Directory {path} already exists. Skipping creation.")

def load_and_process_chunk(index_chunk, config, axis_info, train_test='train') -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Loads and processes a chunk of data for both AIRS and FGS instruments.
    """
    path_folder = config['PATH_FOLDER']
    cut_inf, cut_sup = config.get('CUT_INF', 39), config.get('CUT_SUP', 321)
    l = cut_sup - cut_inf
    chunk_size = len(index_chunk)

    # Pre-allocate GPU memory for the entire chunk
    # AIRS
    # TODO: Allocate space in memory if certain transformations are needed 
    all_airs_signals = cp.empty((chunk_size, 11250, 32, l), dtype=cp.float64)
    all_airs_deads = cp.empty((chunk_size, 32, l), dtype=cp.float64)
    all_airs_darks = cp.empty((chunk_size, 32, l), dtype=cp.float64)
    all_airs_flats = cp.empty((chunk_size, 32, l), dtype=cp.float64)
    if config.get('DO_THE_NL_CORR', False):
        all_airs_linear_corrs = cp.empty((chunk_size, 6, 32, l), dtype=cp.float64)

    # FGS1
    # TODO: Allocate space in memory if certain transformations are needed 
    all_fgs_signals = cp.empty((chunk_size, 135000, 32, 32), dtype=cp.float64)
    all_fgs_deads = cp.empty((chunk_size, 32, 32), dtype=cp.float64)
    all_fgs_darks = cp.empty((chunk_size, 32, 32), dtype=cp.float64)
    all_fgs_flats = cp.empty((chunk_size, 32, 32), dtype=cp.float64)
    if config.get('DO_THE_NL_CORR', False):
        all_fgs_linear_corrs = cp.empty((chunk_size, 6, 32, 32), dtype=cp.float64)

    # Load all data for the chunk from disk first
    for i, idx in enumerate(index_chunk):
        # AIRS data
        df_airs = pd.read_parquet(os.path.join(path_folder, f'{train_test}/{idx}/AIRS-CH0_signal_0.parquet'))
        all_airs_signals[i] = cp.asarray(df_airs.values.reshape((df_airs.shape[0], 32, 356))[:, :, cut_inf:cut_sup])
        
        calib_path_airs = os.path.join(path_folder, f'{train_test}/{idx}/AIRS-CH0_calibration_0')
        all_airs_darks[i] = cp.asarray(pd.read_parquet(os.path.join(calib_path_airs, 'dark.parquet')).values.reshape((32, 356))[:, cut_inf:cut_sup])
        all_airs_deads[i] = cp.asarray(pd.read_parquet(os.path.join(calib_path_airs, 'dead.parquet')).values.reshape((32, 356))[:, cut_inf:cut_sup])
        all_airs_flats[i] = cp.asarray(pd.read_parquet(os.path.join(calib_path_airs, 'flat.parquet')).values.reshape((32, 356))[:, cut_inf:cut_sup])
        if config.get('DO_THE_NL_CORR', False):
            all_airs_linear_corrs[i] = cp.asarray(pd.read_parquet(os.path.join(calib_path_airs, 'linear_corr.parquet')).values.reshape((6, 32, 356))[:, :, cut_inf:cut_sup])

        # FGS1 data
        df_fgs = pd.read_parquet(os.path.join(path_folder, f'{train_test}/{idx}/FGS1_signal_0.parquet'))
        all_fgs_signals[i] = cp.asarray(df_fgs.values.reshape((df_fgs.shape[0], 32, 32)))

        calib_path_fgs = os.path.join(path_folder, f'{train_test}/{idx}/FGS1_calibration_0')
        all_fgs_darks[i] = cp.asarray(pd.read_parquet(os.path.join(calib_path_fgs, 'dark.parquet')).values.reshape((32, 32)))
        all_fgs_deads[i] = cp.asarray(pd.read_parquet(os.path.join(calib_path_fgs, 'dead.parquet')).values.reshape((32, 32)))
        all_fgs_flats[i] = cp.asarray(pd.read_parquet(os.path.join(calib_path_fgs, 'flat.parquet')).values.reshape((32, 32)))
        if config.get('DO_THE_NL_CORR', False):
            all_fgs_linear_corrs[i] = cp.asarray(pd.read_parquet(os.path.join(calib_path_fgs, 'linear_corr.parquet')).values.reshape((6, 32, 32)))

    # Process AIRS chunk
    airs_clean = ADC_convert(all_airs_signals)
    if config.get('DO_MASK', True):
        airs_clean = mask_hot_dead(airs_clean, all_airs_deads, all_airs_darks)
    if config.get('DO_THE_NL_CORR', False):
        airs_clean = apply_linear_corr(all_airs_linear_corrs, airs_clean)
    if config.get('DO_DARK', True):
        dt_airs = (axis_info['AIRS-CH0-integration_time'].dropna().values)
        # dt_airs = dt_airs[~cp.any(cp.isnan(dt_airs), axis=0)]
        dt_airs[1::2] += 0.1  # Adjust for even/odd integration times
        dt_airs = cp.asarray(dt_airs, dtype=cp.float64)

        airs_clean = clean_dark(airs_clean, all_airs_deads, all_airs_darks, dt_airs)
    
    airs_cds = get_cds(airs_clean)
    del airs_clean, all_airs_signals, all_airs_darks # Free memory
    
    if config.get('DO_FLAT', True):
        airs_cds = correct_flat_field(all_airs_flats, all_airs_deads, airs_cds)
    del all_airs_flats, all_airs_deads

    # Process FGS1 chunk
    fgs_clean = ADC_convert(all_fgs_signals)
    if config.get('DO_MASK', True):
        fgs_clean = mask_hot_dead(fgs_clean, all_fgs_deads, all_fgs_darks)
    if config.get('DO_THE_NL_CORR', False):
        fgs_clean = apply_linear_corr(all_fgs_linear_corrs, fgs_clean)
    if config.get('DO_DARK', True):
        dt_fgs1 = cp.ones(fgs_clean.shape[1]) * 0.1
        dt_fgs1[1::2] += 0.1
        fgs_clean = clean_dark(fgs_clean, all_fgs_deads, all_fgs_darks, dt_fgs1)

    fgs_cds = get_cds(fgs_clean)
    del fgs_clean, all_fgs_signals, all_fgs_darks # Free memory

    if config.get('DO_FLAT', True):
        fgs_cds = correct_flat_field(all_fgs_flats, all_fgs_deads, fgs_cds)
    del all_fgs_flats, all_fgs_deads

    # Time Binning
    if config.get('TIME_BINNING'):
        airs_cds_binned = bin_obs(airs_cds, binning=30)
        fgs_cds_binned = bin_obs(fgs_cds, binning=30 * 12)
    else:
        airs_cds_binned = airs_cds.transpose(0, 1, 3, 2)
        fgs_cds_binned = fgs_cds.transpose(0, 1, 3, 2)
        
    return airs_cds_binned, fgs_cds_binned

def compress_and_save(folder: str = [], data: str = '', output: str = 'output.npz') -> None: 
    """
    Compresses data in a single archize and saves it.
    """
    if data == '':
        raise ValueError("Data name must be provided.")
    
    files = glob.glob(os.path.join(folder, f'{data}_*.npy'))
    arrays = {}
    for file in files:
        name = os.path.basename(file).replace('.npy', '')
        arrays[name] = cp.load(file)
    
    # Save as a compressed .npz file
    cp.savez_compressed(output, **arrays)
    print(f"Data saved to {output}.")
    # Clean up GPU memory
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    print("GPU memory cleaned up.")
    del arrays, files
    # os.remove(files)  # Optionally remove individual files after saving

def main():
    config = load_config()

    path_folder = config['PATH_FOLDER']
    path_out = config['PATH_OUT']
    tmp_path = config['OUTPUT_DIR']
    
    create_dir(path_out)
    create_dir(tmp_path)
    
    train_test = config['TRAIN_TEST']
    CHUNKS_SIZE = config.get('CHUNKS_SIZE', 1)
    INTERVAL = config.get('INTERVAL', [0, 1])

    files = glob.glob(os.path.join(path_folder, f'{train_test}/*/'))
    index_chunks = get_index(files, CHUNKS_SIZE, INTERVAL)  
    axis_info = pd.read_parquet(os.path.join(path_folder, 'axis_info.parquet'))

    for n, index_chunk in enumerate(tqdm(index_chunks, desc="Processing Chunks")):
        
        AIRS_binned, FGS1_binned = load_and_process_chunk(index_chunk, config, axis_info)

        # Save data
        cp.save(os.path.join(tmp_path, f'AIRS_clean_train_{n}.npy'), AIRS_binned)
        cp.save(os.path.join(tmp_path, f'FGS1_train_{n}.npy'), FGS1_binned)
        
        # Clean up GPU memory after each chunk
        del AIRS_binned, FGS1_binned
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

        print(f"Chunk {n} processed and saved.")

    compress_and_save(tmp_path, data='AIRS_clean_train', output=os.path.join(path_out, 'AIRS_clean_train.npz'))
    compress_and_save(tmp_path, data='FGS1_clean_train', output=os.path.join(path_out, 'FGS1_clean_train.npz'))
    print(f"Data compressed and saved to {tmp_path}.")

    
    print("All chunks processed successfully.")

if __name__ == "__main__":
    main()