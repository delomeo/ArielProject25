# %%writefile data_transforms_gpu.py

import cupy as cp

# NOTE: The original sigma_clip_numba is fundamentally a CPU-optimized function
# that relies on dynamic array resizing and selection (data[mask]).
# A direct, high-performance CuPy/CUDA kernel equivalent is complex to write
# and typically involves custom kernels.
# For simplicity and to avoid a full CUDA kernel rewrite, this version of
# sigma_clip_gpu will perform the clipping by masking with NaN and iterating.
# If performance for sigma_clip is critical, consider finding a pre-optimized
# CuPy-based sigma clip, or writing a custom CUDA kernel using Numba's CUDA capabilities.

def sigma_clip_gpu(data: cp.ndarray, sigma: float = 3.0, maxiters: int = 5) -> cp.ndarray:
    """
    Custom CuPy-compatible sigma clipping using NaN masking.
    This version returns an array with clipped values replaced by NaN.
    It simulates the iterative clipping process on the GPU.

    Parameters:
        data (cp.ndarray): 1D array of values on GPU.
        sigma (float): Clipping threshold.
        maxiters (int): Maximum number of iterations.
    Returns:
        clipped_data (cp.ndarray): Array with outliers replaced by NaN.
    """
    # Make a writable copy if input is not already
    clipped_data = data.copy()
    
    # Initialize mask for valid data (not NaN initially)
    mask = ~cp.isnan(clipped_data)

    for _ in range(maxiters):
        # Get current valid data for median/std calculation
        temp_data = clipped_data[mask]

        if temp_data.size == 0:
            break

        median = cp.median(temp_data)
        std = cp.std(temp_data)

        if std == 0: # Avoid division by zero if all values are the same
            break

        # Calculate bounds on GPU
        lower_bound = median - sigma * std
        upper_bound = median + sigma * std

        # Update mask: values within bounds are kept, others are marked for clipping
        new_mask = (clipped_data >= lower_bound) & (clipped_data <= upper_bound)
        
        # If the mask hasn't changed, we've converged
        if cp.all(new_mask == mask):
            break
        
        # Apply the new mask by setting outliers to NaN
        clipped_data[~new_mask] = cp.nan
        mask = new_mask # Update mask for next iteration

    return clipped_data

# STEP 1
def ADC_convert(signal: cp.ndarray, gain: float = 0.4369, offset: float = -1000) -> cp.ndarray:
    '''
    The Analog-to-Digital Conversion (adc) is performed by the detector to convert
    the pixel voltage into an integer number. Since we are using the same conversion number
    this year, we have simply hard-coded it inside.
    '''
    signal = signal.astype(cp.float64)
    signal /= gain
    signal += offset
    return signal

# STEP 2
def mask_hot_dead(signal: cp.ndarray, dead: cp.ndarray, dark: cp.ndarray, sigma: float = 5.0, maxiters: int = 5, dynamic_clip: bool = True) -> cp.ndarray:

    hot = sigma_clip_gpu(dark, sigma, maxiters)
    hot = cp.tile(hot, (signal.shape[0], 1, 1))
    dead = cp.tile(dead, (signal.shape[0], 1, 1))
    # print(hot.shape)
    print(dead.shape) 
    print(signal.shape)
    signal = cp.where(dead, cp.nan, signal)  # Set dead pixels to NaN
    signal = cp.where(hot, cp.nan, signal)  # Set hot pixels to NaN
    return signal


# STEP 3
def apply_linear_corr(linear_corr: cp.ndarray, clean_signal: cp.ndarray) -> cp.ndarray:
    '''
    Applies non-linearity correction using CuPy.
    This replaces the slow Python/Numpy loop with vectorized CuPy operations.
    Assumes linear_corr is (1, coeffs, y, x) and clean_signal is (time, y, x).
    '''



    return corrected_signal


# STEP 4
def clean_dark(signal: cp.ndarray, dead: cp.ndarray, dark: cp.ndarray, dt: cp.ndarray) -> cp.ndarray:
    '''
    Cleans dark current from the signal on GPU.
    Uses CuPy for all operations.
    '''
    # Convert dead to boolean mask if it's not already (e.g., 0/1 values)
    if dead.dtype != cp.bool_:
        dead_mask = (dead != 0)
    else:
        dead_mask = dead

    # Apply dead mask to dark frame, set masked values to NaN
    dark_masked = dark.copy().astype(cp.float64)
    dark_masked[dead_mask] = cp.nan
    return signal

# STEP 5
def get_cds(signal: cp.ndarray) -> cp.ndarray:
    '''
    Performs Correlated Double Sampling (CDS) on GPU.
    Assumes signal is (batch, time, y, x).
    '''
    # Check if signal has enough frames for CDS
    if signal.shape[1] % 2 != 0:
        # Handle odd number of frames, e.g., by dropping the last frame or raising error
        # For this example, we'll assume even number of frames or trim.
        # If `signal[:,1::2,:,:]` and `signal[:,::2,:,:]` are not same length, CuPy errors.
        print("Warning: Odd number of frames in signal for CDS. Dropping last frame.")
        signal = signal[:, :-(signal.shape[1] % 2), :, :] # Trim to even number of frames
        
    cds = signal[:, 1::2, :, :] - signal[:, ::2, :, :]
    return cds

# STEP 6 (OPTIONAL)
def bin_obs(cds_signal: cp.ndarray, binning: int) -> cp.ndarray:
    '''
    Time series observations are binned together at specified frequency on GPU.
    Assumes cds_signal is (batch, time, y, x).
    '''
    # Transpose is (0,1,3,2) in original for (batch, time, x, y)
    # The original comment said (0,1,3,2) for (batch,time,spectrum,32) if y was 32 and x was spectrum len
    # Given the input shape for AIRS is (time, 32, SPECTRAL_LEN) after read,
    # and then get_cds gives (batch, time, 32, SPECTRAL_LEN),
    # the original transpose (0,1,3,2) would swap last two axes.
    # So, (batch, time, y, x) -> (batch, time, x, y)
    cds_transposed = cds_signal.transpose(0, 1, 3, 2)

    # Calculate the number of binned frames
    num_binned_frames = cds_transposed.shape[1] // binning
    
    # Ensure signal length is a multiple of binning, or handle remainders
    # We will truncate for simplicity, as in the numpy version.
    trimmed_signal = cds_transposed[:, :num_binned_frames * binning, :, :]

    # Reshape for sum-binning: (batch, num_binned_frames, binning, x, y)
    binned_shape = (trimmed_signal.shape[0], num_binned_frames, binning, trimmed_signal.shape[2], trimmed_signal.shape[3])
    
    # Perform the sum over the binning dimension
    cds_binned = trimmed_signal.reshape(binned_shape).sum(axis=2)
    
    return cds_binned

# STEP 7
def correct_flat_field(flat: cp.ndarray, dead: cp.ndarray, signal: cp.ndarray) -> cp.ndarray:
    '''
    Applies flat field correction on GPU.
    Uses CuPy for all operations.
    '''
    # Original: flat = flat.transpose(1, 0)
    # flat is (Y, X)
    # The transpose is if flat comes in as (X, Y) but needs to be (Y, X) for consistent operations.
    # Assuming flat comes in as (Y, X) already (matching image dimensions), we might not need this.
    # If it consistently comes as (356, 32) for AIRS, then transpose to (32, 356) for (Y, X) is needed.
    # For FGS1, it's 32x32, so transpose (32,32) would still be (32,32).
    
    # Let's assume `flat` and `dead` come in with the correct (Y, X) orientation.
    # If `flat` is (32, 356) for AIRS and `dead` is also (32, 356).
    # If the transpose in the original code implies a necessary reorientation:
    # flat_reoriented = flat.transpose(1, 0) # if flat is (X,Y) and needs to be (Y,X)
    # dead_reoriented = dead.transpose(1, 0) # if dead is (X,Y) and needs to be (Y,X)
    # For now, let's assume they are correctly oriented or the transpose is handled implicitly
    # by how data is loaded/passed to this function.

    # Convert dead to boolean mask if it's not already (e.g., 0/1 values)
    if dead.dtype != cp.bool_:
        dead_mask = (dead != 0)
    else:
        dead_mask = dead
    
    # Apply dead mask to flat field, set masked values to NaN
    flat_masked = flat.copy().astype(cp.float64)
    flat_masked[dead_mask] = cp.nan

    # Broadcast flat_masked across batch and time dimensions for division
    # signal is (batch, time, Y, X)
    # flat_masked is (Y, X)
    # Use cp.newaxis to add batch and time dimensions to flat_masked
    signal = signal / flat_masked[cp.newaxis, cp.newaxis, :, :]
    
    return signal