# pylint: disable-all
"""
CuPy-based sigma clipping implementation optimized for CUDA acceleration.
Based on astropy.stats.sigma_clipping but using CuPy for GPU computation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Literal, Union

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from cupy.typing import NDArray as CupyArray

__all__ = ["SigmaClipGPU", "sigma_clip_gpu", "sigma_clipped_stats_gpu"]


def _nanmedian_gpu(data: cp.ndarray, axis=None) -> cp.ndarray:
    """CuPy-compatible nanmedian function."""
    if axis is None:
        valid_data = data[~cp.isnan(data)]
        if valid_data.size == 0:
            return cp.nan
        return cp.median(valid_data)
    else:
        return cp.nanmedian(data, axis=axis)


def _nanmean_gpu(data: cp.ndarray, axis=None) -> cp.ndarray:
    """CuPy-compatible nanmean function."""
    return cp.nanmean(data, axis=axis)


def _nanstd_gpu(data: cp.ndarray, axis=None, ddof=0) -> cp.ndarray:
    """CuPy-compatible nanstd function."""
    return cp.nanstd(data, axis=axis, ddof=ddof)


def _mad_std_gpu(data: cp.ndarray, axis=None) -> cp.ndarray:
    """
    Calculate the median absolute deviation (MAD) based standard deviation.
    """
    if axis is None:
        valid_data = data[~cp.isnan(data)]
        if valid_data.size == 0:
            return cp.nan
        median = cp.median(valid_data)
        mad = cp.median(cp.abs(valid_data - median))
        return mad * 1.4826  # Factor to make MAD consistent with std for normal distribution
    else:
        median = _nanmedian_gpu(data, axis=axis)
        # Expand dimensions for broadcasting
        if not cp.isscalar(median):
            shape = list(data.shape)
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                shape[ax] = 1
            median = median.reshape(shape)
        
        mad = _nanmedian_gpu(cp.abs(data - median), axis=axis)
        return mad * 1.4826


class SigmaClipGPU:
    """
    GPU-accelerated sigma clipping class using CuPy.
    
    The data will be iterated over, each time rejecting values that are
    less or more than a specified number of standard deviations from a
    center value.
    
    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower
        and upper clipping limit. Default is 3.0.
    sigma_lower : float or None, optional
        The number of standard deviations to use as the lower bound.
        If None, uses sigma. Default is None.
    sigma_upper : float or None, optional
        The number of standard deviations to use as the upper bound.
        If None, uses sigma. Default is None.
    maxiters : int or None, optional
        Maximum number of iterations. If None, iterate until convergence.
        Default is 5.
    cenfunc : {'median', 'mean'} or callable, optional
        Function to compute center value. Default is 'median'.
    stdfunc : {'std', 'mad_std'} or callable, optional
        Function to compute standard deviation. Default is 'std'.
    """
    
    def __init__(
        self,
        sigma: float = 3.0,
        sigma_lower: float | None = None,
        sigma_upper: float | None = None,
        maxiters: int | None = 5,
        cenfunc: Literal["median", "mean"] | Callable = "median",
        stdfunc: Literal["std", "mad_std"] | Callable = "std",
    ) -> None:
        self.sigma = sigma
        self.sigma_lower = sigma_lower or sigma
        self.sigma_upper = sigma_upper or sigma
        self.maxiters = maxiters or cp.inf
        self.cenfunc = cenfunc
        self.stdfunc = stdfunc
        self._cenfunc_parsed = self._parse_cenfunc(cenfunc)
        self._stdfunc_parsed = self._parse_stdfunc(stdfunc)
        self._min_value = cp.nan
        self._max_value = cp.nan
        self._niterations = 0

    def __repr__(self) -> str:
        return (
            f"SigmaClipGPU(sigma={self.sigma}, sigma_lower={self.sigma_lower}, "
            f"sigma_upper={self.sigma_upper}, maxiters={self.maxiters}, "
            f"cenfunc={self.cenfunc!r}, stdfunc={self.stdfunc!r})"
        )

    @staticmethod
    def _parse_cenfunc(cenfunc) -> Callable:
        """Parse the center function."""
        if isinstance(cenfunc, str):
            if cenfunc == "median":
                return _nanmedian_gpu
            elif cenfunc == "mean":
                return _nanmean_gpu
            else:
                raise ValueError(f"{cenfunc} is an invalid cenfunc.")
        return cenfunc

    @staticmethod
    def _parse_stdfunc(stdfunc) -> Callable:
        """Parse the standard deviation function."""
        if isinstance(stdfunc, str):
            if stdfunc == "std":
                return _nanstd_gpu
            elif stdfunc == "mad_std":
                return _mad_std_gpu
            else:
                raise ValueError(f"{stdfunc} is an invalid stdfunc.")
        return stdfunc

    def _compute_bounds(
        self,
        data: cp.ndarray,
        axis: int | tuple[int, ...] | None = None,
    ) -> None:
        """Compute clipping bounds."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cen = self._cenfunc_parsed(data, axis=axis)
            std = self._stdfunc_parsed(data, axis=axis)
            self._min_value = cen - (std * self.sigma_lower)
            self._max_value = cen + (std * self.sigma_upper)

    def _sigmaclip_noaxis(
        self,
        data: cp.ndarray,
        masked: bool = True,
        return_bounds: bool = False,
        copy: bool = True,
    ) -> cp.ndarray | tuple[cp.ndarray, float, float]:
        """
        Sigma clip when axis is None.
        """
        if copy:
            filtered_data = data.copy().ravel()
        else:
            filtered_data = data.ravel()

        # Remove invalid values
        good_mask = cp.isfinite(filtered_data)
        if cp.any(~good_mask):
            filtered_data = filtered_data[good_mask]

        nchanged = 1
        iteration = 0
        while nchanged != 0 and (iteration < self.maxiters):
            iteration += 1
            size = filtered_data.size
            self._compute_bounds(filtered_data, axis=None)
            
            valid_mask = (filtered_data >= self._min_value) & (filtered_data <= self._max_value)
            filtered_data = filtered_data[valid_mask]
            nchanged = size - filtered_data.size

        self._niterations = iteration

        if masked:
            # Create masked array equivalent
            result = cp.copy(data) if copy else data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mask = cp.logical_or(data < self._min_value, data > self._max_value)
            result[mask] = cp.nan
        else:
            result = filtered_data

        if return_bounds:
            return result, float(self._min_value), float(self._max_value)
        else:
            return result

    def _sigmaclip_withaxis(
        self,
        data: cp.ndarray,
        axis: int | tuple[int, ...] | None = None,
        masked: bool = True,
        return_bounds: bool = False,
        copy: bool = True,
    ) -> cp.ndarray | tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Sigma clip with specified axis.
        """
        if data.dtype.kind != "f":
            filtered_data = data.astype(cp.float32)
        else:
            filtered_data = data.copy() if copy else data

        # Handle invalid values
        bad_mask = ~cp.isfinite(filtered_data)
        if cp.any(bad_mask):
            filtered_data[bad_mask] = cp.nan

        if axis is not None:
            # Convert negative axis to positive
            if not cp.iterable(axis):
                axis = (axis,)
            axis = tuple(filtered_data.ndim + n if n < 0 else n for n in axis)

            # Define shape for broadcasting
            mshape = tuple(
                1 if dim in axis else size
                for dim, size in enumerate(filtered_data.shape)
            )

        nchanged = 1
        iteration = 0
        while nchanged != 0 and (iteration < self.maxiters):
            iteration += 1
            self._compute_bounds(filtered_data, axis=axis)
            
            if not cp.isscalar(self._min_value):
                self._min_value = self._min_value.reshape(mshape)
                self._max_value = self._max_value.reshape(mshape)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                new_mask = (filtered_data < self._min_value) | (filtered_data > self._max_value)
            
            filtered_data[new_mask] = cp.nan
            nchanged = cp.count_nonzero(new_mask)

        self._niterations = iteration

        if masked:
            if not copy:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)    
                    mask = cp.logical_or(data < self._min_value, data > self._max_value)
                result = cp.copy(data)
                result[mask] = cp.nan
            else:
                result = filtered_data
        else:
            result = filtered_data

        if return_bounds:
            return result, self._min_value, self._max_value
        else:
            return result

    def __call__(
        self,
        data: cp.ndarray,
        axis: int | tuple[int, ...] | None = None,
        masked: bool = True,
        return_bounds: bool = False,
        copy: bool = True,
    ) -> cp.ndarray | tuple[cp.ndarray, ...]:
        """
        Perform sigma clipping on the provided data.
        
        Parameters
        ----------
        data : cp.ndarray
            The data to be sigma clipped.
        axis : None or int or tuple of int, optional
            The axis or axes along which to sigma clip.
        masked : bool, optional
            If True, return array with NaN for clipped values.
        return_bounds : bool, optional
            If True, also return clipping bounds.
        copy : bool, optional
            If True, copy the input data.
            
        Returns
        -------
        result : cp.ndarray or tuple
            Sigma-clipped data and optionally the bounds.
        """
        data = cp.asarray(data)

        if data.size == 0:
            if return_bounds:
                return data, self._min_value, self._max_value
            else:
                return data

        if axis is None:
            return self._sigmaclip_noaxis(
                data, masked=masked, return_bounds=return_bounds, copy=copy
            )
        else:
            return self._sigmaclip_withaxis(
                data, axis=axis, masked=masked, return_bounds=return_bounds, copy=copy
            )


def sigma_clip_gpu(
    data: cp.ndarray,
    sigma: float = 3.0,
    sigma_lower: float | None = None,
    sigma_upper: float | None = None,
    maxiters: int | None = 5,
    cenfunc: Literal["median", "mean"] | Callable = "median",
    stdfunc: Literal["std", "mad_std"] | Callable = "std",
    axis: int | tuple[int, ...] | None = None,
    masked: bool = True,
    return_bounds: bool = False,
    copy: bool = True,
) -> cp.ndarray | tuple[cp.ndarray, ...]:
    """
    Perform sigma-clipping on CuPy arrays with GPU acceleration.
    
    Parameters
    ----------
    data : cp.ndarray
        The data to be sigma clipped.
    sigma : float, optional
        Number of standard deviations for clipping. Default is 3.0.
    sigma_lower : float or None, optional
        Lower bound standard deviations. Default is None (uses sigma).
    sigma_upper : float or None, optional
        Upper bound standard deviations. Default is None (uses sigma).
    maxiters : int or None, optional
        Maximum iterations. Default is 5.
    cenfunc : {'median', 'mean'} or callable, optional
        Center function. Default is 'median'.
    stdfunc : {'std', 'mad_std'} or callable, optional
        Standard deviation function. Default is 'std'.
    axis : None or int or tuple of int, optional
        Axis along which to clip. Default is None.
    masked : bool, optional
        Return masked array (NaN for clipped). Default is True.
    return_bounds : bool, optional
        Return clipping bounds. Default is False.
    copy : bool, optional
        Copy input data. Default is True.
        
    Returns
    -------
    result : cp.ndarray or tuple
        Sigma-clipped data and optionally bounds.
    """
    sigclip = SigmaClipGPU(
        sigma=sigma,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=maxiters,
        cenfunc=cenfunc,
        stdfunc=stdfunc,
    )

    return sigclip(
        data, axis=axis, masked=masked, return_bounds=return_bounds, copy=copy
    )


class SigmaClippedStatsGPU:
    """
    GPU-accelerated sigma-clipped statistics using CuPy.
    
    Parameters
    ----------
    data : cp.ndarray
        Input data array.
    sigma : float, optional
        Sigma clipping parameter. Default is 3.0.
    sigma_lower : float or None, optional
        Lower sigma bound. Default is None.
    sigma_upper : float or None, optional
        Upper sigma bound. Default is None.
    maxiters : int, optional
        Maximum iterations. Default is 5.
    cenfunc : {'median', 'mean'} or callable, optional
        Center function. Default is 'median'.
    stdfunc : {'std', 'mad_std'} or callable, optional
        Standard deviation function. Default is 'std'.
    axis : None or int or tuple of int, optional
        Computation axis. Default is None.
    """
    
    def __init__(
        self,
        data: cp.ndarray,
        sigma: float = 3.0,
        sigma_lower: float | None = None,
        sigma_upper: float | None = None,
        maxiters: int = 5,
        cenfunc: Literal["median", "mean"] | Callable = "median",
        stdfunc: Literal["std", "mad_std"] | Callable = "std",
        axis: int | tuple[int, ...] | None = None,
    ) -> None:
        sigclip = SigmaClipGPU(
            sigma=sigma,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            maxiters=maxiters,
            cenfunc=cenfunc,
            stdfunc=stdfunc,
        )

        self.data = sigclip(
            data, axis=axis, masked=True, return_bounds=False, copy=True
        )
        self.axis = axis

    def min(self) -> float | cp.ndarray:
        """Calculate minimum of clipped data."""
        return cp.nanmin(self.data, axis=self.axis)

    def max(self) -> float | cp.ndarray:
        """Calculate maximum of clipped data."""
        return cp.nanmax(self.data, axis=self.axis)

    def sum(self) -> float | cp.ndarray:
        """Calculate sum of clipped data."""
        return cp.nansum(self.data, axis=self.axis)

    def mean(self) -> float | cp.ndarray:
        """Calculate mean of clipped data."""
        return cp.nanmean(self.data, axis=self.axis)

    def median(self) -> float | cp.ndarray:
        """Calculate median of clipped data."""
        return _nanmedian_gpu(self.data, axis=self.axis)

    def std(self, ddof: int = 0) -> float | cp.ndarray:
        """Calculate standard deviation of clipped data."""
        return cp.nanstd(self.data, axis=self.axis, ddof=ddof)

    def var(self, ddof: int = 0) -> float | cp.ndarray:
        """Calculate variance of clipped data."""
        return cp.nanvar(self.data, axis=self.axis, ddof=ddof)

    def mad_std(self) -> float | cp.ndarray:
        """Calculate MAD-based standard deviation of clipped data."""
        return _mad_std_gpu(self.data, axis=self.axis)


def sigma_clipped_stats_gpu(
    data: cp.ndarray,
    sigma: float = 3.0,
    sigma_lower: float | None = None,
    sigma_upper: float | None = None,
    maxiters: int | None = 5,
    cenfunc: Literal["median", "mean"] | Callable = "median",
    stdfunc: Literal["std", "mad_std"] | Callable = "std",
    std_ddof: int = 0,
    axis: int | tuple[int, ...] | None = None,
) -> tuple[float | cp.ndarray, float | cp.ndarray, float | cp.ndarray]:
    """
    Calculate sigma-clipped statistics on CuPy arrays.
    
    Parameters
    ----------
    data : cp.ndarray
        Input data.
    sigma : float, optional
        Sigma clipping parameter.
    sigma_lower : float or None, optional
        Lower sigma bound.
    sigma_upper : float or None, optional
        Upper sigma bound.
    maxiters : int or None, optional
        Maximum iterations.
    cenfunc : {'median', 'mean'} or callable, optional
        Center function.
    stdfunc : {'std', 'mad_std'} or callable, optional
        Standard deviation function.
    std_ddof : int, optional
        Delta degrees of freedom for std calculation.
    axis : None or int or tuple of int, optional
        Computation axis.
        
    Returns
    -------
    mean, median, stddev : tuple
        Mean, median, and standard deviation of sigma-clipped data.
    """
    stats = SigmaClippedStatsGPU(
        data,
        sigma=sigma,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=maxiters,
        cenfunc=cenfunc,
        stdfunc=stdfunc,
        axis=axis,
    )

    return stats.mean(), stats.median(), stats.std(ddof=std_ddof)