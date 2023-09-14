# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

def _compute_variance_features(window):
    """
    Computes the variance x, y and z acceleration over the given window. 
    """
    return np.var(window, axis=0)

def _compute_median_features(window):
    """
    Computes the median x, y and z acceleration over the given window. 
    """
    return np.median(window, axis=0)

def _compute_magnitude_features(window):
    """
    Computes the magnitude x, y and z acceleration over the given window. 
    """    
    return np.linalg.norm(window, axis=0)

def _compute_max_features(window):
    """
    Computes the max x, y and z acceleration over the given window. 
    """
    return np.argmax(window, axis=0)

def _compute_min_features(window):
    """
    Computes the min x, y and z acceleration over the given window. 
    """
    return np.argmin(window, axis=0)

def _compute_fft_features(window):
    """
    Computes the FFT x, y and z acceleration over the given window. 
    """
    win = np.array(window)
    fft = np.abs(np.fft.rfft(win.astype(float), axis=0))
    return np.sum(fft, axis=0)

def _compute_entropy_features(window):
    """
    Computes the entropy x, y and z acceleration over the given window. 
    """
    return entropy(window)

def _compute_peakcount_features(window):
    """
    Computes the number of peaks of x, y and z acceleration over the given window. 
    """
    peaks, _ = find_peaks(window)
    
    return len(peaks)


# define functions to compute more features

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    """
    Statistical
    These include the mean, variance and the rate of zero- or mean-crossings. The
    minimum and maximum may be useful, as might the median
    
    FFT features
    use rfft() to get Discrete Fourier Transform
    
    Entropy
    Integrating acceleration
    
    Peak Features:
    Sometimes the count or location of peaks or troughs in the accelerometer signal can be
    an indicator of the type of activity being performed. This is basically what you did in
    assignment A1 to detect steps. Use the peak count over each window as a feature. Or
    try something like the average duration between peaks in a window.
    """

    
    x = []
    feature_names = []
    win = np.array(window)
    
    # Mean
    x.append(_compute_mean_features(win[:,0]))
    x.append(_compute_mean_features(win[:,1]))
    x.append(_compute_mean_features(win[:,2]))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")

    # call functions to compute other features. Append the features to x and the names of these features to feature_names
    
    # Variance
    x.append(_compute_variance_features(win[:,0]))
    x.append(_compute_variance_features(win[:,1]))
    x.append(_compute_variance_features(win[:,2]))
    feature_names.append("x_var")
    feature_names.append("y_var")
    feature_names.append("z_var")
    
    # Median
    x.append(_compute_median_features(win[:,0]))
    x.append(_compute_median_features(win[:,1]))
    x.append(_compute_median_features(win[:,2]))
    feature_names.append("x_med")
    feature_names.append("y_med")
    feature_names.append("z_med")
    
    # Magnitude
    x.append(_compute_magnitude_features(win[:,0]))
    x.append(_compute_magnitude_features(win[:,1]))
    x.append(_compute_magnitude_features(win[:,2]))
    feature_names.append("x_mag")
    feature_names.append("y_mag")
    feature_names.append("z_mag")
    
    # Max
    x.append(_compute_max_features(win[:,0]))
    x.append(_compute_max_features(win[:,1]))
    x.append(_compute_max_features(win[:,2]))
    feature_names.append("x_max")
    feature_names.append("y_max")
    feature_names.append("z_max")
    
    # Min
    x.append(_compute_min_features(win[:,0]))
    x.append(_compute_min_features(win[:,1]))
    x.append(_compute_min_features(win[:,2]))
    feature_names.append("x_min")
    feature_names.append("y_min")
    feature_names.append("z_min")
    
    # FFT
    fft_x = win[:,0] 
    fft_y = win[:,1] 
    fft_z = win[:,2] 
    x.append(_compute_fft_features(np.sqrt(fft_x**2 + fft_y**2 + fft_z**2)))
    feature_names.append("mag_fft")
    
    # Entropy
    ent_x = win[:,0] 
    ent_y = win[:,1] 
    ent_z = win[:,2] 
    x.append(_compute_entropy_features(np.sqrt(ent_x**2 + ent_y**2 + ent_z**2)))
    feature_names.append("mag_ent")
    
    # Peak Count    
    x.append(_compute_peakcount_features(win[:,0]))
    x.append(_compute_peakcount_features(win[:,1]))
    x.append(_compute_peakcount_features(win[:,2]))
    feature_names.append("x_peakcount")
    feature_names.append("y_peakcount")
    feature_names.append("z_peakcount")

    feature_vector = list(x)
    return feature_names, feature_vector