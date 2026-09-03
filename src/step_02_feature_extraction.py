"""
Compute qEEG features (relative band power: delta, theta, alpha, beta)
for each epoch and each channel.

Logic:
  1. Compute PSD (Power Spectral Density) for each epoch/channel
     -> obtain power values across fine frequency bins.
  2. Aggregate the fine bins into 4 broader frequency bands
     -> obtain 4 features (delta/theta/alpha/beta) per channel per epoch.

     Take raw time-domain values
     -> transform them into the frequency domain (PSD)
     -> group frequencies into 4 clinically meaningful bands
     -> place them into one tensor
     -> convert them into percentages (relative power)
     -> obtain the final node features for the graph.
"""

import numpy as np
import config


def compute_psd(epochs, fmin=0.5, fmax=30.0):
    """
    Compute Power Spectral Density for each epoch and channel.

    Returns
    -------
    psd_data : np.ndarray, shape (n_epochs, n_channels, n_freqs)
    freqs : np.ndarray, shape (n_freqs,) - frequencies corresponding
            to the last dimension of psd_data
    """
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)
    return psd_data, freqs


def compute_band_power(psd_data, freqs, band_range):
    """
    Aggregate power within a selected frequency band.

    Parameters
    ----------
    psd_data : np.ndarray, shape (n_epochs, n_channels, n_freqs)
    freqs : np.ndarray, shape (n_freqs,)
    band_range : tuple (fmin, fmax), e.g. (0.5, 4.0) for delta

    Returns
    -------
    np.ndarray, shape (n_epochs, n_channels)
    """
    fmin, fmax = band_range
    freq_mask = (freqs >= fmin) & (freqs < fmax)
    band_power = psd_data[:, :, freq_mask].mean(axis=-1)
    return band_power


def extract_band_powers(epochs, bands=None):
    """
    Main function: convert an Epochs object into a dictionary with
    band power for each band (delta/theta/alpha/beta).

    Parameters
    ----------
    epochs : mne.Epochs
    bands : dict, e.g. {"delta": (0.5, 4.0), ...} (default: config.FREQ_BANDS)

    Returns
    -------
    dict: {"delta": np.ndarray(n_epochs, n_channels), "theta": ..., ...}
    """
    bands = bands or config.FREQ_BANDS

    psd_data, freqs = compute_psd(epochs, fmin=0.5, fmax=30.0)

    band_powers = {}
    for band_name, band_range in bands.items():
        band_powers[band_name] = compute_band_power(psd_data, freqs, band_range)

    return band_powers


def band_powers_to_feature_matrix(band_powers, bands=None):
    """
    Convert the band_powers dictionary into a tensor suitable
    for graph node features.

    Parameters
    ----------
    band_powers : dict, result from extract_band_powers()
    bands : list of band names, defines the feature order
            (default: config.FREQ_BANDS keys - delta, theta, alpha, beta)

    Returns
    -------
    np.ndarray, shape (n_epochs, n_channels, n_bands)
    """
    band_names = list(bands.keys()) if bands else list(config.FREQ_BANDS.keys())
    feature_matrix = np.stack([band_powers[b] for b in band_names], axis=-1)
    return feature_matrix


def compute_relative_power(feature_matrix):
    """
    Convert absolute power into relative power
    (each band as a percentage of total power for that channel/epoch).

    This matters because absolute power values can vary strongly between
    subjects due to electrode impedance, skull thickness, and similar factors.
    Relative power is a more robust comparative measure.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (n_epochs, n_channels, n_bands)

    Returns
    -------
    np.ndarray, same shape, but each row sums to 1.0 (100%)
    """
    total_power = feature_matrix.sum(axis=-1, keepdims=True)
    relative = feature_matrix / total_power
    return relative


if __name__ == "__main__":
    # Test
    from step_01_preprocessing import load_and_epoch_subject

    test_subject = "sub-001"
    epochs = load_and_epoch_subject(test_subject)

    band_powers = extract_band_powers(epochs)
    for band_name, arr in band_powers.items():
        print(f"{band_name}: shape = {arr.shape}")

    feature_matrix = band_powers_to_feature_matrix(band_powers)
    print(f"\nFeature matrix shape: {feature_matrix.shape}")

    relative_matrix = compute_relative_power(feature_matrix)
    print(f"Пример - епоха 0, канал 0 (relative power): {relative_matrix[0, 0]}")
    # The sum should be ~1.0
    print(f"Сума : {relative_matrix[0, 0].sum()}")
