"""
Compute functional connectivity with Pearson correlation between
pairs of EEG channels for each epoch separately.

Result: for each epoch, a matrix (n_channels x n_channels) where
each value [i, j] shows how synchronized channel i and channel j are
within that time window.

These connectivity values are later used to create edges between nodes.
"""
import numpy as np
from scipy.signal import coherence
from scipy.stats import rankdata

# TODO compute connectivity separately by frequency band (alpha, beta, gamma, delta) / use another connectivity metric (PLV)
SUPPORTED_CONNECTIVITY_METHODS = ("pearson", "spearman", "coherence")


def compute_connectivity_matrix(
    epoch_data,
    method="pearson",
    sfreq=None,
    fmin=0.5,
    fmax=30.0,
):
    """
    Compute a connectivity matrix for ONE epoch.

    Parameters
    ----------
    epoch_data : np.ndarray, shape (n_channels, n_times)
                 raw time-domain values for all channels, ONE epoch
                 (e.g. epochs.get_data()[0] -> shape (19, 2000))

    Returns
    -------
    np.ndarray, shape (n_channels, n_channels)
                symmetric matrix, values between -1 and +1,
                diagonal = 1.0
    """
    method = method.lower()
    if method not in SUPPORTED_CONNECTIVITY_METHODS:
        raise ValueError(
            f"Unsupported connectivity method '{method}'. "
            f"Choose from {SUPPORTED_CONNECTIVITY_METHODS}."
        )

    if method == "pearson":
        return np.corrcoef(epoch_data)

    if method == "spearman":
        ranked = np.apply_along_axis(rankdata, 1, epoch_data)
        return np.corrcoef(ranked)

    if sfreq is None:
        raise ValueError("sfreq is required for coherence connectivity")

    _, n_times = epoch_data.shape
    nperseg = min(n_times, max(8, int(round(2.0 * sfreq))))
    # Broadcasting computes every channel pair in one vectorized Welch call:
    # (channels, 1, times) x (1, channels, times) -> (channels, channels, freqs).
    frequencies, values = coherence(
        epoch_data[:, np.newaxis, :],
        epoch_data[np.newaxis, :, :],
        fs=sfreq,
        nperseg=nperseg,
        axis=-1,
    )
    mask = (frequencies >= fmin) & (frequencies < fmax)
    if not np.any(mask):
        raise ValueError(f"No coherence frequency bins in [{fmin}, {fmax}) Hz")
    matrix = values[..., mask].mean(axis=-1)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    return matrix


def compute_connectivity_all_epochs(
    epochs_data,
    method="pearson",
    sfreq=None,
    fmin=0.5,
    fmax=30.0,
):
    """
    Compute connectivity matrices for ALL epochs at once.

    Parameters
    ----------
    epochs_data : np.ndarray, shape (n_epochs, n_channels, n_times)
                  e.g. epochs.get_data() -> (149, 19, 2000)

    Returns
    -------
    np.ndarray, shape (n_epochs, n_channels, n_channels)
    """
    n_epochs = epochs_data.shape[0]
    n_channels = epochs_data.shape[1]

    all_matrices = np.zeros((n_epochs, n_channels, n_channels))

    for i in range(n_epochs):
        all_matrices[i] = compute_connectivity_matrix(
            epochs_data[i], method=method, sfreq=sfreq, fmin=fmin, fmax=fmax
        )

    return all_matrices

def threshold_connectivity(conn_matrix, top_k_percent=0.3):
    """
    Keep only the strongest connections in the connectivity matrix,
    setting weaker ones to 0 (sparse graph - fewer but stronger edges).

    Parameters
    ----------
    conn_matrix : np.ndarray, shape (n_channels, n_channels)
    top_k_percent : float, e.g. 0.3 = keep the top 30% strongest connections

    Returns
    -------
    np.ndarray, same shape, but weaker connections are set to 0
    """
    n_channels = conn_matrix.shape[0]

    # Use absolute values because a strong negative correlation is still
    # a strong connection, only in the opposite direction.
    abs_matrix = np.abs(conn_matrix.copy())

    # Ignore the diagonal (each channel with itself = always 1.0,
    # which does not add useful connectivity information).
    np.fill_diagonal(abs_matrix, 0)

    # Use only values above the diagonal (upper triangle)
    # because the matrix is symmetric.
    upper_indices = np.triu_indices(n_channels, k=1)
    upper_values = abs_matrix[upper_indices]

    # Determine the cutoff: connections below this value are discarded.
    n_keep = int(len(upper_values) * top_k_percent)
    threshold = np.sort(upper_values)[::-1][n_keep - 1] if n_keep > 0 else 1.0

    # Create a new matrix, keeping only values above the cutoff.
    result = conn_matrix.copy()
    mask_to_zero = (abs_matrix < threshold)
    np.fill_diagonal(mask_to_zero, False)
    result[mask_to_zero] = 0

    return result


if __name__ == "__main__":
    # Teest
    from step_01_preprocessing import load_and_epoch_subject

    test_subject = "sub-001"
    epochs = load_and_epoch_subject(test_subject)
    epochs_data = epochs.get_data()  # (149, 19, 2000)

    print("Облик на epochs_data:", epochs_data.shape)

    # Connectivity for the first epoch only
    conn_single = compute_connectivity_matrix(epochs_data[0])
    print("Connectivity матрица (епоха 0), облик:", conn_single.shape)
    print("Пример - Fp1 corr со сите канали:\n", conn_single[0])

    # Connectivity for all epochs
    conn_all = compute_connectivity_all_epochs(epochs_data)
    print("\nВкупна connectivity матрица, облик:", conn_all.shape)

    # Threshold (keep top 30%)
    thresholded = threshold_connectivity(conn_single, top_k_percent=0.3)
    n_nonzero = np.count_nonzero(thresholded) - 19  # minus the diagonal
    print(f"\nПо threshold-ирање, останати не-нула врски: {n_nonzero}")
