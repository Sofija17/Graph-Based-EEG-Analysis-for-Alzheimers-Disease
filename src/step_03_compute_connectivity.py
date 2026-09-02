"""
Пресметка на functional connectivity со Pearson correlation меѓу
парови на EEG канали за секоја епоха посебно.

Резултат: за секоја епоха, матрица (n_channels x n_channels) каде
секоја вредност [i, j] покажува колку канал i и канал j се
"синхронизирани" во тој временски прозорец.

Правиме ребра меѓу јазлите
"""
import numpy as np
from scipy.signal import coherence
from scipy.stats import rankdata

#TODO посебна пресметка според фреквенциски опсег (алфа, бета, гама, делта) / според друга метрика за корелација (PLV)
SUPPORTED_CONNECTIVITY_METHODS = ("pearson", "spearman", "coherence")


def compute_connectivity_matrix(
    epoch_data,
    method="pearson",
    sfreq=None,
    fmin=0.5,
    fmax=30.0,
):
    """
    Пресметува Pearson correlation матрица за ЕДНА епоха.

    Параметри
    ---------
    epoch_data : np.ndarray, облик (n_channels, n_times)
                 сурови временски вредности за сите канали, ЕДНА епоха
                 (пр. epochs.get_data()[0] -> облик (19, 2000))

    Враќа
    -----
    np.ndarray, облик (n_channels, n_channels)
                симетрична матрица, вредности меѓу -1 и +1,
                дијагонала = 1.0
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
    Пресметува connectivity матрица за СИТЕ епохи одеднаш.

    Параметри
    ---------
    epochs_data : np.ndarray, облик (n_epochs, n_channels, n_times)
                  пр. epochs.get_data() -> (149, 19, 2000)

    Враќа
    -----
    np.ndarray, облик (n_epochs, n_channels, n_channels)
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
    Ги задржува само најсилните врски во connectivity матрицата,
    ги става на 0 сите послаби (спарс граф - помалку, но посилни edges).

    Параметри
    ---------
    conn_matrix : np.ndarray, облик (n_channels, n_channels)
    top_k_percent : float, пр. 0.3 = задржи топ 30% најсилни врски

    Враќа
    -----
    np.ndarray, ист облик, но со послабите врски поставени на 0
    """
    n_channels = conn_matrix.shape[0]

    # Работиме со апсолутна вредност бидејќи и силна негативна
    # корелација е "силна врска" (само во спротивна насока)
    abs_matrix = np.abs(conn_matrix.copy())

    # Ја игнорираме дијагоналата (секој канал со себе = секогаш 1.0,
    # не носи корисна информација за connectivity)
    np.fill_diagonal(abs_matrix, 0)

    # Ги земаме само вредностите над дијагоналата (горниот триаголник)
    # бидејќи матрицата е симетрична
    upper_indices = np.triu_indices(n_channels, k=1)
    upper_values = abs_matrix[upper_indices]

    # Одредуваме праг: вредноста под која се отфрлаат врските
    n_keep = int(len(upper_values) * top_k_percent)
    threshold = np.sort(upper_values)[::-1][n_keep - 1] if n_keep > 0 else 1.0

    # Правиме нова матрица, задржувајќи ги само вредностите над прагот.
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

    # Connectivity за само првата епоха
    conn_single = compute_connectivity_matrix(epochs_data[0])
    print("Connectivity матрица (епоха 0), облик:", conn_single.shape)
    print("Пример - Fp1 corr со сите канали:\n", conn_single[0])

    # Connectivity за сите епохи
    conn_all = compute_connectivity_all_epochs(epochs_data)
    print("\nВкупна connectivity матрица, облик:", conn_all.shape)

    # Threshold (keep top 30%)
    thresholded = threshold_connectivity(conn_single, top_k_percent=0.3)
    n_nonzero = np.count_nonzero(thresholded) - 19  # минус дијагоналата
    print(f"\nПо threshold-ирање, останати не-нула врски: {n_nonzero}")
