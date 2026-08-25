"""
Пресметка на qEEG features (relative band power: delta, theta, alpha, beta)
за секоja епоха, за секој канал

Логика:
  1. Пресметај PSD (Power Spectral Density) за секоја епоха/канал
     -> добиваме моќност на многу фини фреквенции (bins)
  2. Собери ги фините bins во 4 пошироки frequency bands
     -> добиваме 4 features (delta/theta/alpha/beta) по канал по епоха

     Земаме сурови временски вредности
     → трансформираме во frequency приказ (PSD)
     → групираме фреквенции во 4 клинички смислени опсези
     → ги ставаме во еден тензор
     → конвертираме во проценти (релативна power)
     → добиваме финални node features за графот.
"""

import numpy as np
import config


def compute_psd(epochs, fmin=0.5, fmax=30.0):
    """
    Пресметува Power Spectral Density за секоја епоха и канал

    Враќа
    -----
    psd_data : np.ndarray, облик (n_epochs, n_channels, n_freqs)
    freqs : np.ndarray, облик (n_freqs,) - фреквенциите што одговараат
            на последната димензија на psd_data
    """
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)
    return psd_data, freqs


def compute_band_power(psd_data, freqs, band_range):
    """
    Собира моќност во определен frequency опсег

    Параметри
    ---------
    psd_data : np.ndarray, облик (n_epochs, n_channels, n_freqs)
    freqs : np.ndarray, облик (n_freqs,)
    band_range : tuple (fmin, fmax), пр. (0.5, 4.0) за delta

    Враќа
    -----
    np.ndarray, облик (n_epochs, n_channels)
    """
    fmin, fmax = band_range
    freq_mask = (freqs >= fmin) & (freqs < fmax)
    band_power = psd_data[:, :, freq_mask].mean(axis=-1)
    return band_power


def extract_band_powers(epochs, bands=None):
    """
    Главна функција: од Epochs објект до речник со band power за
    секој опсег (delta/theta/alpha/beta).

    Параметри
    ---------
    epochs : mne.Epochs
    bands : dict, пр. {"delta": (0.5, 4.0), ...} (default: config.FREQ_BANDS)

    Враќа
    -----
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
    Го конвертира речникот од band_powers во тензор,
    погоден за node features во граф

    Параметри
    ---------
    band_powers : dict, резултат од extract_band_powers()
    bands : list на имиња на опсези, го одредува редоследот на features
            (default: config.FREQ_BANDS клучеви - delta, theta, alpha, beta)

    Враќа
    -----
    np.ndarray, облик (n_epochs, n_channels, n_bands)
    """
    band_names = list(bands.keys()) if bands else list(config.FREQ_BANDS.keys())
    feature_matrix = np.stack([band_powers[b] for b in band_names], axis=-1)
    return feature_matrix


def compute_relative_power(feature_matrix):
    """
    Конвертира апсолутна моќност во РЕЛАТИВНА моќност
    (секој band како процент од вкупната моќност на тој канал/епоха).

    Ова е важно затоа што апсолутните вредности на моќност може да
    варираат многу меѓу субјекти (заради разлики во импеданса на
    електродите, дебелина на черепот, итн.) ,релативната моќност
    е поробусна споредбена мерка.

    Параметри
    ---------
    feature_matrix : np.ndarray, облик (n_epochs, n_channels, n_bands)

    Враќа
    -----
    np.ndarray, ист облик, но секој ред сумира до 1.0 (100%)
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
    #Sumata treba da e ~1.0
    print(f"Сума : {relative_matrix[0, 0].sum()}")