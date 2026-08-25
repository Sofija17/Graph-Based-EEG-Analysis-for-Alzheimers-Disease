import matplotlib.pyplot as plt
import mne

# Патека до derivatives верзија на субјект 1
subject_id = "sub-001"
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
eeg_path = PROJECT_ROOT / "data" / "raw" / "ds004504" / "derivatives" / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"

raw = mne.io.read_raw_eeglab(eeg_path, preload=True)

print(raw.info)
print("Sampling frequency:", raw.info['sfreq']) # =500 херци, колку мерења во секунда, ново мерење на секои 2 секунди
print("Времетраење (сек):", raw.n_times / raw.info['sfreq'])
print("Веќе применети филтри:", raw.info['highpass'], "-", raw.info['lowpass'], "Hz")

# Визуелен преглед (проверка дека изгледа "чисто")
raw.plot(duration=10, n_channels=19, scalings='auto')
plt.show()

# PSD (проверка дека нема повеќе 50Hz шум)
raw.compute_psd(fmax=60).plot()
plt.show()

# --- Epoching: сегментирање во 4-секундни прозорци ---
# Го зема континуираниот сигнал (10 минути = 300,000 временски точки по канал)
# и го сече на парчиња од точно 4 секунди (2000 точки), едно по друго, без преклопување (overlap=0.0).
# Секое парче станува една независна епоха
# секоја епоха подоцна ќе стане еден граф што ќе влезе во GCN
epochs = mne.make_fixed_length_epochs(raw, duration=4.0, overlap=0.0, preload=True)

print("Број на епохи/сегменти:", len(epochs))
print("Облик на податоците:", epochs.get_data().shape)

# --- Дефинирање на frequency bands ---
#TODO move this to config.py
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

# --- Пресметка на PSD за секој сегмент, секој канал ---
# ова враќа моќност (power) на секоја фреквенција, за секоја епоха и канал
psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=30.0, verbose=False)
psd_data, freqs = psd.get_data(return_freqs=True)

print("Облик на PSD податоците:", psd_data.shape)
print("Фреквенции (Hz):", freqs)