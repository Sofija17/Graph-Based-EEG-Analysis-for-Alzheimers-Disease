import matplotlib.pyplot as plt
import mne

# Path to the derivatives version of subject 1
subject_id = "sub-001"
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
eeg_path = PROJECT_ROOT / "data" / "raw" / "ds004504" / "derivatives" / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"

raw = mne.io.read_raw_eeglab(eeg_path, preload=True)

print(raw.info)
print("Sampling frequency:", raw.info['sfreq']) # =500 hertz, number of measurements per second
print("Времетраење (сек):", raw.n_times / raw.info['sfreq'])
print("Веќе применети филтри:", raw.info['highpass'], "-", raw.info['lowpass'], "Hz")

# Visual inspection (check that the signal looks clean)
raw.plot(duration=10, n_channels=19, scalings='auto')
plt.show()

# PSD (check that there is no remaining 50Hz noise)
raw.compute_psd(fmax=60).plot()
plt.show()

# --- Epoching: segmenting into 4-second windows ---
# Take the continuous signal (10 minutes = 300,000 time points per channel)
# and cut it into exact 4-second pieces (2000 points), one after another, without overlap (overlap=0.0).
# Each piece becomes one independent epoch.
# Each epoch will later become one graph that enters the GCN.
epochs = mne.make_fixed_length_epochs(raw, duration=4.0, overlap=0.0, preload=True)

print("Број на епохи/сегменти:", len(epochs))
print("Облик на податоците:", epochs.get_data().shape)

# --- Define frequency bands ---
#TODO move this to config.py
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

# --- Compute PSD for each segment and each channel ---
# This returns power at each frequency for each epoch and channel.
psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=30.0, verbose=False)
psd_data, freqs = psd.get_data(return_freqs=True)

print("Облик на PSD податоците:", psd_data.shape)
print("Фреквенции (Hz):", freqs)
