"""
Функции за вчитување на веќе-прочистени (derivatives/) EEG записи
и нивно сегментирање во кратки епохи (сегменти).

"""

import mne
import config

def load_subject_raw(subject_id, use_derivatives=True):
    """
    Вчитува еден EEG запис (непроцесиран).

    Параметри
    ---------
    subject_id : str
        пр. "sub-001"
    use_derivatives : bool
        True = користи derivatives/ (веќе прочистена верзија)
        False = користи raw/ (сурова верзија)

    Враќа
    -----
    mne.io.Raw објект
    """
    base = config.DATA_RAW / "derivatives" if use_derivatives else config.DATA_RAW

    eeg_path = base / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"

    if not eeg_path.exists():
        raise FileNotFoundError(f"Не постои фајлот: {eeg_path}")

    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
    return raw


def epoch_raw(raw, duration=None, overlap=None):
    """
    Сегментира continuous EEG сигнал во кратки сегмемти со фиксна должина.

    Параметри
    ---------
    raw : mne.io.Raw
    duration : float
        должина на секоја епоха во секунди (default: config.EPOCH_DURATION)
    overlap : float
        преклопување меѓу епохи во секунди (default: config.EPOCH_OVERLAP)

    Враќа
    -----
    mne.Epochs објект
    """
    duration = duration or config.EPOCH_DURATION
    overlap = overlap if overlap is not None else config.EPOCH_OVERLAP

    epochs = mne.make_fixed_length_epochs(
        raw, duration=duration, overlap=overlap, preload=True, verbose=False
    )
    return epochs


def load_and_epoch(subject_id, use_derivatives=True):
    """
    Комбинирана функција: вчитува + epoch-ира еден субјект во еден чекор.

    Враќа
    -----
    mne.Epochs објект
    """
    raw = load_subject_raw(subject_id, use_derivatives=use_derivatives)
    epochs = epoch_raw(raw)
    return epochs


if __name__ == "__main__":
    #Testing: 
    test_subject = "sub-001"
    epochs = load_and_epoch(test_subject)
    print(f"{test_subject}: {len(epochs)} епохи, облик = {epochs.get_data().shape}")