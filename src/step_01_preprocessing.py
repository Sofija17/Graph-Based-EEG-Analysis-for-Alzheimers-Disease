"""
Функции за вчитување на веќе-прочистени (derivatives/) EEG записи
и нивно сегментирање во кратки епохи (сегменти).

"""

import mne
import config
import warnings
from helpers.load_participants import get_ad_cn_subjects

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

    if len(raw.annotations) > 0:
        annotations = raw.annotations.copy()
        descriptions = annotations.description.astype(str)

        boundary_mask = descriptions == "boundary"
        annotations.description[boundary_mask] = "bad_boundary"

        raw = raw.copy()
        raw.set_annotations(annotations)

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    return epochs


def load_and_epoch_subject(subject_id, use_derivatives=True):
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

    warnings.filterwarnings(
        "ignore",
        message="The data contains 'boundary' events.*"
    )

    subjects = get_ad_cn_subjects()

    for subject_id, label in subjects:
        raw = load_subject_raw(subject_id)

        old_epochs = mne.make_fixed_length_epochs(
            raw,
            duration=config.EPOCH_DURATION,
            overlap=config.EPOCH_OVERLAP,
            preload=True,
            reject_by_annotation=False,
            verbose=False,
        )

        new_epochs = epoch_raw(raw)

        removed = len(old_epochs) - len(new_epochs)

        label_name = "AD" if label == 1 else "CN"
        descriptions = set(raw.annotations.description.astype(str))

        print(
            f"{subject_id} ({label_name}) | "
            f"annotations={len(raw.annotations)} | "
            f"types={descriptions} | "
            f"old={len(old_epochs)} | "
            f"new={len(new_epochs)} | "
            f"removed={removed}"
        )

