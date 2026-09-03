"""
Load participants.tsv and prepare a list of (subject_id, label)
pairs for AD vs CN classification (FTD subjects are excluded).
"""

import pandas as pd
import config


def load_participants():
    df = pd.read_csv(config.PARTICIPANTS_TSV, sep="\t")
    return df


def get_ad_cn_subjects():
    """
    Keep only AD (Group='A') and CN (Group='C') subjects,
    excluding FTD (Group='F') subjects.

    Returns
    -------
    list of tuples: [(subject_id, label), ...]
    label: 0 = CN (Healthy Control), 1 = AD (Alzheimer's Disease)
    """
    df = load_participants()
    df_filtered = df[df["Group"].isin(config.GROUP_MAP.keys())]

    subjects = []
    for _, row in df_filtered.iterrows():
        subject_id = row["participant_id"]
        label = config.GROUP_MAP[row["Group"]]
        subjects.append((subject_id, label))

    return subjects


if __name__ == "__main__":
    subjects = get_ad_cn_subjects()
    n_ad = sum(1 for _, label in subjects if label == 1)
    n_cn = sum(1 for _, label in subjects if label == 0)

    print(f"Вкупно субјекти (AD+CN): {len(subjects)}")
    print(f"AD: {n_ad}, CN: {n_cn}")
    print("\nПрви 5 субјекти:")
    for subject_id, label in subjects[:5]:
        print(f"  {subject_id}: label={label}")
