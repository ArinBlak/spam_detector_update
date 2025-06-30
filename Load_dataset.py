# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 22:22:41 2025

@author: Aries
"""

from pathlib import Path
import pandas as pd

def load_dataset(spam_path, ham_path, hard_ham_path):
    D = []
    for filepath in Path(spam_path).glob("*"):
         if filepath.name.startswith('.') or not filepath.is_file():
                continue
         with open(filepath, 'r', encoding = 'latin1') as file:
            D.append((file.read(), 1))

    for filepath in Path(ham_path).glob("*"):
         if filepath.name.startswith('.') or not filepath.is_file():
                continue
         with open(filepath, 'r', encoding = 'latin1') as file:
            D.append((file.read(), 0))

    for filepath in Path(hard_ham_path).glob("*"):
        if filepath.name.startswith(".") or not filepath.is_file():
            continue
        with open(filepath, 'r', encoding = 'latin1') as file:
            D.append((file.read(), 0))

    df = pd.DataFrame(D, columns = ['text', 'label'])
    df = df.drop_duplicates().reset_index(drop = True)
    df = df[df['text'].str.strip().astype(bool)]
    df = df.dropna(subset=['text'])
    df = df.sample(frac = 1, random_state = 42).reset_index(drop = True)
    return df