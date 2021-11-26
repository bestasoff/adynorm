import argparse
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Any

class StringConverter(dict):
    def __contains__(self, item: Any) -> bool:
        return True

    def __getitem__(self, item: Any) -> Any:
        return str

    def get(self, default=None) -> Any:
        return str

class CTD_disease_medic:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

        self.tsv_config = {
            "usecols": [0, 1, 2, 7],
            "names": ['Name', 'ID', 'AltID', 'Synonyms']
        }

    def load_ctd(self) -> pd.DataFrame:
        data = pd.read_csv(
            self.input_path,
            usecols=self.tsv_config['usecols'],
            names=self.tsv_config['names'],
            comment='#',
            delimiter='\t',
            converters=StringConverter(),
            header=None
        )
        return data

    def preprocess(self) -> None:
        ctd_data = self.load_ctd()
        results = []

        for idx, row in tqdm(ctd_data.iterrows()):
            ID = row['ID']
            if pd.isna(row.AltID):
                alt_ids = ''
            else:
                alt_ids = row.AltID

            if len(alt_ids) > 0:
                ids = '|'.join([ID, alt_ids])
            else:
                ids = ID

            name = row.Name
            synonyms = row.Synonyms if not pd.isna(row.Synonyms) else ''

            names = []
            for n in [name, synonyms]:
                if len(n) > 0:
                    names.append(n)

            if len(names) == 0:
                continue

            names = '|'.join(names) if len(names) > 1 else names[0]

            results.append('||'.join([ids, names]))

        with open(self.output_path, 'w') as file:
            for res in results:
                file.write(res)
                file.write('\n')

def preprocess(input_path: str, output_path: str) -> None:
    ctd = CTD_disease_medic(input_path, output_path)
    ctd.preprocess()

def main() -> None:
    parser = argparse.ArgumentParser(description="Args to preprocess CTD diseases MEDIC")
    parser.add_argument('--input_path', type=str, required=True,
                    help='Path to MEDIC CTD_diseases')
    parser.add_argument('--output_path', type=str, required=True,
                    help='Path to save preprocessed output')

    args = parser.parse_args()
    preprocess(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
