import argparse
import glob
import os

from tqdm import tqdm
from typing import List, Dict

from mention_preprocessor import MCPreprocessor


def dict_2_name2cui(
        dictionary: List[str],
        preprocessor: MCPreprocessor,
        name2cui: Dict[str, str] = {}
) -> Dict[str, str]:
    for record in tqdm(dictionary):
        cui, names = record.split('||')
        for name in names.strip().split('|'):
            name = preprocessor(name)

            if name not in name2cui:
                name2cui[name] = cui
            elif name in name2cui and cui not in name2cui[name]:
                name2cui[name] = name2cui[name] + '|' + cui

    return name2cui


def preprocess(
        dictionary_path: str,
        lowercase: bool,
        remove_punct: bool,
        output_path: str,
        extra_data_path: str = None,
) -> None:
    with open(dictionary_path, 'r', encoding='utf-8') as file:
        dictionary = file.readlines()

    preprocessor = MCPreprocessor(
        lowercase=lowercase,
        remove_punct=remove_punct
    )

    name2cui = dict_2_name2cui(
        dictionary=dictionary,
        preprocessor=preprocessor
    )

    if extra_data_path is not None:
        add_data = []
        concept_files = glob.glob(os.path.join(extra_data_path, "*.concept"))
        for filename in concept_files:
            with open(filename, 'r') as file:
                concepts = file.readlines()

            for concept in concepts:
                _, _, _, name, cui = concept.strip().split("||")
                add_data.append('||'.join([cui, name]))

        name2cui = dict_2_name2cui(
            dictionary=add_data,
            preprocessor=preprocessor,
            name2cui=name2cui
        )

    print(f'Number of unique names: {len(name2cui)}')
    dirname = os.path.dirname(output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(output_path, 'w') as file:
        for name, cui in name2cui.items():
            print(f"{cui}||{name}", file=file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dict_path', type=str, required=True,
                        help='Path to created MEDIC dictionary.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path for preprocessed dictionary.')

    parser.add_argument('--extra_data_path', type=str, default=None,
                        help='Path to other extra data.')

    parser.add_argument('--lowercase', action="store_true")
    parser.add_argument('--remove_punct', action="store_true")

    args = parser.parse_args()

    preprocess(
        args.input_dict_path,
        args.lowercase,
        args.remove_punct,
        args.output_path,
        args.extra_data_path
    )


if __name__ == "__main__":
    main()
