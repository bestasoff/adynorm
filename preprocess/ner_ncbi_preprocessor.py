import argparse
import nltk
import os

from tqdm import tqdm
from typing import Any, List, Tuple


def get_concept(concept_filename: str) -> List[Any]:
    with open(concept_filename, 'r') as file:
        lines = file.readlines()

    concept = []
    for line in lines:
        if len(line.strip()) == 0:
            continue
        concept.append(line.strip().split('||'))

    return concept


def get_ta(ta_filename: str) -> str:
    with open(ta_filename, 'r') as file:
        lines = file.readlines()

    ta = []
    for line in lines:
        if len(line.strip()) == 0:
            continue
        ta.append(line.strip())
    return ' '.join(ta)


def tokenize_and_match_iob(ta: str, coordinates: List[int]) -> List[Tuple[List[str], List[str]]]:
    sentences = nltk.sent_tokenize(ta)
    tokenized_matched = []
    ta_pointer = 0
    coordinates_iter = iter(coordinates)
    try:
        start, end = next(coordinates_iter)
    except StopIteration:
        start, end = -1, -1

    for sent in sentences:
        sent_tokens = nltk.wordpunct_tokenize(sent)
        iob_tags = []
        for token in sent_tokens:
            for c in token:
                while ta[ta_pointer] != c:
                    ta_pointer += 1
                ta_pointer += 1

            if ta_pointer - len(token) == start:
                iob_tags.append('B')
            elif ta_pointer <= end and ta_pointer > start:
                if ta_pointer == end:
                    try:
                        start, end = next(coordinates_iter)
                    except StopIteration:
                        start, end = -1, -1

                iob_tags.append('I')
            else:
                iob_tags.append('O')
        tokenized_matched.append((sent_tokens, iob_tags))
    return tokenized_matched


def prepare_ner_for_split(input_path: str, output_path: str, log_file):
    _, _, fnames = next(iter(os.walk(input_path)))

    output_file = open(output_path, 'w')
    for concept_file, ta_file in tqdm(zip(*[iter(sorted(fnames))] * 2)):
        concept = get_concept(os.path.join(input_path, concept_file))
        ta = get_ta(os.path.join(input_path, ta_file))
        coordinates = [list(map(int, c[1].split('|'))) for c in concept]
        tokenized_matched = tokenize_and_match_iob(ta, coordinates)
        for tple in tokenized_matched:
            for token, tag in zip(*tple):
                print(f'{token}\t{tag}', file=output_file)
            print(file=output_file)
    output_file.close()


def preprocess_ner_data(
        input_path: str,
        output_path: str,
        splits: List[str],
        log_filename: str
):
    log_file = open(log_filename, 'w')
    if not os.path.exists(input_path):
        print(f"Input directory {input_path} does not exist.", file=log_file)
        log_file.close()
        assert False
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for split in splits:
        input_dir = os.path.join(input_path, split)
        if not os.path.exists(input_dir):
            print(f'Directory {input_dir} does not exist.\n', file=log_file)
            continue
        output_dir = os.path.join(output_path, f'{split}.txt')
        print(
            f'Started processing directory: {input_dir} and split: {split}.',
            file=log_file
        )
        prepare_ner_for_split(input_dir, output_dir, log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments to preprocess NCBI.')
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to directory with concepts and ta files.")
    parser.add_argument(
        "--splits",
        nargs='*',
        help="Types of provided corpuses in the same order."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to directory to save preprocessed NER data.")
    parser.add_argument(
        "--log_filename",
        type=str,
        help="Path to log file.",
        default='./log_file_ner.txt'
    )

    args = parser.parse_args()
    preprocess_ner_data(
        args.input_path,
        args.output_path,
        args.splits,
        args.log_filename
    )
