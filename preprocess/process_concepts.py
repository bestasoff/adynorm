import argparse
import os
import re

import pandas as pd

from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Set,
    Tuple
)
from tqdm import tqdm

from abbreviation_expander import AbbreviationExpander
from mc_preprocessor import MCPreprocessor

def get_cui_set(dict_path: str) -> Set[str]:
    cui_set = set()
    with open(dict_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        cuis = line.strip().split('||')[0]
        for cui in cuis.split('|'):
            cui_set.add(cui.strip())
    return cui_set

def get_concept(concept_filename: str) -> List[Any]:
    with open(concept_filename, 'r') as file:
        lines = file.readlines()
    concept = []
    for line in lines:
        concept.append(line.strip().split('||'))
    return concept

def expand_abbreviations(
    concept: List[Any],
    abbreviation_dict: Dict[str, str]
) -> List[Any]:
    expanded = []
    for i, record in enumerate(concept):
        cur_mention = record[-2]
        while True:
            tokens = [token.strip() for token in cur_mention.split()]
            exp_tokens = []
            for token in tokens:
                if '/' in token:
                    token = '/'.join(
                        [abbreviation_dict.get(t.strip(), t.strip())
                         for t in token.split('/')]
                    )
                if token.endswith(','):
                    token = token.replace(',', '')
                    token = abbreviation_dict.get(token, token) + ','
                else:
                    token = abbreviation_dict.get(token, token)
                exp_tokens.append(token)
            expanded_mention = ' '.join(exp_tokens)
            if expanded_mention == cur_mention:
                break
            cur_mention = expanded_mention
        record[-2] = expanded_mention
        concept[i] = record
    return concept

def resolve_composite_mentions(concept: List[Any]) -> List[Any]:
    for i, record in enumerate(concept):
        if '|' in record[-2]:
            continue
        mentions, cui = resolve(record[-2].strip(), record[-1].strip())
        cuis = cui.strip().split('|')
        if len(cuis) > 1 and len(mentions) == len(cuis):
            record[-2] = '|'.join(mentions)
            record[-1] = '|'.join(cuis)
        concept[i] = record
    return concept

def prefix_of_pattern(mention: str, cuis: List[str]) -> Tuple[List[str], str]:
    pattern = re.compile(
        "(?P<prefix>[a-zA-Z-]+) of (both )?(the )?(?P<suffix1>([a-zA-Z-]+ )+)(and|or) (the )?(?P<suffix2>([a-zA-Z-]+ ?)+)"
    )
    match = pattern.match(mention)
    if not match:
        return False, None, None
    prefix = match.group('prefix')
    mention1 = ' '.join([prefix, 'of', match.group('suffix1').strip()])
    mention2 = ' '.join([prefix, 'of', match.group('suffix2').strip()])
    return True, [mention1, mention2], '|'.join(cuis)

def nested_and_pattern(mention: str, cuis: List[str]) -> Tuple[List[str], str]:
    pattern = re.compile(
        "(?P<prefix_list>([a-zA-Z-]+,? )+)(and|or|and/or) (?P<prefix_last>[a-zA-Z-]+ and [a-zA-Z-]+) (?P<stem>.*)"
    )
    match = pattern.match(mention)
    if not match:
        return False, None, None
    stem = match.group('stem')
    prefix_list = match.group('prefix_list').strip().split(',')
    prefix_list.append(match.group('prefix_last'))
    prefix_list = filter(len, prefix_list)
    mention_list = [prefix.strip() + ' ' + stem for prefix in prefix_list]
    return True, mention_list, '|'.join(cuis)

def trivial_pattern(mention: str, cuis: List[str]) -> Tuple[List[str], str]:
    pattern = re.compile(
        "(?P<prefix_list>([a-zA-Z-]+,? )+)(and|or|and/or) (?P<prefix_last>(the )?[a-zA-Z-]+) (?P<stem>.*)"
    )
    match = pattern.match(mention)
    if not match:
        return False, None, None
    stem = match.group('stem')
    prefix_list = match.group('prefix_list').strip().split(',')
    prefix_list.append(match.group('prefix_last'))
    prefix_list = filter(len, prefix_list)
    mention_list = [prefix.strip() + ' ' + stem for prefix in prefix_list]
    return True, mention_list, '|'.join(cuis)

def slash_pattern(mention: str, cuis: List[str]) -> Tuple[List[str], str]:
    pattern = re.compile(
        "(?P<prefix>(.* )*)(?P<composite1>.*)\/(?P<composite2>.*)?(?P<suffix>( .*)*)"
    )
    match = pattern.match(mention)
    if not match:
        return False, None, None
    prefix = match.group('prefix').strip()
    suffix = match.group('suffix').strip()
    composite_list = [match.group('composite1').strip(), match.group('composite2').strip()]
    mention_list = [prefix + ' ' + composite + ' ' + suffix for composite in composite_list]
    mention_list = [mention.strip() for mention in mention_list]
    return True, mention_list, '|'.join(cuis)

def resolve(mention: str, cui: str) -> Tuple[List[str], str]:
    cuis = cui.strip().split('|')
    OMIM_in = all(['OMIM' in cui for cui in cuis])
    MESH_in = all(['OMIM' not in cui for cui in cuis])
    if not (OMIM_in or MESH_in):
        cuis = list(filter(lambda cui: 'OMIM' not in cui, cuis))
    if len(cuis) == 1:
        return [mention], cuis[0]

    tokens = mention.split()

    matches = prefix_of_pattern(mention, cuis)
    if matches[0]:
#         print(mention)
#         print(matches[1])
#         print()
        return matches[1], matches[2]
    matches = nested_and_pattern(mention, cuis)
    if matches[0]:
#         print(mention)
#         print(matches[1])
#         print()
        return matches[1], matches[2]
    matches = trivial_pattern(mention, cuis)
    if matches[0]:
#         print(mention)
#         print(matches[1])
#         print()
        return matches[1], matches[2]
    matches = slash_pattern(mention, cuis)
    if matches[0] and 'and/or' not in mention:
#         print(mention)
#         print(matches[1])
#         print()
        return matches[1], matches[2]

    if len(cuis) == len(tokens):
        return tokens, '|'.join(cuis)
    else:
        return [mention], '|'.join(cuis)

def apply_preprocessor(concept: List[Any], preprocessor: MCPreprocessor) -> List[Any]:
    for i, record in enumerate(concept):
        record[-2] = preprocessor(record[-2])
        concept[i] = record
    return concept

def process_records_without_cui(
    concept: List[Any],
    remove_without_cui: bool,
    cui_set: Set[str]
) -> List[Any]:
    processed = []
    for record in concept:
        cui = record[-1].replace('OMIM:', '').replace('MESH:', '')
        concept_cui_set = set([c.strip() for c in cui.split('|')])
        cui_diff_set = concept_cui_set - cui_set
        if len(cui_diff_set) == 0:
            record[-1] = cui
        else:
            record[-1] = "-1"
        processed.append(record)
    return processed

def preprocess(
    input_path: str,
    output_path: str,
    dict_path: str,
    do_lowercase: bool,
    remove_punct: bool,
    misspell_path: str,
    remove_without_cui: bool,
    ab3p_path: str,
    resolve_comp_mentions: bool
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.mkdir(exist_ok=True)

    files = sorted(
        map(
            lambda path: path.stem,
            filter(lambda path: path.suffix == '.concept', input_path.iterdir())
        )
    )

    abbr_expander = AbbreviationExpander(ab3p_path=ab3p_path)
    preprocessor = MCPreprocessor(
        do_lowercase=do_lowercase,
        remove_punct=remove_punct,
        ignore_punct_items=['|'],
        misspell_path=misspell_path
    )
    cui_set = get_cui_set(dict_path)

    q = 0
    for filename in tqdm(files):
        concept_file = input_path / f'{filename}.concept'
        title_abstract_file = input_path / f'{filename}.txt'
        output_file = output_path / f'{filename}.concept'

        concept = get_concept(concept_file)
        abbreviation_dict = abbr_expander(title_abstract_file)
        concept = expand_abbreviations(concept, abbreviation_dict)

        if resolve_comp_mentions:
            concept = resolve_composite_mentions(concept)

        concept = apply_preprocessor(concept, preprocessor)
        concept = process_records_without_cui(concept, remove_without_cui, cui_set)

        with open(output_file, 'w') as file:
            for record in concept:
                print('||'.join(record), file=file)

        q += len(concept)
    print(f'Records processed: {q}')

def main():
    parser = argparse.ArgumentParser(description='Preprocess mantions and concepts')

    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)

    parser.add_argument('--dict_path',  type=str, default=None)
    parser.add_argument('--ab3p_path',  type=str, default=None, help='ab3p path')
    parser.add_argument('--misspell_path',  type=str, default=None)

    parser.add_argument('--remove_without_cui',  action="store_true")
    parser.add_argument('--resolve_comp_mentions',  action="store_true")
    parser.add_argument('--do_lowercase',  action="store_true")
    parser.add_argument('--remove_punct',  action="store_true")

    args = parser.parse_args()
    preprocess(
        input_path=args.input_path,
        output_path=args.output_path,
        dict_path=args.dict_path,
        do_lowercase=args.do_lowercase,
        remove_punct=args.remove_punct,
        misspell_path=args.misspell_path,
        remove_without_cui=args.remove_without_cui,
        ab3p_path=args.ab3p_path,
        resolve_comp_mentions=args.resolve_comp_mentions
    )

if __name__ == "__main__":
    main()
