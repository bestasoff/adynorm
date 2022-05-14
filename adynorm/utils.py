import os
import pandas as pd
import re

from dataclasses import dataclass
from enum import Enum
from typing import List, TextIO, Dict


@dataclass
class Concept:
    disease_name: str
    disease_id: str
    parent_ids: List[str]
    synonyms: List[str]


def get_concepts_list(MEDIC: pd.DataFrame) -> List[Concept]:
    if not isinstance(MEDIC, pd.DataFrame):
        raise ValueError("MEDIC is not a DataFrame")

    MEDIC1 = MEDIC.fillna("")
    concepts: List[Concept] = []
    for i in range(len(MEDIC1)):
        record = MEDIC1.iloc[i]
        concepts.append(Concept(
            disease_name=record['DiseaseName'],
            disease_id=record['DiseaseID'],
            parent_ids=record['ParentIDs'].split('|'),
            synonyms=record['Synonyms'].split('|')
        ))

    return concepts


TEXT_LINE_RE = re.compile(r'^(\d+)\|([ta])\|(.*)$')


class Split(Enum):
    train = "train"
    val = "develop"
    test = "test"


class Annotation:
    def __init__(self, PMID: str, start: int, end: int, text: str, type_: str, norms: List[str]):
        self.PMID = PMID
        self.start = start
        self.end = end
        self.text = text
        self.type = type_
        self.norms = norms

    def verify(self, title_abstract: str):
        if title_abstract[self.start:self.end] != self.text:
            raise ValueError("text mismatch")

    def __repr__(self) -> str:
        return f'''
PMID: {self.PMID}
start: {self.start}
end: {self.end}
text: {self.text}
type: {self.type}
norms: {self.norms}
'''


def parse_annotation_line(line: str, ln: int):
    fields = [f.strip() for f in line.split('\t')]

    if len(fields) != 6:
        raise ValueError(f'Failed to parse line {ln}: {line}')

    PMID = fields[0]
    try:
        start = int(fields[1])
        end = int(fields[2])
    except:
        raise ValueError(f'Failed to parse line {ln}: {line}')

    text = fields[3]
    type_ = fields[4]
    if len(text) != end - start:
        raise ValueError(f'Text "{text}" has length {len(text)}, but end-start is ({end}-{start})')

    norms = fields[5].split('|')

    return Annotation(PMID, start, end, text, type_, norms)


def check_PMID(current: str, seen: str) -> str:
    if current is None or current == seen:
        return seen
    else:
        raise ValueError(f'Expected PMID {current}, got {seen}')


def read_ncbi_disease(file: TextIO) -> List[Annotation]:
    annotations_all = []
    current_PMID, title, abstract, annotations = None, None, None, []
    for ln, line in enumerate(file, start=1):
        line = line.rstrip('\n')
        if not line:
            if current_PMID is not None:
                annotations_all.extend(annotations)
            current_PMID, title, abstract = None, None, None
            annotations = []
            continue

        m = TEXT_LINE_RE.match(line)
        if m:
            PMID, tiab, text = m.groups()
            current_PMID = check_PMID(current_PMID, PMID)
        else:
            annotation = parse_annotation_line(line, ln)
            current_PMID = check_PMID(current_PMID, annotation.PMID)
            annotations.append(annotation)
    if current_PMID is not None:
        annotations_all.extend(annotations)

    return annotations_all


def load_ncbi_disease(filename: str):
    with open(filename) as f:
        return read_ncbi_disease(f)


def get_annotations(path: str, mode: Split) -> List[List[Annotation]]:
    if not isinstance(mode, Split):
        raise ValueError("mode incorrect type")

    mode = mode.value
    mode_path = os.path.join(path, f"NCBI{mode}set_corpus.txt")
    return load_ncbi_disease(mode_path)


def get_all_mode_annotations(path: str) -> Dict[str, List[List[Annotation]]]:
    all_annotations = {}
    for mode_type in ['train', 'val', 'test']:
        mode = getattr(Split, mode_type)
        all_annotations[mode_type] = get_annotations(path, mode)

    return all_annotations