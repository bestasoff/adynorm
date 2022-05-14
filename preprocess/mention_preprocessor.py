import re

from string import punctuation
from typing import Dict, List


class MCPreprocessor:
    def __init__(
            self,
            lowercase: bool,
            remove_punct: bool,
            ignore_punct_items: List[str] = None,
            misspell_path: str = None
    ):
        self.lowercase = lowercase
        self.remove_punct = remove_punct

        punctuations = punctuation
        if ignore_punct_items is not None:
            for p_item in ignore_punct_items:
                punctuations = punctuations.replace(p_item, "")

        self.split_punct_regex = re.compile(
            r'[\s{}]+'.format(re.escape(punctuations))
        )

        self.misspell = {}
        if misspell_path is not None:
            self.misspell = self._load_mispell(misspell_path)

    def _remove_punctuation(self, text: str) -> str:
        return ' '.join(self.split_punct_regex.split(text)).strip()

    def _lowercase(self, text: str) -> str:
        return text.lower().strip()

    def _load_mispell(self, path: str) -> Dict[str, str]:
        with open(path, 'r') as file:
            lines = file.readlines()

        mispell = {}
        for line in lines:
            tokens = line.strip().split('||')
            mispell[tokens[0]] = "" if len(tokens) == 1 else tokens[1]

        return mispell

    def _coorect_misspell(self, text: str) -> str:
        tokens = text.split()
        for i, token in enumerate(tokens):
            if token in self.misspell:
                tokens[i] = self.misspell[token]

        return ' '.join(tokens).strip()

    def __call__(self, text: str) -> str:
        if self.lowercase:
            text = self._lowercase(text)

        if len(self.misspell) > 0:
            text = self._coorect_misspell(text)

        if self.remove_punct:
            text = self._remove_punctuation(text)

        return text
