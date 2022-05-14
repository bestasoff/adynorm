import subprocess

from typing import Dict


class AbbreviationExpander:
    def __init__(self, ab3p_path: str):
        self.ab3p_path = ab3p_path

    def __call__(self, text_corpus_path: str) -> Dict[str, str]:
        result = subprocess.run(
            [self.ab3p_path, text_corpus_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        result_str = result.stdout.decode('utf-8')
        if 'Cannot open' in result_str:
            raise Exception(result_str)
        if 'failed to open' in result_str:
            raise Exception(result_str)
        if 'for type cshset does not exist' in result_str:
            raise Exception(result_str)

        abbreviation_mapper = {}
        for line in result_str.split('\n'):
            if len(line.split('|')) != 3:
                continue
            short, long, _ = line.split('|')
            abbreviation_mapper[short.strip()] = long.strip()

        return abbreviation_mapper
