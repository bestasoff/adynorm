import argparse
import os

from typing import (
    Any,
    List
)

def preprocess(inp_filename: str, out_directory: str) -> None:
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    with open(inp_filename, 'r') as file:
        lines = file.readlines() + ['\n']

    queries, pmids = [], []
    doc_cnt, q_cnt = 0, 0

    for line in lines:
        line = line.strip()

        if "|t|" in line:
            title = line.split('|')[2]
        elif "|a|" in line:
            abstract = line.split('|')[2]
        elif "\t" in line:
            line = line.split('\t')
            assert len(line) == 6
            pmid, start, end, mention, cls, cui = line
            query = pmid + "||" + start + "|" + end + "||" + cls + "||" + mention + "||" + cui
            queries.append(query)
        elif len(queries) != 0:
            if pmid in pmids:
                print(f"PMID {pmid} was processed before. Skipping...")
                queries, title, abstract = [], "", ""
                continue

            title_abstract = title + '\n\n' + abstract + '\n'
            concept = '\n'.join(queries) + '\n'

            ta_filename = os.path.join(out_directory, f"{pmid}.txt")
            concept_filename = os.path.join(out_directory, f"{pmid}.concept")

            with open(ta_filename, "w") as file:
                file.write(title_abstract)
            with open(concept_filename, 'w') as file:
                file.write(concept)

            doc_cnt += 1
            q_cnt += len(queries)
            queries, title, abstract = [], "", ""

    print(f"Output folder: {out_directory}\nDocuments processed: {doc_cnt}\nQueries processed: {q_cnt}")

def main() -> None:
    parser = argparse.ArgumentParser(description='Process arguments to preprocess NCBI.')
    parser.add_argument("--input_file", type=str, help="Path to NCBI{train, test, devel}.txt")
    parser.add_argument("--output_directory", type=str, help="Path to save preprocessed output")

    args = parser.parse_args()
    preprocess(args.input_file, args.output_directory)

if __name__ == "__main__":
    main()
