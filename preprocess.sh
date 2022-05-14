python3 ncbi_corpus_preprocessor.py \
--input_directories ../datasets_new/NCBI/NCBItrainset_corpus.txt ../datasets_new/NCBI/NCBItestset_corpus.txt ../datasets_new/NCBI/NCBIdevelopset_corpus.txt \
--output_directory ../datasets_new/NCBI/preprocessed \
--splits train test val;
python3 dictionary_preprocess.py \
--input_dict_path ../datasets_new/NCBI/dictionaries/medic_06Jul2012.txt \
--output_path ../datasets_new/NCBI/dictionaries/train_dictionary.txt \
--lowercase \
--remove_punct;
python3 ./preprocess_mentions.py \
--input_path ../datasets_new/NCBI/preprocessed/train/ \
--output_path ../datasets_new/NCBI/preprocessed/processed_train/ \
--dict_path ../datasets_new/NCBI/dictionaries/train_dictionary.txt \
--ab3p_path ../Ab3P/identify_abbr \
--misspell_path ../datasets_new/NCBI/resources/ncbi-spell-check.txt \
--remove_without_cui \
--resolve_comp_mentions \
--lowercase \
--remove_punct;
python3 dictionary_preprocess.py \
--input_dict_path ../datasets_new/NCBI/dictionaries/train_dictionary.txt \
--extra_data_path ../datasets_new/NCBI/preprocessed/processed_train/ \
--output_path ../datasets_new/NCBI/dictionaries/val_dictionary.txt \
--lowercase \
--remove_punct;
python3 ./preprocess_mentions.py \
--input_path ../datasets_new/NCBI/preprocessed/val/ \
--output_path ../datasets_new/NCBI/preprocessed/processed_val/ \
--dict_path ../datasets_new/NCBI/dictionaries/val_dictionary.txt \
--ab3p_path ../Ab3P/identify_abbr \
--misspell_path ../datasets_new/NCBI/resources/ncbi-spell-check.txt \
--remove_without_cui \
--resolve_comp_mentions \
--lowercase \
--remove_punct;
python3 dictionary_preprocess.py \
--input_dict_path ../datasets_new/NCBI/dictionaries/val_dictionary.txt \
--extra_data_path ../datasets_new/NCBI/preprocessed/processed_val/ \
--output_path ../datasets_new/NCBI/dictionaries/test_dictionary.txt \
--lowercase \
--remove_punct;
python3 ./preprocess_mentions.py \
--input_path ../datasets_new/NCBI/preprocessed/test/ \
--output_path ../datasets_new/NCBI/preprocessed/processed_test/ \
--dict_path ../datasets_new/NCBI/dictionaries/test_dictionary.txt \
--ab3p_path ../Ab3P/identify_abbr \
--misspell_path ../datasets_new/NCBI/resources/ncbi-spell-check.txt \
--remove_without_cui \
--resolve_comp_mentions \
--lowercase \
--remove_punct;
python3 ner_ncbi_preprocessor.py \
--input_path ../datasets_new/NCBI/preprocessed/ \
--output_path ../datasets_new/NCBI/preprocessed_ner/ \
--splits train test val