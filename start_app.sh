python3 app.py \
        --port 8080 \
        --ner_model_path trained_models/trained_ner_model_85_88_87.pth \
        --entity_classifier_path trained_models/entity_classifier_model.pth \
        --adynorm_path trained_models/adynorm_model \
        --adynorm_net_path trained_models/adynorm_model.pth \
        --val_dict_path datasets/NCBI/dictionaries/test_dictionary.txt \
        --model_name_or_path 'dmis-lab/biobert-base-cased-v1.1'
