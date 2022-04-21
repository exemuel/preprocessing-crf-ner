#!/usr/bin/env python
"""Indonesian NER based on CRF.

Named Entity Recognition based on Conditional Random Field in Indonesian Text.
"""

# standard library
import sys
from datetime import datetime

# 3rd party packages
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# custom modules
from util import *
from benchmarking import *

def main():
    data_path = "data/SINGGALANG.tsv"
    pretrained_tagger = "pre-trained-model/all_indo_man_tag_corpus_model.crf.tagger"

    fr = FileReader(data_path)
    ext = fr.check_type()
    if ext == ".tsv":
        df_raw = fr.read_tsv()
    else:
        df_raw = pd.DataFrame()
        print(f"Currently, the given file cannot be processed.")
        sys.exit()
    
    print(f"--- 1st Experiment ---")
    start_time = datetime.now()
    exp1(df_raw, pretrained_tagger)
    end_time = datetime.now()
    print("--- %s seconds ---" % (end_time - start_time))
    print(f"\n")

    print(f"--- 2nd Experiment ---")
    start_time = datetime.now()
    exp2(df_raw, pretrained_tagger)
    end_time = datetime.now()
    print("--- %s seconds ---" % (end_time - start_time))
    print(f"\n")

    print(f"--- 3rd Experiment ---")
    start_time = datetime.now()
    exp3(df_raw, pretrained_tagger)
    end_time = datetime.now()
    print("--- %s seconds ---" % (end_time - start_time))
    print(f"\n")

    print(f"--- 4th Experiment ---")
    start_time = datetime.now()
    exp4(df_raw, pretrained_tagger)
    end_time = datetime.now()
    print("--- %s seconds ---" % (end_time - start_time))
    print(f"\n")

    print(f"--- 5th Experiment ---")
    start_time = datetime.now()
    exp5(df_raw, pretrained_tagger)
    end_time = datetime.now()
    print("--- %s seconds ---" % (end_time - start_time))
    print(f"\n")
    
    print(f"--- 6th Experiment ---")
    start_time = datetime.now()
    exp6(df_raw, pretrained_tagger)
    end_time = datetime.now()
    print("--- %s seconds ---" % (end_time - start_time))
    print(f"\n")

if __name__ == "__main__":
    main()