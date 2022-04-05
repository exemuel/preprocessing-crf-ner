#!/usr/bin/env python
"""Indonesian NER based on CRF.

Named Entity Recognition based on Conditional Random Field in Indonesian Text.
"""

# standard library
import sys
from tabnanny import verbose

# 3rd party packages
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# custom modules
from util import *

__author__ = "Samuel Situmeang"
__copyright__ = "Copyright 2022, Institut Teknologi Del"
__credits__ = ["Samuel Situmeang"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Samuel Situmeang"
__email__ = "samuel.situmeang@del.ac.id"
__status__ = "Production"

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
    
    dp = DatasetPreparator(df_raw, pretrained_tagger)
    df = dp.check_post()
    df = dp.add_bio_ne()
    
    getter = SentenceGetter(df)
    sentences = getter.sentences

    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    classes = np.unique(df[["bio_ne"]])
    classes = classes.tolist()
    classes.remove("O")
    new_classes = classes.copy()

    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist ]
    y_pred_flat = [item for sublist in y_pred for item in sublist ]
    print(classification_report(y_test_flat, y_pred_flat, labels = new_classes))

if __name__ == "__main__":
    main()