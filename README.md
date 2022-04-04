# Text Preprocessing on Indonesian NER based on CRF
**Impact of Text Preprocessing on Named Entity Recognition based on Conditional Random Field in Indonesian Text**
Samuel I. G. Situmeang

Paper : TBA
Abstract : TBA

## Requirements
* Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.7 installation.
* I recommend sklearn-crfsuite 0.36, which I used for all experiments.
* Download [singgalang.tsv](https://github.com/ialfina/ner-dataset-modified-dee/tree/master/singgalang) and store it in the `data` directory.
* Download [all_indo_man_tag_corpus_model.crf.tagger](https://drive.google.com/file/d/12yJ82GzjnqzrjX14Ob_p9qnPKtcSmqAx/view) and store it in the `pre-trained-model` directory.

## Usage
`python main.py`