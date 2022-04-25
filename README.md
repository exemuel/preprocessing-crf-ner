# preprocessing-crf-ner

## Description
This work contributes to extensively assessing the impact of preprocessing tasks on the named entity recognition success in Indonesian text at various feature dimensions and possible interactions among these tasks.

![Flowchart of Experimental Methods on text preprocessing in Indonesian NER based on CRF](https://github.com/exemuel/preprocessing-crf-ner/blob/main/images/flowchart.png)

### Preprocessing Procedures
1. Contractions Expansion
2. Lowercase Conversion
3. Stemming
4. Number to Words Conversion
5. Hyphen and Comma Splitting

###	Feature Extraction
1.	The word
2.	The length of the word or number of characters 
3.	Prefixes and suffixes of the word of varying lengths
4.	The word in lowercase
5.	Stemmed version of the word, which deletes all vowels along with g, y, n from the end of the word, but leaves at least a 2 character long stem
6.	If the word is a punctuation mark
7.	If the word is a digit
8.	Features mentioned above for the previous word, the following word, and the words two places before and after
9.	Word POS tag
10.	If the word is at the beginning of the sentence (BOS) or the end of the sentence (EOS) or neither

## Requirements
* Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.7 installation.
* I recommend sklearn-crfsuite 0.36, which I used for all experiments.
* Download [singgalang.tsv](https://github.com/ialfina/ner-dataset-modified-dee/tree/master/singgalang) and store it in the `data` directory.
* Download [all_indo_man_tag_corpus_model.crf.tagger](https://drive.google.com/file/d/12yJ82GzjnqzrjX14Ob_p9qnPKtcSmqAx/view) and store it in the `pre-trained-model` directory.

## Usage
`python main.py`