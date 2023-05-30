#bin/bash

mkdir "stanford_corenlp"

wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.2.zip -P stanford_corenlp

unzip stanford_corenlp/stanford-corenlp-4.5.2.zip -d stanford_corenlp

mv -v stanford_corenlp/stanford-corenlp-4.5.2/* stanford_corenlp

rm -rf stanford_corenlp/stanford-corenlp-4.5.2 stanford_corenlp/stanford-corenlp-4.5.2.zip