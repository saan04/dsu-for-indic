# DSU for Indic Languages

Discrete Speech Units (DSU) extracted from the 6th layer of HuBERT and KMeans clustering offer a novel way to represent speech and lay the groundwork for Speech-to-Speech translation without the need for intermediary text, enabling advancements in Textless NLP.

## Motivation
The HuBERT model, a popular method for generating speech representations, is primarily trained on foreign-accented speech. This creates a gap in applying the technology to Indic languages, which are highly diverse and spoken by millions. There are two main approaches to addressing this:

1. Train a new HuBERT model on Indic data, which is a resource-heavy and time-consuming process.
2. Leverage existing multilingual models trained on diverse datasets, such as mHuBERT-147 and Wav2Vec2-XLSR, which have broader language coverage, to see if we can still obtain favorable results for Indic languages.

## Approach
In this repository, we aim to extract DSU using the mHuBERT-147 model, trained on 147 languages, and Wav2Vec2-XLSR, trained on 50+ languages. These models are more robust across languages and accents, making them suitable for our experiments on Indian-accented speech.

## Models Used
1. mHuBERT-147: A multilingual version of HuBERT trained on 147 languages, designed to handle a wide variety of speech data, including Indian accents.
2. Wav2Vec2-XLSR: Cross-Language Speech Representation model trained on over 50 languages, further enhancing the ability to work with underrepresented languages like those from the Indian subcontinent.

## Repository Contents
1. mhubert.py, mhubert-v2.py, mhubert-v3.py: Versioned scripts to load the mHuBERT model and apply KMeans clustering for DSU extraction.
2. w2v2.py: Script for processing with the Wav2Vec2-XLSR model for cross-lingual DSU extraction.
