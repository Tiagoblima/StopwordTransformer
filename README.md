# StopwordTransformer
The pre-processing is commonly used on text mining to remove unnecessary information from the text. There are several techniques such as, stemming, word-grouping and **stop word removal**. Here, I will use stop word removal to pre process the text from the corpus above.  It build a stopwords list to each one of them. 

The steps are the following:

1. Count the Frequency of each word (term). 
2. Save the 100 most frequents in a dictionary.
3. Remove from each document (bible verse) the stop words. 
4. Remove from each document the words with the lowest Inverse Term Frequency.
5. Remove words with a single occurance

**Main Reference:**

    "Methods based on Zipf’s Law (Z-Methods): In
    addition to the classic stop list, we use three stop
    word creation methods moved by Zipf‟s law,
    including: removing most frequent words (TF-High)
    and removing words that occur once, i.e. singleton
    words (TF1). We also consider removing words with
    low inverse document frequency (IDF) [7, 8]."
    vijayarani, Preprocessing techniques for text mining-an overview

### Find Stopwords

This function build the stopwords of a corpus using the Zip's Law.
It creates a list of words adding: 
1. The 'n' most frequent words (TF)
2. The words with only 1 occurance 
3. The 'n' words with low IDF values

## References

#### Preprocessing
@article{petrovic2019influence,
  title={The Influence of Text Preprocessing Methods and Tools on Calculating Text Similarity},
  author={Petrovi{\'c}, {\DJ}or{\dj}e and Stankovi{\'c}, Milena},
  journal={Facta Universitatis, Series: Mathematics and Informatics},
  volume={34},
  pages={973--994},
  year={2019}
}

@article{vijayarani2015preprocessing,
  title={Preprocessing techniques for text mining-an overview},
  author={Vijayarani, S and Ilamathi, Ms J and Nithya, Ms},
  journal={International Journal of Computer Science \& Communication Networks},
  volume={5},
  number={1},
  pages={7--16},
  year={2015}
}

@article{denny2017text,
  title={Text preprocessing for unsupervised learning: Why it matters, when it misleads, and what to do about it},
  author={Denny, Matthew and Spirling, Arthur},
  journal={When It Misleads, and What to Do about It (September 27, 2017)},
  year={2017}
}
