from stopword import StopWordTransformer
import pandas as pd

ind_cls = pd.read_csv('https://github.com/Tiagoblima/indigenous-corpus/raw/main/indigenous_cls.csv').dropna()
stopword = StopWordTransformer()

corpus = ind_cls.loc[ind_cls['LANG'] == 'Portuguese', 'TEXT'].to_numpy().squeeze()
print(corpus[:10])
print(stopword.fit(corpus).transform(corpus[:10]))
