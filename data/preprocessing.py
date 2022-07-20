from sklearn.model_selection import train_test_split
import re
from preprocessing import df
import numpy as np

def changeByteToStr(df):
    print(type(df['text'][0]))
    print(type(df['y'][0]))

    df['text'] = df['text'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df['y'] = df['y'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    print(type(df['text'][0]))
    print(type(df['y'][0]))
    return df


## create stopwords
lst_stopwords = nltk.corpus.stopwords.words("english")
## add words that are too frequent
lst_stopwords = lst_stopwords + ["cnn", "say", "said", "new"]

## cleaning function
def utils_preprocess_text(txt, punkt=True, lower=True, slang=True, lst_stopwords=None, stemm=False, lemm=True):
    ### separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    ### remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    ### strip
    txt = " ".join([word.strip() for word in txt.split()])
    ### lowercase
    txt = txt.lower() if lower is True else txt
    ### slang
    txt = contractions.fix(txt) if slang is True else txt
    ### tokenize (convert from string to list)
    lst_txt = txt.split()
    ### stemming (remove -ing, -ly, ...)
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]
    ### lemmatization (convert the word into root word)
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]
    ### remove Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in
                   lst_stopwords]
    ### back to string
    txt = " ".join(lst_txt)
    return txt

def preprocessText(df):
    df["text_clean"] = df["text"].apply(
        lambda x: utils_preprocess_text(x, punkt=True, lower=True, slang=False, lst_stopwords=lst_stopwords, stemm=False,
                                        lemm=False))
    df["y_clean"] = df["y"].apply(
        lambda x: utils_preprocess_text(x, punkt=True, lower=True, slang=False, lst_stopwords=lst_stopwords, stemm=False,
                                        lemm=False))
    return df

def addEndStartTokens(df):
    df['y_clean'] = df['y_clean'].apply(lambda x: 'sostok ' + x + ' eostok')
    return df

def splitData(df,testSize,randomState):
    X_train, X_test, y_train, y_test = train_test_split(np.array(df['text_clean']), np.array(df['y_clean']),
                                                        test_size=testSize, random_state=randomState, shuffle=True)
    return X_train,X_test,y_train,y_test