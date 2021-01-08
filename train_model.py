import spacy
import pandas as pd
import pl_core_news_sm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from joblib import dump, load

spacy.load('pl_core_news_sm')

POLISH_SENTIMENT_DATASET_PATH = "../polish_sentiment_dataset.csv"
SAVED_MODEL_PATH = "clf_lsvc_polish_sentiment.joblib"

nlp = pl_core_news_sm.load()


def prepare_sentiment_dataset(path):
    df = pd.read_csv(POLISH_SENTIMENT_DATASET_PATH, sep=",")
    df.dropna(inplace=True)
    blanks = []
    for i, d, l, r in df.itertuples():
        if d.isspace() or d == "0":
            blanks.append(i)
    df.drop(blanks, inplace=True)
    return df


def train_sentiment_model(data_frame):
    X = data_frame['description']
    y = data_frame['rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC()), ])
    return text_clf_lsvc.fit(X_train, y_train)


def save_sentiment_model(model, path):
    dump(model, path)

def get_sentiment_model(path):
    model =  load(path)
    print(model.predict("nie kupujcie"))


if __name__ == '__main__':
    #df = prepare_sentiment_dataset(POLISH_SENTIMENT_DATASET_PATH)
    #model = train_sentiment_model(df)
    #save_sentiment_model(model, SAVED_MODEL_PATH)
    model = get_sentiment_model(SAVED_MODEL_PATH)
    #print(type(model))
    #print(model.predict("nie kupujcie tych akcji bo słabe są"))
