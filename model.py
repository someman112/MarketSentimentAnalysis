import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

pd.options.mode.chained_assignment = None  # default='warn'

def get_sentiment(string):  # helper function for getting sentiment from a text
    analysis = TextBlob(string)
    return analysis.sentiment.subjectivity


def get_sentiment2(string):  # helper function for getting sentiment from a text
    analysis = TextBlob(string)
    return analysis.sentiment.polarity


def SIA(string):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(string)


df = pd.read_csv("Combined_News_DJIA.csv")
headlines = df.iloc[:, 2:27]  # headlines (first two are index, directional boolean)

# for each day, merging each headline to one paragraph for sentiment analysis
merged_headlines = []
for row in range(0, len(headlines.index)):
    merged_headlines.append(' '.join(str(x) for x in headlines.iloc[row, :]))

# cleaning headlines for sentiment analysis
clean_headlines = []
for i in range(0, len(merged_headlines)):
    clean_headlines.append(merged_headlines[i])
    clean_headlines[i] = re.sub("b[(')]", "", clean_headlines[i])  # remove b'
    clean_headlines[i] = re.sub('b[(")]', "", clean_headlines[i])  # remove b"

df['clean'] = clean_headlines  # adding clean headlines back to the dataset

df['sub'] = df['clean'].apply(get_sentiment)  # feature1: Subjectivity
df['pol'] = df['clean'].apply(get_sentiment2)  # feature2: Polarity

df['compound'] = None  # feature3: Compound
df['neg'] = None  # feature4: Negativity
df['neu'] = None  # feature5: Neutrality
df['post'] = None  # feature6: positivity
for i in range(0, len(df["clean"])):
    sentiment = SIA(df["clean"][i])
    df['compound'][i] = sentiment['compound']
    df['neg'][i] = sentiment['neg']
    df['neu'][i] = sentiment['neu']
    df['post'][i] = sentiment['pos']

needed_columns = ["Label", "sub", "pol", "compound", "neg", "neu", "post"]
data = df[needed_columns]

inpt = np.array(data.drop(['Label'], axis=1))
print(inpt)
actual_results = np.array(data["Label"])
print(actual_results)

# Test/Train ratio 80-20 (80% train, 20% test)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(inpt, actual_results, test_size=0.2, random_state=0)


# model creation/ training
model = LinearDiscriminantAnalysis().fit(x_train, y_train)
model_predictions = model.predict(x_test)
print(model_predictions)
print(y_test)

#results
print(classification_report(y_test, model_predictions))