import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
sample_data = {
    'ID': [3364, 352, 8312, 4371, 4433, 6273],
    'Entity': ['Facebook', 'Amazon', 'Microsoft', 'CS-GO', 'Google', 'FIFA'],
    'Sentiment': ['Irrelevant', 'Neutral', 'Negative', 'Negative', 'Neutral', 'Negative']
}
data = pd.DataFrame(sample_data)
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'
data['Sentiment'] = data['Sentiment'].apply(get_sentiment)
plt.figure(figsize=(8, 6))
data['Sentiment'].value_counts().plot(kind='bar', color=['blue', 'red', 'green'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/content/twitter_validation.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Display the preprocessed data
print(data.head())

# Function to get sentiment polarity
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis
data['sentiment'] = data['text'].apply(get_sentiment)

# Display the sentiment analysis results
print(data[['text', 'sentiment']].head())

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
data['sentiment'].value_counts().plot(kind='bar', color=['blue', 'red', 'green'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

