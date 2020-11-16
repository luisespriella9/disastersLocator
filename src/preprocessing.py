import nltk
import re
import numpy as np
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stemmer = PorterStemmer() 
stopwords_english = stopwords.words('english') 

def build_vocab(tweets_array):
    vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 
    for tweet in tweets_array:
        for word in tweet:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def process_dataset(dataset, vocab=None, max_len=None, tweets_column='text', target_column='target'):
    tweets = dataset[tweets_column].values
    targets = np.array(dataset[target_column].values)
    processed_tweets, vocab, max_len = process_tweets(tweets, vocab, max_len)
    return processed_tweets, targets, vocab, max_len

def process_tweets(tweets, vocab=None, max_len=None):
    cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]
    if not vocab:
        vocab = build_vocab(cleaned_tweets)
    sequences = tweets_to_sequences(cleaned_tweets, vocab, unk_tag='__UNK__')
    if not max_len:
        max_len = max([len(seq) for seq in sequences])+1 #+1 for end of sentence tag 
    padded_tweets = pad_sequences(sequences, vocab, max_len, end_tag='__</e>__', pad_tag='__PAD__')
    return padded_tweets, vocab, max_len

def clean_tweet(text):
    if type(text)!=str and type(text)!=np.str_:
        print("type of ", type(text), "cannot be processed")
        return
    # remove special characters
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    # remove stopwords
    cleaned_tweet = []
    for word in text.split():
        stem_word = stemmer.stem(word)
        if stem_word not in stopwords_english:
            cleaned_tweet.append(stem_word)
    return cleaned_tweet

def tweets_to_sequences(sentence_tweets, vocab, unk_tag='__UNK__'):
    transformed_tweets = []
    for tweet in sentence_tweets:
        processed_tweet = []
        for word in tweet:
            if word in vocab:
                processed_tweet.append(vocab[word])
            else:
                processed_tweet.append(vocab[unk_tag])
        transformed_tweets.append(np.array(processed_tweet, dtype=np.int64))
    return transformed_tweets

def pad_sequences(sequences, vocab, max_len, end_tag='__</e>__', pad_tag='__PAD__'):
    padded_sequences = []
    for sequence in sequences:
        padded_sequence = np.array(list(sequence) + [vocab[end_tag]] + [vocab[pad_tag]]*(max_len-len(sequence)-1))
        if (len(padded_sequence) == max_len):
            padded_sequences.append(padded_sequence)
    return np.array(padded_sequences)