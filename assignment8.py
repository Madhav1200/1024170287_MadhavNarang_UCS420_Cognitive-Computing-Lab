# UCS420 - Lab Assignment 8
# Name: Madhav Narang
# Roll Number: 1024170287



# Question 1

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Paragraph about my favourite topic: Cricket
paragraph = """
Cricket is my favorite sport because it combines strategy, patience, and teamwork.
I enjoy watching international matches, especially when skilled batters build partnerships under pressure.
Fast bowlers use pace and swing, while spinners trick batters with flight and turn.
A great captain studies the pitch, chooses smart field placements, and motivates every player.
The final overs are exciting because one boundary can change the result.
Cricket teaches discipline, resilience, and respect for opponents.
"""

# 1. Convert to lowercase and remove punctuation
lower_text = paragraph.lower()
clean_text = re.sub(r'[^\w\s]', '', lower_text)

print("Original Paragraph:")
print(paragraph)

print("\nLowercase Text:")
print(lower_text)

print("\nText after removing punctuation:")
print(clean_text)

# 2. Tokenize into sentences and words
sentences = sent_tokenize(paragraph)
words = word_tokenize(clean_text)

print("\nSentence Tokens:")
print(sentences)

print("\nWord Tokens:")
print(words)

# 3. Compare split() and word_tokenize()
split_tokens = lower_text.split()
nltk_tokens = word_tokenize(lower_text)

print("\nTokens using split():")
print(split_tokens)

print("\nTokens using word_tokenize():")
print(nltk_tokens)

print("\nDifference:")
print("split() separates text mainly using whitespace.")
print("word_tokenize() recognizes punctuation as separate tokens.")

# 4. Remove stopwords
stop_words = set(stopwords.words('english'))

filtered_words = [
    word for word in words
    if word not in stop_words and word.isalpha()
]

print("\nWords after removing stopwords:")
print(filtered_words)

# 5. Display word frequency distribution
frequency = FreqDist(filtered_words)

print("\nWord Frequency Distribution:")
for word, count in frequency.most_common():
    print(f"{word}: {count}")


# Expected observation: split() generally leaves punctuation attached to words, whereas NLTK's tokenizer can separate punctuation into individual tokens. Stopword removal excludes common words such as the, and, and for. NLTK's word_tokenize() may require the downloaded sentence-tokenization resources.


# Question 2

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Same paragraph as Q1
paragraph = """
Cricket is my favorite sport because it combines strategy, patience, and teamwork.
I enjoy watching international matches, especially when skilled batters build partnerships under pressure.
Fast bowlers use pace and swing, while spinners trick batters with flight and turn.
A great captain studies the pitch, chooses smart field placements, and motivates every player.
The final overs are exciting because one boundary can change the result.
Cricket teaches discipline, resilience, and respect for opponents.
"""

# 1. Extract alphabetic words using regular expressions
words = re.findall(r'[a-zA-Z]+', paragraph.lower())

print("Alphabetic Words:")
print(words)

# 2. Remove stopwords
stop_words = set(stopwords.words('english'))

filtered_words = [
    word for word in words
    if word not in stop_words
]

print("\nWords after Removing Stopwords:")
print(filtered_words)

# 3. Perform stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

print("\nStemmed Words:")
print(stemmed_words)

# 4. Perform lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [
    lemmatizer.lemmatize(word) for word in filtered_words
]

print("\nLemmatized Words:")
print(lemmatized_words)

# 5. Compare the outputs
print("\nComparison:")
for original, stem, lemma in zip(
    filtered_words, stemmed_words, lemmatized_words
):
    print(f"{original:15} Stem: {stem:12} Lemma: {lemma}")


# Explanations:
# - Stemming: Removes word endings using rules. The result may not be a valid English word; for example, studies may become studi.
# - Lemmatization: Converts a word to a dictionary form using linguistic information. For example, bowlers becomes bowler.
# - Conclusion: Stemming is useful when faster, simpler text normalization is needed. Lemmatization is preferable when readable and linguistically meaningful words are important. WordNet lemmatization can also depend on the part of speech supplied to it.


# Question 3

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Three cricket-related texts
texts = [
    "The batter played attacking cricket. The batter scored a century and helped the cricket team win.",
    "The bowler used swing bowling. The bowler took five wickets with accurate yorkers in the final overs.",
    "The captain led the team calmly. The captain chose smart field placements and motivated the cricket team."
]

# 1. Bag of Words using CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
bow_matrix = count_vectorizer.fit_transform(texts)

print("Bag of Words Vocabulary:")
print(count_vectorizer.get_feature_names_out())

print("\nBag of Words Matrix:")
print(bow_matrix.toarray())

# 2. Calculate TF-IDF scores
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

feature_names = tfidf_vectorizer.get_feature_names_out()

print("\nTF-IDF Matrix:")
print(np.round(tfidf_matrix.toarray(), 3))

# 3. Display top 3 keywords for each text
print("\nTop 3 Keywords for Each Text:")

for i, row in enumerate(tfidf_matrix.toarray()):
    # Sort by descending TF-IDF score
    top_indices = sorted(
        range(len(feature_names)),
        key=lambda j: (-row[j], feature_names[j])
    )[:3]

    print(f"\nText {i + 1}:")
    for index in top_indices:
        print(f"{feature_names[index]}: {row[index]:.4f}")


# Interpretation:
# 
# Bag of Words counts how often each vocabulary word occurs in each text. TF-IDF assigns higher importance to terms that are useful for distinguishing one document from the others, rather than simply counting occurrences. The top three keywords will be printed automatically. Because these are short texts, some terms may receive similar scores; the actual results depend on the vocabulary and all three documents.


# Question 4

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Two technology descriptions, each with 5 lines
ai_text = """
Artificial intelligence enables computers to learn from data.
AI systems recognize patterns and solve complex problems.
Machine learning improves predictions through training.
AI is used in healthcare, education, and automation.
Intelligent systems help people make faster decisions.
"""

blockchain_text = """
Blockchain stores data across a distributed network.
Blocks record transactions in a linked digital ledger.
Cryptography helps keep blockchain records secure.
Blockchain supports digital payments and smart contracts.
Network participants verify transactions without one central authority.
"""

# 1. Preprocess and tokenize both texts
tokens_ai = re.findall(r'[a-zA-Z]+', ai_text.lower())
tokens_blockchain = re.findall(r'[a-zA-Z]+', blockchain_text.lower())

clean_ai = " ".join(tokens_ai)
clean_blockchain = " ".join(tokens_blockchain)

print("AI Tokens:")
print(tokens_ai)

print("\nBlockchain Tokens:")
print(tokens_blockchain)

# 2a. Jaccard Similarity using sets
set_ai = set(tokens_ai)
set_blockchain = set(tokens_blockchain)

intersection = set_ai.intersection(set_blockchain)
union = set_ai.union(set_blockchain)

jaccard = len(intersection) / len(union) if union else 0

print("\nCommon Words:")
print(intersection)

print("\nJaccard Similarity:", round(jaccard, 4))

# 2b. Cosine Similarity using TF-IDF
texts = [clean_ai, clean_blockchain]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

cosine = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:2]
)[0][0]

print("Cosine Similarity:", round(cosine, 4))

# 2c. Compare the metrics
print("\nAnalysis:")
print("Jaccard measures the overlap between unique words.")
print("Cosine similarity compares TF-IDF-weighted text vectors.")

if jaccard < 0.2 and cosine < 0.2:
    print("Both metrics indicate low word-based similarity.")
else:
    print("The metrics show some degree of word-based similarity.")

print(
    "For these descriptions, cosine similarity can provide more "
    "information about term importance, while Jaccard gives a "
    "simple comparison of shared vocabulary."
)


# Analysis: Jaccard similarity considers the shared unique words divided by all unique words in both texts. Cosine similarity compares their TF-IDF vectors, accounting for term weighting. Cosine similarity is often more informative when term importance matters. However, neither metric truly understands meaning: AI and Blockchain could be conceptually related even when they use different vocabulary.


# Question 5

from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Three short product reviews
reviews = [
    "These headphones have excellent sound quality and deep bass. "
    "The battery lasts all day, and the fit is very comfortable.",

    "I love these headphones because they sound amazing. "
    "The delivery was fast, and the product is worth buying.",

    "The headphones feel cheap and the sound keeps cutting out. "
    "The battery life is poor, and customer support was unhelpful.",

    "The headphones connect using Bluetooth and include a charging cable. "
    "They have adjustable ear cups and three control buttons."
]

positive_reviews = []

# 1 and 2. Find polarity, subjectivity, and classify reviews
for i, review in enumerate(reviews, start=1):
    analysis = TextBlob(review)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = "Positive"
        positive_reviews.append(review)
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"\nReview {i}: {review}")
    print(f"Polarity: {polarity:.3f}")
    print(f"Subjectivity: {subjectivity:.3f}")
    print(f"Classification: {sentiment}")

# 3. Generate a word cloud from positive reviews
if positive_reviews:
    positive_text = " ".join(positive_reviews)

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white',
        stopwords=STOPWORDS,
        collocations=False
    ).generate(positive_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Positive Product Reviews")
    plt.show()
else:
    print("No positive reviews were found for the word cloud.")


# Interpretation:
# - Polarity: Ranges from −1 to +1. Negative values suggest negative sentiment, positive values suggest positive sentiment, and values near zero suggest neutral sentiment.
# - Subjectivity: Ranges from 0 to 1. Higher values indicate more subjective or opinion-based language.
# - Word cloud: Displays frequently occurring words from the reviews classified as positive.


# Question 6

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

# 1. Training data: paragraph about cricket (approximately 100 words)
training_text = """
Cricket is a team sport played with a bat and ball.
Two teams compete to score runs and take wickets.
The batter watches the bowler carefully and chooses the right shot.
A fast bowler uses pace and swing, while a spin bowler changes
the flight and turn of the ball.
Fielders work together to stop runs and catch the ball.
The captain makes plans based on the pitch, weather, and match situation.
Good players practise regularly, stay patient, and support their teammates.
In a close match, every run matters.
Cricket teaches teamwork, discipline, confidence, and respect.
Fans enjoy the excitement of a boundary, a wicket, and a final-over victory.
"""

# 2. Tokenize the text using Keras Tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
sentences = [
    sentence.strip()
    for sentence in training_text.lower().split(".")
    if sentence.strip()
]

tokenizer.fit_on_texts(sentences)

total_words = len(tokenizer.word_index) + 1

print("Total unique tokens:", len(tokenizer.word_index))
print("\nWord Index:")
print(tokenizer.word_index)

# 3. Create input sequences
input_sequences = []

for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]

    # Create progressively longer sequences
    for i in range(2, len(token_list) + 1):
        input_sequences.append(token_list[:i])

# Pad sequences to equal length
max_sequence_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding='pre'
)

# Separate predictors and target words
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("\nInput Sequence Shape:", X.shape)
print("Target Shape:", y.shape)

# 4. Build an LSTM model
model = Sequential([
    Input(shape=(max_sequence_len - 1,)),
    Embedding(
        input_dim=total_words,
        output_dim=32
    ),
    LSTM(64),
    Dense(total_words, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Train the model
model.fit(
    X,
    y,
    epochs=150,
    batch_size=16,
    verbose=1
)

# 6. Generate text from a seed word
def generate_text(seed_word, next_words=10):
    current_text = seed_word.lower()

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]

        padded_sequence = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        predictions = model.predict(
            padded_sequence,
            verbose=0
        )[0]

        predicted_id = int(np.argmax(predictions))
        predicted_word = tokenizer.index_word.get(
            predicted_id, ""
        )

        if not predicted_word or predicted_word == "<OOV>":
            break

        current_text += " " + predicted_word

    return current_text.capitalize()

# 7. Generate three lines from three seed words
seed_words = ["cricket", "captain", "bowler"]

print("\nGenerated Text:")
for seed in seed_words:
    print(generate_text(seed, next_words=10))


# Expected observation: The program creates sequences of words, trains an LSTM to predict the next word, and generates three lines beginning with Cricket, Captain, and Bowler.
# 
# The exact generated sentences will vary depending on training and model predictions. Because the training data is small, the generated text may be repetitive or grammatically imperfect.
