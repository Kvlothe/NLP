import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
# import scipy

# print(np.__version__)
# print(scipy.__version__)


# Function for detecting non-english words and symbols
def contains_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric shapes extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental arrows-C
                               u"\U0001FA00-\U0001FA6F"  # Chess symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return bool(emoji_pattern.search(text))


# Function for cleaning the text, lower casing all text, removing punctuation and tokenizing the data,
# removing stopwords and non-alphabetic tokens
def clean_text(text):
    # Lowercase the text
    text = text.lower()

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    return tokens


# Initialize an empty DataFrame
df = pd.DataFrame(columns=['sentence', 'sentiment'])

# Define file paths
files = ['yelp_labelled.txt', 'imdb_labelled.txt', 'amazon_cells_labelled.txt']

# Stopwords in English
stop_words = set(stopwords.words('english'))

# Loop through each file
# Initialize an empty list to store the rows
rows_list = []
for file_path in files:
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            sentiment = 'positive' if line[-1] == '1' else 'negative'
            sentence = line[:-2].strip()
            # Create a new row as a dictionary
            new_row = {'sentence': sentence, 'sentiment': sentiment}
            # Append the new row dictionary to the list
            rows_list.append(new_row)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(rows_list)

# Now df contains all your sentences along with their corresponding sentiment labels
# Print out first 5 columns/rows to make sure DF was created correctly
print(df.head())

# Check for any missing values/rows
print("Missing Values")
missing_values_count = df.isna().sum()
missing_values_count = missing_values_count[missing_values_count > 0]
print(missing_values_count)

# Get a count of the dataset
# print(df.count)

# Apply the contains_emoji function to each row in the 'sentence' column
df['contains_emoji'] = df['sentence'].apply(lambda x: contains_emoji(x))
print(df.head())
true_count = df['contains_emoji'].sum()
print(f"Number of rows with emojis: {true_count}")

# Apply the function clean_text to your dataframe
df['cleaned_text'] = df['sentence'].apply(clean_text)

all_words = [word for tokens in df['cleaned_text'] for word in tokens]
word_counts = Counter(all_words)

# Vocabulary size
vocab_size = len(word_counts)
print(f"Vocabulary size: {vocab_size}")

# Plot the distribution of lengths
sequence_lengths = [len(tokens) for tokens in df['cleaned_text']]
plt.hist(sequence_lengths, bins=30)
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()

# Choosing a maximum sequence length
avg_length = np.mean(sequence_lengths)
std_length = np.std(sequence_lengths)
max_length = int(avg_length + 2 * std_length)
print(f"Suggested maximum sequence length: {max_length}")

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df['sentence'])

# Convert sentences to sequences of integers
sequences = tokenizer.texts_to_sequences(df['sentence'])

# Pad sequences to ensure uniform length
X_padded = pad_sequences(sequences, maxlen=max_length, padding='post')
print(X_padded[:5])
clean_df = pd.DataFrame(X_padded)
clean_df.to_csv('preped_data.csv', index=False)

# Convert 'sentiment' to numerical labels
# 'positive' -> 1, 'negative' -> 0
df['sentiment_label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
y = df['sentiment_label'].values
df.to_csv("prepared_data.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the model's validation loss
                               patience=3,  # Stop after 3 epochs if the validation loss hasn't improved
                               restore_best_weights=True)  # Restore model weights from the epoch with the best validation loss

# Train the Neural Network
# Adjusting the regularization strength and dropout rate
model = Sequential()
model.add(Input(shape=(max_length,)))
model.add(Embedding(input_dim=vocab_size, output_dim=50))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001)))  # Adjusted regularization
model.add(MaxPooling1D(pool_size=5))
model.add(Flatten())
model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.001)))  # Adjusted regularization
model.add(Dropout(0.3))  # Adjusted dropout
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],  # Include the early stopping callback
                    verbose=2)

loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {accuracy}")

model.summary()

# Plot metrics to view Training and validation accuracy
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# Plot metrics to view training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.save('classification.h5')

