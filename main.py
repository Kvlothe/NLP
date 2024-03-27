import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Initialize an empty DataFrame
df = pd.DataFrame(columns=['sentence', 'sentiment'])

# Define file paths
files = ['yelp_labelled.txt', 'imdb_labelled.txt', 'amazon_cells_labelled.txt']

# Loop through each file
for file_path in files:
    with open(file_path, 'r') as file:
        for line in file:
            # Strip newline and whitespace characters
            line = line.strip()
            # Check the last character for sentiment, assuming a space before the last character
            sentiment = 'positive' if line[-1] == '1' else 'negative'
            # Store the sentence without the last two characters (space and label)
            sentence = line[:-2].strip()
            # Append to the DataFrame
            df = df._append({'sentence': sentence, 'sentiment': sentiment}, ignore_index=True)

# Now df contains all your sentences along with their corresponding sentiment labels
# Print out first 5 columns/rows to make sure DF was created correctly
print(df.head())

# Check for any missing values/rows
print("Missing Values")
missing_values_count = df.isna().sum()
missing_values_count = missing_values_count[missing_values_count > 0]
print(missing_values_count)

# Get a count of the dataset
print(df.count)

# Vertorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['sentence'])
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Neural Network
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluation / Metrics
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

