import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis Model Evaluation')
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument('--rnn', action='store_true', help='Evaluate RNN model')
    model_group.add_argument('--lstm', action='store_true', help='Evaluate LSTM model')
    args = parser.parse_args()

    # Set the hyperparameters
    vocab_size = 10000
    max_sequence_length = 300
    embedding_dim = 16
    hidden_dim = 64
    learning_rate = 1e-3
    batch_size = 32
    epochs = 20

    # Preprocess the data
    print("Loading and preprocessing the dataset...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data(vocab_size, max_sequence_length)

    if args.rnn or (not args.rnn and not args.lstm):
        # Build the RNN model
        model_rnn = build_rnn_model(vocab_size, embedding_dim, hidden_dim)

        # Train and evaluate the RNN model
        print("Evaluating RNN Model")
        train_accuracy_rnn, test_accuracy_rnn = train_and_evaluate_model(model_rnn, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate)

        print('RNN Model:')
        print('Training Accuracy:', train_accuracy_rnn)
        print('Test Accuracy:', test_accuracy_rnn)

        # Evaluate on longer sequences for the RNN model
        accuracy_rnn_longer_sequences = evaluate_longer_sequences(model_rnn, X_test, y_test, num_samples=100)
        print('Accuracy on Longer Sequences (RNN):', accuracy_rnn_longer_sequences)

    if args.lstm or (not args.rnn and not args.lstm):
        # Build the LSTM model
        model_lstm = build_lstm_model(vocab_size, embedding_dim, hidden_dim)

        # Train and evaluate the LSTM model
        print("Evaluating LSTM Model")
        train_accuracy_lstm, test_accuracy_lstm = train_and_evaluate_model(model_lstm, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate)

        print('LSTM Model:')
        print('Training Accuracy:', train_accuracy_lstm)
        print('Test Accuracy:', test_accuracy_lstm)

        # Evaluate on longer sequences for the LSTM model
        accuracy_lstm_longer_sequences = evaluate_longer_sequences(model_lstm, X_test, y_test, num_samples=100)
        print('Accuracy on Longer Sequences (LSTM):', accuracy_lstm_longer_sequences)




def load_and_preprocess_data(vocab_size, max_sequence_length):
    # Load the IMDb reviews dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    # Pad sequences to have the same length
    X_train = pad_sequences(X_train, maxlen=max_sequence_length)
    X_test = pad_sequences(X_test, maxlen=max_sequence_length)

    return X_train, y_train, X_test, y_test

def build_rnn_model(vocab_size, embedding_dim, hidden_dim):
    # Build simple RNN with one hidden layer
    # and hyperparameters specified in the doc

    model = keras.Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(SimpleRNN(units=hidden_dim, activation='tanh'))
    model.add(Dense(units=1, activation='sigmoid'))

    return model

def build_lstm_model(vocab_size, embedding_dim, hidden_dim):
    # Build LSTM Network with one hidden layer
    # and hyperparameters specified in the doc

    model = keras.Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(LSTM(units=hidden_dim, activation='tanh'))
    model.add(Dense(units=1, activation='sigmoid'))

    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate):
    # Must compile the model before fitting it
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Fit model to training data so we can evaluate on test set
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # Report accuracy for train and test
    _, train_accuracy = model.evaluate(X_train, y_train)
    _, test_accuracy = model.evaluate(X_test, y_test)

    return train_accuracy, test_accuracy

def evaluate_longer_sequences(model, X, y, num_samples=100):
    # Find the number of samples with the longest sentences by looking at the ones with the least padding
    sorted_indices = sorted(range(len(X)), key=lambda i: len(X[i]))
    longest_samples = sorted_indices[-num_samples:]

    # Pad the longest samples to the same length as the original sequences
    longest_sequences = pad_sequences(X[longest_samples], maxlen=X.shape[1])

    # Evaluate the model on the longest sequences
    accuracy = model.evaluate(longest_sequences, y[longest_samples], verbose=0)[1]

    return accuracy


if __name__ == "__main__":
    main()


