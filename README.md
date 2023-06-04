# Sentiment Analysis Model Evaluation

This repository contains code for evaluating RNN and LSTM models on the IMDb reviews dataset for sentiment analysis.

## Usage

1. Clone the repository:

```shell
git clone https://github.com/lodriscoll/sentiment-analysis.git
```

2. Navigate to the project directory:

```shell
cd sentiment-analysis
```

3. Install the required dependencies. It is recommended to set up a virtual environment before installing the dependencies:

```shell
python3 -m venv venv
source venv/bin/activate  # Activate the virtual environment
pip install -r requirements.txt
```

4. Run the evaluation script:

```shell
python evaluate_models.py [--rnn] [--lstm]
```

- The `--rnn` flag evaluates the RNN model.
- The `--lstm` flag evaluates the LSTM model.
- If no flags are provided, both models are evaluated by default.

5. View the evaluation results:

The script will display the training accuracy and test accuracy for each model, as well as the accuracy on longer sequences.

## Hyperparameters

The hyperparameters of the models are set as follows:

- Vocabulary size: 10,000
- Maximum sequence length: 300
- Embedding dimension: 16
- Hidden dimension: 64
- Learning rate: 0.001
- Batch size: 32
- Number of epochs: 20

Feel free to modify these hyperparameters in the `evaluate_models.py` script to experiment with different configurations.

## Dataset

The IMDb reviews dataset is used for sentiment analysis. It consists of movie reviews labeled with positive or negative sentiment. The dataset is split into a training set and a test set.

The `load_and_preprocess_data` function preprocesses the dataset by loading it and performing sequence padding to ensure all sequences have the same length.

## Models

Two models are implemented for sentiment analysis:

1. Simple RNN model:
   - Embeds the input text as a sequence of vectors
   - Transforms the sequence of embeddings into a vector using a single-layer, simple RNN
   - Applies a feed-forward layer to obtain a label

2. LSTM model:
   - Embeds the input text as a sequence of vectors
   - Transforms the sequence of embeddings into a vector using a single-layer LSTM
   - Applies a feed-forward layer to obtain a label

Both models are trained and evaluated using the same hyperparameters.

## Evaluation on Longer Sequences

The code also includes functionality to evaluate the models on longer sequences. The `evaluate_longer_sequences` function finds the number of samples with the longest sentences and evaluates the model's accuracy on these samples.

## References

- IMDb reviews dataset: [Link](https://ai.stanford.edu/~amaas/data/sentiment/)
- Keras documentation: [Link](https://www.tensorflow.org/api_docs/python/tf/keras)
- TensorFlow documentation: [Link](https://www.tensorflow.org/api_docs/python/tf)