# Data
import pandas as pd
import numpy as np

# Machine Learning
import keras
import tensorflow as tf
import ml_edu.experiment
import ml_edu.results
import plotly.express as px
from plotly.graph_objs.indicator import number

#loading dataset
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Updating dataframe to use specific columns
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print('Read dataset completed successfully.')
print('Total number of rows in the dataset: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)

# knowing your data
# Answering the following questions will help you understand your data better. You can use the describe() function to get a summary of the dataset.
training_df.describe(include='all')
answer = '''
  What is the maximum fare?
  What is the mean distance across all trips?
  How many cab companies are in the dataset?
  What is the most frequent payment type?
  Are any features missing data?
'''

#What is the maximum fare?
max_fare = training_df['FARE'].max()
print("What is the maximum fare?  Answer: ${0:.2f}".format(max_fare))

# What is the mean distance across all trips?
mean_distance = training_df['TRIP_MILES'].mean()
print("What is the mean distance across all trips? 		Answer: {mean:.4f} miles".format(mean = mean_distance))

# How many cab companies are in the dataset?
num_unique_companies =  training_df['COMPANY'].nunique()
print("How many cab companies are in the dataset? 		Answer: {number}".format(number = num_unique_companies))

# What is the most frequent payment type?
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("What is the most frequent payment type? 		Answer: {type}".format(type = most_freq_payment_type))

# Are any features missing data?
missing_values = training_df.isnull().sum().sum()
print("Are any features missing data? 				Answer:", "No" if missing_values == 0 else "Yes")


# Correlation analysis
# Using a correlations matrix, we can see how the features are related to the label, which feature relates which
# and if they are or not linearly correlated.
print(training_df.corr(numeric_only = True))

#Visualize the relatonships in the dataset:
# visualizing the relationships between the features and the label can help us understand how they are related to each other. We can use scatter plots to visualize the relationships between the features and the label.
fig = px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
fig.show()

#Defne the function to create and train the model

def create_model(
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """Create and compile a simple linear regression model."""""
    # Describe the topography of the model.
    # The topography of a simple linear regression model is a single node in a single layer
    # 1. Create an "entry slot" (placeholder) for each feature in settings.input_features.
    # shape=(1,) tells Keras each feature is a single number per taxi ride (e.g., 8.5 miles).
    # Using a dictionary comprehension makes it flexible for 1 or multiple features.
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}

    # 2. Bundle all the separate feature inputs side-by-side into a single vector
    # (e.g., [TRIP_MILES, TRIP_MINUTES]) so they can be passed to the math layer together.
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))

    # 3. Perform the Linear Regression math: y' = b + w1*x1 + w2*x2 ...
    # A Dense layer with units=1 is a single calculation node that outputs 1 number (predicted FARE).
    # Under the hood, it automatically creates a Weight (w) for each input feature and one Bias (b).
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)

    # 4. Package the starting inputs and final output into a complete Keras Model object
    # so it can be compiled and trained.
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model topography into code that Keras can efficiently execute.
    # Uses RMSprop (gradient descent optimizer) to minimize Mean Squared Error (MSE) loss.
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
        loss="mean_squared_error",
        metrics=metrics
    )

    return model

def train_model(
        experiment_name: str,
        model: keras.Model,
        dataset: pd.DataFrame,
        label_name: str,
        settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(x=features, y=label, batch_size=settings.batch_size, epochs=settings.number_epochs)

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )

print("SUCCESS: defining linear regression functions complete.")

# Training the model with one feature
# The following variables are the hyperparameters.
settings_1 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size =50,
    input_features = ['TRIP_MILES']
)

settings_2 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 200,
    batch_size = 500,
    input_features = ['TRIP_MILES']
)

# Training model with 2 input features
settings_3 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES', 'TRIP_MINUTES']
)

# Adding a new key value pair in the training dataframe: trip minutes which is trip second/60
training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

# Creating model_1 with the hyperparameter settings and error metrics
model_1 = create_model(settings_1, metrics)

experiment_1 = train_model('one_feature', model_1, training_df, 'FARE', settings_1)

ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_1, training_df, 'FARE')


# Creating model 2 and experimenting with different settings
model_2 = create_model(settings_2, metrics)

experiment_2 = train_model('one_feature_hyper', model_2, training_df, 'FARE', settings_2)

ml_edu.results.plot_experiment_metrics(experiment_2, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_2, training_df, 'FARE')


# Creating model 3 and experimentng with 3 input features
model_3 = create_model(settings_3, metrics)

experiment_3 = train_model('two_inputs', model_3, training_df, 'FARE', settings_3)

ml_edu.results.plot_experiment_metrics(experiment_3, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_3, training_df, 'FARE')