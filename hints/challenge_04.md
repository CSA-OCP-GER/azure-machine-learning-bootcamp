# Hints for Challenge 4

By now, we have a good understanding how Azure Machine Learning works. In this last challenge, we'll take a data set and use Automated Machine Learning for testing out different regression algorithms automatically. Automated Machine Learning is currently able to perform `classification`, `regression` and also `forecasting`.

For this challenge, we'll be using the [Boston house prices dataset](http://scikit-learn.org/stable/datasets/index.html#boston-dataset).

Let's create a new notebook called `challenge_04.ipynb`. As always, include our libraries and connect to our Workspace:

```python
import logging

import numpy as np
from sklearn import datasets

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

ws = Workspace.from_config()
```

```python
experiment_name = 'automl-local-regression'
project_folder = './automl-local-regression'

experiment = Experiment(ws, experiment_name)
```

Let's load our dataset and split it into a train and test set (this time, we didn't get pre-prepared data sets):

```python
from sklearn.datasets import load_boston

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

Now, we need to configure our Automated Machine Learning project:

```python
automl_config = AutoMLConfig(task = 'regression',
                             iteration_timeout_minutes = 2,
                             iterations = 10,
                             primary_metric = 'normalized_root_mean_squared_error',
                             n_cross_validations = 5,
                             debug_log = 'automl.log',
                             verbosity = logging.INFO,
                             X = X_train, 
                             y = y_train,
                             path = project_folder)
```

This is the most interesting part:

* `task` - Type of problem (classification or regression)
* `primary_metric` - The metric that we want to optimize
* `iteration_timeout_minutes` - Time limit per iteration
* `iterations` - Number of iterations (number of algorithms tested)
* `n_cross_validations` - Number of cross validation splits
* `X` - Training data
* `y` - Training labels
* `path` - Output for configuration files

Depending on your task, there are [a lot more](https://docs.microsoft.com/en-us/python/api/azureml-train-automl/azureml.train.automl.automlconfig(class)?view=azure-ml-py) configuration parameters!

Let's run it locally in our Notebook, as the data isn't too big (depending on the complexity of our data, we would want to use Batch AI again):

```python
local_run = experiment.submit(automl_config, show_output = True)
```

![alt text](../images/04-test_iterations.png "Test iterations")

Once done, we can retrieve the best performing model (in this case, it is a [Pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-ml-pipelines)):

```python
best_run, fitted_model = local_run.get_output()
print("Best run:", best_run)
print("Best model:", fitted_model)
```

We can use the best model/pipeline, to score our test data and calculate our accuracy:

```python
y_pred = fitted_model.predict(X_test)

sum_actuals = sum_errors = 0
for actual_val, predict_val in zip(y_test, y_pred):
    abs_error = actual_val - predict_val
    if abs_error < 0:
        abs_error = abs_error * -1

    sum_errors = sum_errors + abs_error
    sum_actuals = sum_actuals + actual_val

mean_abs_percent_error = sum_errors / sum_actuals
print("Model MAPE", mean_abs_percent_error)
print("Model Accuracy", 1 - mean_abs_percent_error)
```

From here, we can have a look at the [following examples](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/automated-machine-learning) to see how we can save and deploy the model.

At this point:

* We took the `Boston house prices dataset` and split it up into a train and test set
* We let Automated Machine Learning evaluate 10 algorithms to predict the house prices in Boston
* We picked the best performing model and ran it against the test dataset to get a final accuracy
* There are a [lot more possibilities](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train) with Automated Machine Learning, especially `Forecasting` is also supported
* A lot more example notebooks for Azure Machine Learning can be found [here on GitHub](https://github.com/Azure/MachineLearningNotebooks)
