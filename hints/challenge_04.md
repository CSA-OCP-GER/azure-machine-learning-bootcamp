# Hints for Challenge 4

By now, we have a good understanding how Azure Machine Learning works. In this last challenge, we'll take a data set and use Automated Machine Learning for testing out different regression algorithms automatically.

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
                             max_time_sec = 120,
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
* `max_time_sec` - Time limit per iteration
* `iterations` - Number of iterations (number of algorithms tested)
* `n_cross_validations` - Number of cross validation splits
* `X` - Training data
* `y` - Training labels
* `path` - Output for configuration files

Depending on your task, there are [a lot more](https://docs.microsoft.com/en-us/python/api/azureml-train-automl/azureml.train.automl.automlconfig(class)?view=azure-ml-py) configuration parameters!

Let's run it locally in our Notebook (not a smart idea for larger datasets):

```python
local_run = experiment.submit(automl_config, show_output = True)
```

We can summarize the results graphically:

```python
from azureml.train.widgets import RunDetails
RunDetails(local_run).show()
```

And also retrieve the best performing model:

```python
best_run, fitted_model = local_run.get_output()
print("Best run:", best_run)
print("Best model:", fitted_model)
```

From here on, we can perform the same steps are before:

1. Register the model
1. Containerize it
1. Deploy it to a production target of our choice

At this point:

* We took the `Boston house prices dataset` and split it up into a train and test set (we haven't used the test set in our code though!)
* We let Automated Machine Learning evaluate 10 algorithms to predict the house prices in Boston
* We picked the best performing model
