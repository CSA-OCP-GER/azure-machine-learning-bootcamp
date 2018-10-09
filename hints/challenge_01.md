# Hints for Challenge 1

## Setup part

In the Azure Portal, create a new `Machine Learning service workspace (preview)` Service:

* Workspace name: `azure-ml-bootcamp`
* Resource Group: `azure-ml-bootcamp`
* Location: `West Europe` (but does not really matter)

![alt text](../images/01-create_workspace.png "Create Machine Learning Workspace")

Let's have a look at our Resource Group:

![alt text](../images/01-resource_group.png "Our resource group")

* Application Insights - used for monitoring our models in production
* Storage account - this will store our logs, model outputs, etc.
* Key vault - stores our secrets
* Container registry - stores our containerized models
* Machine Learning service workspace - the centre point for Machine Learning on Azure

Create a free account on [Azure Notebooks](https://notebooks.azure.com). Then, create a new notebook (no need to make it `Public`):

![alt text](../images/01-create_notebook.png "Our new Azure Notebook for our code")

Inside the newly created Azure Notebook, create a `config.json` and replace the values with your own (you'll find your Subscription ID in the Azure Portal at the top of your Resource Group):

```json
{
    "subscription_id": "xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxx",
    "resource_group": "azure-ml-bootcamp",
    "workspace_name": "azure-ml-bootcamp"
}
```

The `config.json` is used by the Azure Machine Learning SDK to connect to your ML workspace.

Download [`utils.py`](../utils.py) and upload it into your notebook.

Finally, we can create a new Python Notebook where we'll run our code in:

![alt text](../images/01-create_ipynb.png "Our new Python Notebook")

## Training a basic Machine Learning model

Inside your `challenge01.ipynb` notebook, create a new cell:

```python
from azureml.core import Workspace, Experiment, Run

ws = Workspace.from_config()
```

You can run the cell by hitting `Run` or pressing `Shift+Enter`. This cell imports the relevant libraries from the Azure Machine Learning SDK, reads our `config.json` and connects the notebook to our Machine Learning Workspace in Azure.

Next, let's create a new experiment (this will later show up in our ML Workspace in Azure). This is where all our experiments will be logged to:

```python
experiment = Experiment(workspace = ws, name = "scikit-learn-mnist")
```

Let's load some test data into our notebook. In a later challenge, we'll use Azure Batch AI to train a more powerful model, but for now, we'll train a model with the CPU that is running our notebook:

```python
import os
import urllib.request

os.makedirs('./data', exist_ok = True)

urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename='./data/train-images.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename='./data/train-labels.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename='./data/test-images.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename='./data/test-labels.gz')
```

Let's create a fourth cell for training our model. Make sure that `utils.py` is in the notebook. In case you've forgot to include `utils.py` or added it after you've stared the notebook, you'll need to restart your notebook (via `Kernel --> Restart`) and re-run all the cells.

```python
from utils import load_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

# We need to scale our data to values between 0 and 1
X_train = load_data('./data/train-images.gz', False) / 255.0
y_train = load_data('./data/train-labels.gz', True).reshape(-1)
X_test = load_data('./data/test-images.gz', False) / 255.0
y_test = load_data('./data/test-labels.gz', True).reshape(-1)

# Tell our Azure ML Workspace that a new run is starting
run = experiment.start_logging()

# Create a Logistic Regression classifier and train it
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict classes of our testing dataset
y_pred = clf.predict(X_test)

# Calculate accuracy
acc = np.average(y_pred == y_test)
print('Accuracy is', acc)

# Log accuracy to our Azure ML Workspace
run.log('accuracy', acc)

# Tell our Azure ML Workspace that the run has completed
run.complete()
```

This code does the following things:

1. It imports `sklearn` (Scikit-Learn) as the Machine Learning framework
1. It loads our MNIST train and test data, and scales the values within `[0, 1]`
1. It tells our Azure ML Experiment to start logging a training run
1. It creates a `LogisticRegression`-based classifier and trains it using the training data
1. It then uses the classifier to predict the numbers in the test dataset
1. It compares the predictions to the ground truth and calculated the accuracy score
1. Finally, it logs to accuracy to our run and then finishes the run

As we can see, our model achieves `92%` accuracy, which is pretty low for the MNIST dataset - we'll get back to this in the next challenge!

In the Azure ML Workspace, we can see that our experiment is now showing up:

![alt text](../images/01-running_experiment.png "Our first experiment")

Inside our experiment, we can see our first run:

![alt text](../images/01-runs.png "Viewing our runs")

If we click the run number, we can see its details:

![alt text](../images/01-run_details.png "Run Details")

Here we could log more values or even time series, which would directly show up as diagrams. However, as we want to keep the code short, we'll skip this part for now.

Finally, we can export our model and upload it to our Azure ML Workspace:

```python
from sklearn.externals import joblib

# Write model to disk
joblib.dump(value=clf, filename='scikit-learn-mnist.pkl')

# Upload our model to our experiment
run.upload_file(name = 'outputs/scikit-learn-mnist.pkl', path_or_stream = './scikit-learn-mnist.pkl')
```

In the portal, we can see the output of our run:

![alt text](../images/01-run_model.png "Our model output in our ML Workspace")

We can also query the metrics and outputs for our run:

```python
print("Run metrics:", run.get_metrics())
print("Run model files", run.get_file_names())
```

As a last step, we can register our model in our workspace:

```python
model = run.register_model(model_name='scikit-learn-mnist-model', model_path='outputs/scikit-learn-mnist.pkl')
print(model.name, model.id, model.version, sep = '\t')
```

Under the `Models` tab, we can now see that our model has been registered:

![alt text](../images/01-registered_model.png "Our model has been registered")

Our model has been stored in the Storage Account that has been created initially for us:

![alt text](../images/01-model_in_blob.png "Our model has been stored in Azure Blob")

At this point:

* We've trained a Machine Learning model using Scikit-Learn inside our Azure Notebook
* We achieved `92%` accuracy (not very good for this data set)
* Azure ML knows about our experiment and our initial run
* Azure ML has the output files of our trained model in Blob storage
* We have registered our initial model as a Azure ML Model in our Workspace

In the [next challenge](challenge_02.md), we'll build a more powerful model and use Azure Batch AI to train it on a remote cluster.