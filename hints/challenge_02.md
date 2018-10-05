# Hints for Challenge 2

Our model in challenge 1 had an accuracy of ~92%. For the MNIST dataset, this is not very good. In order to train a more powerful and complex model, we'll need more compute. Hence, instead of training locally in our Azure Notebook, we'll be using [Azure Batch AI](https://azure.microsoft.com/en-us/services/batch-ai/) to train our model on a dedicated compute cluster.

Firstly, let's create a new notebook `challenge02.ipynb` for this challenge.

As before, let's connect to our Azure ML Workspace and reference our existing experiment:

```python
from azureml.core import Workspace, Experiment, Run
ws = Workspace.from_config()

experiment = Experiment(workspace = ws, name = "scikit-learn-mnist")
```

We should still have our MNIST dataset sitting in the `./data/` folder from challenge 1, but just in case, we'll download it again:

```python
import os
import urllib.request

os.makedirs('./data', exist_ok = True)

urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename='./data/train-images.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename='./data/train-labels.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename='./data/test-images.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename='./data/test-labels.gz')
```

Since we'll need to access that MNIST dataset from our Batch AI cluster, we'll upload it to the datastore that our Azure ML Workspace provides for us. This code will request the default datastore (Azure Files) and upload the 4 files from MNIST into the `./mnist` folder:

```python
ds = ws.get_default_datastore()

print("Datastore details:")
print(ds.datastore_type, ds.account_name, ds.container_name)

ds.upload(src_dir='./data', target_path='mnist', overwrite=True, show_progress=True)
```

If we go to the default Storage Account that the Azure ML Workspace created for us, and go to Azure Files, we can see that the dataset has been uploaded:

![alt text](../images/02-dataset_in_azure_files.png "MNIST dataset in Azure Files")

Next, we can create an empty Batch AI cluster in Azure:

```python
from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
batchai_cluster_name = "traincluster"

try:
    # look for the existing cluster by name
    compute_target = ComputeTarget(workspace=ws, name=batchai_cluster_name)
    if type(compute_target) is BatchAiCompute:
        print('found compute target {}, just use it.'.format(batchai_cluster_name))
    else:
        print('{} exists but it is not a Batch AI cluster. Please choose a different name.'.format(batchai_cluster_name))
except ComputeTargetException:
    print('creating a new compute target...')
    compute_config = BatchAiCompute.provisioning_configuration(vm_size="STANDARD_D2_V2", # small CPU-based VM
                                                                #vm_priority='lowpriority', # optional
                                                                autoscale_enabled=True,
                                                                cluster_min_nodes=1, 
                                                                cluster_max_nodes=1)

    # create the cluster and wait until it has been provisioned
    compute_target = ComputeTarget.create(ws, batchai_cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
    # Get the status of our cluster
    print(compute_target.status.serialize())
```

If we now look under the `Compute` tab, in our Azure ML Workspace, we can see our Batch AI cluster:

![alt text](../images/02-create_cluster.png "Create our Batch AI cluster for training")

Currently, we have all our code in our Azure Notebook. Obviously, our Batch AI cluster needs to somehow get the Python code for reading the data and training our model. Hence, we create a `scripts` folder and put our training Python code in it:

```python
import os, shutil

script_folder = './scripts'
os.makedirs(script_folder, exist_ok=True)
shutil.copy('utils.py', script_folder)
```

Let this cell just write the `train.py` to the `scripts` folder (we could have created it manually and copied it in):

```python
%%writefile $script_folder/train.py

import argparse
import os
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from azureml.core import Run
from utils import load_data

# input image dimensions and number of classes
img_rows, img_cols = 28, 28
num_classes = 10

# let user feed in parameters
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128, help='batch size')
parser.add_argument('--epochs', type=int, dest='epochs', default=12, help='number of epochs')

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
data_folder = os.path.join(args.data_folder, 'mnist')

print('Data folder:', data_folder)

# load train and test set into numpy arrays and scale
x_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
x_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# get hold of the current run
run = Run.get_submitted_run()

# Design our Convolutional Neural Network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

acc = np.float(score[1])
print('Accuracy is', acc)

# Log accuracy to our Azure ML Workspace
run.log('accuracy', acc)

# Save model, the outputs folder is automatically uploaded into experiment record by Batch AI
os.makedirs('outputs', exist_ok=True)
model.save('./outputs/keras-tf-mnist.h5')
```

Lastly, we need to package the scripts and "send" them to Batch AI. Azure ML uses the `Estimator` class for that:

```python
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': ds.as_mount(),
    '--batch-size': 128,
    '--epochs': 10
}

est = Estimator(source_directory=script_folder,
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py',
                conda_packages=['keras'])
```

Lastly, we can kick off the job (this will take a while):

```python
run = experiment.submit(config=est)
run
```

Submitting the job is an asynchronous call, hence we'll have to keep checking the status (the widget will auto-refresh):

```python
from azureml.train.widgets import RunDetails
RunDetails(run).show()
```

In the background, Azure ML will now perform the following steps:

* Package our scripts as a Docker image and push it to our Azure Container Registry (initially this will take ~10 minutes)
* Scale up the Batch AI cluster (if required)
* Pull the Docker image to the Batch AI cluster
* Mount the MNIST data from Azure Files to the Batch AI cluster (for fast local access)
* Start the training job
* Publish the results to our Workspace (same as before)

The first run might take 10-20 minutes. Subsequent runs will be significantly faster as the base Docker image will be ready and already pushed.

In the Batch AI Experiments section, we can see our run:

![alt text](../images/02-batch_ai_experiment.png "Batch AI Experiment run")

With the same code as before (this is the strength of Azure ML), we can retrieve the results of our training run:

```python
print("Run metrics:", run.get_metrics())
print("Run model files", run.get_file_names())
```

Same code for registering our mode:

```python
model = run.register_model(model_name='scikit-learn-mnist-model', model_path='outputs/scikit-learn-mnist.pkl')
print(model.name, model.id, model.version, sep = '\t')
```

And if we want, we can delete our Batch AI cluster:

```python
compute_target.delete()
```

At this point (in addition to the results from challenge 1):

* We used Azure ML to leverage Azure Batch AI to train a Convolutional Neural Network (CNN)
* We switched our training framework from Scikit-learn to Keras with TensorFlow in the backend (without changing any Azure ML code!)
* We registered our new model (>99% accuracy) in our Azure ML Workspace

