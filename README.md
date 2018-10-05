# Azure Machine Learning Bootcamp Materials

Here are the top two resources you'll need this afternoon:

1. [Azure Machine Learning Services documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/)
1. [Azure Machine Learning SDK for Python documentation](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py)

# Challenges

## Challenge 1 - Basic model training on Azure

In this first challenge, you'll be training a basic machine learning model on Azure. We'll be using the popular MNIST dataset, as it allows us to focus on getting familiar with the mechanics of Azure Machine Learning.

![alt text](images/mnist.png "The MNIST dataset")

Tools needed:
* Machine Learning service workspace (preview) - Deploy from Azure Portal
* [Azure Notebooks](https://notebooks.azure.com)
* [`utils.py`](utils.py)

If you are stuck, check out the [hints](hints/challenge_01.md)!

## Challenge 2 - Advanced model training on Azure

In this challenge, you'll be training a more advanced machine learning model on Azure. We'll be using the same dataset, but this time, we'll use Azure Batch AI for speeding up our training.

Tools needed:
* Notebook from Challenge 1

If you are stuck, check out the [hints](hints/challenge_02.md)!

## Challenge 3 - Model deployment on Azure

In this third challenge, you'll be taking the model you've trained in the first or second challenge and deploy it to Azure Container Instances (ACI). Alternatively, you can deploy it to Azure Kubernetes Service (AKS).

If you are stuck, check out the [hints](hints/challenge_03.md)!

## Challenge 4 - Automated Machine Learning

In this last challenge, you'll be using Automated Machine Learning to let Azure figure out which machine learning algorithm performs best on your dataset.

If you are stuck, check out the [hints](hints/challenge_04.md)!

## Further Challenges

* Secure our model endpoint on ACI with an Authentication Key
* Embed Application Insights into our model for monitoring