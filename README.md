# Cyberbullying-Detection-Models

Collection of various models for detecting cyberbullying

## Dataset

The dataset used for training and evaluating the models is sourced from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification). It contains more than 47000 tweets labeled with different types of cyberbullying, including:

- Age
- Ethnicity
- Gender
- Religion
- Other type of cyberbullying
- Not cyberbullying

The data has been balanced in order to contain ~8000 of each class.

## About Models

We can try developing various models for detecting cyberbullying, including machine learning and deep learning models. Here I target to use transfer learning models like BERT, DistilBERT, and RoBERTa. But other models can also be tried.

## Setup

First, You need to clone the repository or download it as a ZIP file, then navigate to the project directory. In order to run the code, you need to have Python installed on your system. It is recommended to create a virtual environment for the project. For more details refer to the [ENV.md](ENV.md) file. Also, you need to install the required packages listed in the `requirements.txt` file. Since there are several models listed in here, you may not need all the packages. You can install the packages as per your requirement. You can see relevant requirements file for each model in the respective folders. Since models are too large to be stored in the repository, you need to download the models from Hugging Face or other sources as mentioned in this README.