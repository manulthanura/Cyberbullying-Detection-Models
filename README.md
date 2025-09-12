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

## Setup

First, You need to clone the repository or download it as a ZIP file, then navigate to the project directory. In order to run the code, you need to have Python installed on your system. It is recommended to create a virtual environment for the project. For more details refer to the [ENV.md](ENV.md) file. Also, you need to install the required packages listed in the `requirements.txt` file. Since there are several models listed in here, you may not need all the packages. You can install the packages as per your requirement. You can see relevant requirements file for each model in the respective folders. Since models are too large to be stored in the repository, you need to download the models from Hugging Face or other sources as mentioned in this README.

## About Models

We can try developing various models for detecting cyberbullying, including machine learning and deep learning models. Here I target to use transfer learning models like BERT, DistilBERT, and RoBERTa. But other models can also be tried.

### Machine Learning Models

Will be updated soon.

### Deep Learning Models

Will be updated soon.

### Transfer Learning Models

1. **[BERT](BERT/)**: This model uses the BERT architecture for text classification. The model is fine-tuned on the cyberbullying dataset to classify tweets into different categories of cyberbullying.


2.  **[Gemma](Gemma/)**: This model uses the Gemma architecture for text classification. The model is fine-tuned on the cyberbullying dataset to classify tweets into different categories of cyberbullying.
    - Model available at: [manulthanura/Gemma-3-270m-Cyberbullying-Classifier](https://huggingface.co/manulthanura/Gemma-3-270m-Cyberbullying-Classifier)
    - To use this model, you need to have a Hugging Face account and generate an access token. You can follow the instructions [here](https://huggingface.co/docs/hub/security-tokens) to create a token. Once you have the token, you can set it as an environment variable or directly use it in the code.
    - There are two files in the Gemma folder:
      - `gemma.py`: This file contains the code to load the model and make predictions. You need to replace the `hf_token` variable with your Hugging Face access token. Also file loads the model from local Models folder. You need to download the model manually from the above link and place it in the `Models` folder.
      - `hf.py`: This file contains the code to load the model directly from Hugging Face using the access token. You can use this file if you don't want to download the model manually.