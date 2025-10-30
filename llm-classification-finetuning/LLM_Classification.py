## Predict human preference between two response from two large language models (LLMs) given the same prompt
# Data:
# Prompt, response A, response B, label indicating which response the human user preferred (or none/tie)
# Reinforcement Learning from Human Feedback (RLHF)

## load the data and check is any parsing / cleaning is required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("C:/Users/Shreya Tanguturi/Desktop/PythonPersonalProjects/llm-classification-finetuning/train.csv")
test = pd.read_csv("C:/Users/Shreya Tanguturi/Desktop/PythonPersonalProjects/llm-classification-finetuning/test.csv")

## check data types and non-null values
#print(train.info())
## shows no missing values, as non-null values for all fields are the same as number of entries (57477)

## check data structure
#print("Train shape:", train.shape)
#print("Test shape:", test.shape)


## Preprocessing & Feature Engineering
# clean text: lowercasing, removing extraneous characters 
# decide how to represent each data point: Prompt + response A, Prompt + response B, might be treated as a pair
## EDA
sns.countplot(x=train["winner_model_b"], palette="viridis")
plt.title("Class Distribution")
plt.show()

## create simple "text-pair" input
def combine_text(row):
    return (
        "Prompt: " + str(row["prompt"]) +
        " Response A: " + str(row["response_a"]) +
        " Response B: " + str(row["response_b"])
    )

train["text"] = train.apply(combine_text, axis=1)
test["text"] = test.apply(combine_text, axis=1)

X = train["text"]
y = train["winner_model"]   # target classes: 0,1,2 maybe?

