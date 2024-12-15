# NegiScan
CPSC 488 FInal Project that determines toxic and non toxic comments.

Install Kaggle:

pip3 install kaggle

Run Command to download CSV file from kaggle website.

Source Link Website: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

Unzip the dataset:

unzip jigsaw-toxic-comment-classification-challenge.zip -d jigsaw_data

Check that train.csv exists

ls jigsaw_data

Install required libraries

pip3 install pandas nltk scikit-learn

NegiScan1.py: Main script that runs the entire toxic comment classification process

train.csv file

jigsaw_data: Contains the training data (CSV files) from Kaggle

How to run file

python3 NegiScan1.py
