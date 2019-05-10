# spambot-detector

This project demonstrate a simple way to contextually analysis tweets using BoW
and decide if a given tweet is made by a bot, input dataset are based on cersci 2017 dataset
 
This was done to give us some information on how we can proceed further in bot detection

It also includes a TF-IDF version and dataset preprocessor

To run this project, you need to have cresci dataset, you need to copy spambot & genuine datasets
in root of the project, rename them bot-#dataset-all.csv (e.g. bot-1-all.csv) or gen-all.csv
and then run `csv-to-txt` notebook. this will generate txt version of tweets with the same name
in project root.
The next step is to run `generate-data` notebook to generate sized dataset, you can select which
dataset you want to use (or if you want to use a combined dataset).
After doing this, you would have (train/test/val/trv)-(bot/gen).txt files. trv contains both the
training and validation dataset and was used in TF-IDF network.
You can now run `tf-idf` or `bot-detection-embedding` notebooks to train and evaluate said networks.