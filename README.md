# Mnist-on-EvaDB-and-DeepLake

## Install Requirements
```
pip install -r EvaDB_requirements.txt
```
## Run EvaDB mnist
```
python EvaDB_mnist.py
```
## Input Kaggle key
Will have to input kaggle username and key to download the dataset
Can find it at the settings in your kaggle profile 
Click create new token, then use the info in the downloaded kaggle.json file 

## Note
The code will run for a while ... 

## Run DeepLake mnist
```
pip install -r DeepLake_requirements.txt
```

## Input Kaggle key
Will have to input kaggle username and key to download the dataset
Can find it at the settings in your kaggle profile 
Click create new token, then use the info in the downloaded kaggle.json file 


## Run DeepLake mnist
### In order to run the file on different datasets might have to change it manually
## So I pprovided the outputs of the cProfile and the timing stats in the timing and profile folder

This will show the results after getting the predictions using MnistImage Classifier
```
python DeepLake_mnist.py
```
This runs the TQuery using local dataset
```
python Create_DeepLake_Dataset.py
```
This runs the TQuery using activeloop dataset that is stored in https://app.activeloop.ai/yuchenjohnlin/home

But the performance will be effected by the network 
```
python DeepLake_Query.py
```
This runs the User defined function using local dataset
```
python DeepLake_UDFilter.py
```


