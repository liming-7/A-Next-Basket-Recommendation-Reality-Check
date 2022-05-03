# Source Code and Appendix for "A Next Basket Recommendation Reality Check"

## Required packages
To run our preprocessing, data splitting, evaluation scripts, Pandas, Numpy and Python >= 3.6 are required.

To run the pubished methods' code, you can go to the original repository and check the required packages.
## Contents of this repository
* Source code and datasets.
* Descriptions of different dataset format.
* Pipelines about how to run and get results.
* A PDF file with the additional plots.

## Structure
* preprocess: contains the script of dataset preprocessing.
* dataset: contains the .csv format dataset after preprocessing.
* jsondata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored seperately.
* mergedata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored together.
* methods: contains the source code of different NBR methods and the original url repository link of these methods.
* keyset_fold.py: splits the datasets across users for train/validate/test.
* evaluation: scripts for evaluation.
    * metrics.py: the general metrics.
    * performance_gain.py: evaluate the contribution of repetition and exploration.
    * model_performance.py: evaluate the baskets' rep/expl ratio, and the recall, phr performance w.r.t. repetition and exploration.
* appendix: contains a PDF file with the additional plots.

## Pipeline
* Step 1. Select the different types of preprossed datasets according to different methods. (Edit the entry or put datasets at the corresponding folder.)
* Step 2. Train the model and save the model. (Note that we use the original implementations of the authors, so we provide the original repository links, which contain the instructions of the environment setting, how to run each method, etc. We also provide our additional instructions in the following section, which can make the running easier.)
* Step 3. Generate the predicted results via the trained model and save the results file.
* Step 4. Use the evaluation scripts to get the performance results.

## Dataset 
### Preprocessing
We provide the scripts of preprocessing, and the preprocessed dataset with different formats, which can be used directly.
If you want to preprocess the dataset yourself, you can download the dataset from the following urls:
* Tafeng: https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset
* Dunnhumby: https://www.dunnhumby.com/source-files/
* Instacart: https://www.kaggle.com/c/instacart-market-basket-analysis/data

then, config the file name in the script and run.
Take tafeng dataset as an example:
```
user_order = pd.read_csv('../DataSource/ta_feng_all_months_merged.csv', usecols=['TRANSACTION_DT', 'CUSTOMER_ID', 'PRODUCT_ID'])
```
"../DataSource/ta_feng_all_months_merged.csv" is the path of the original tafeng dataset you download.

### Format description of preprocessed dataset
* dataset: --> G-TopFreq, P-TopFreq, GP-TopFreq
> csv format
* jsondata: --> Sets2Sets, TIFUKNN, DNNTSP, DREAM

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }

> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}

* mergedata: --> Beacon, CLEA, UP-CF

> {uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}

* Predicted results:

> {uid1: [item, item, ..., item], uid2: [item, item, ..., item], ...}

## Random split the dataset
We want to analyze the experimental results, basket components and perform user level analysis. Instead of using random seeds, we use keyset file to store the random split to repeat experiments:
```
python keyset_fold.py --dataset dunnhumby --fold_id 0
python keyset_fold.py --dataset dunnhumby --fold_id 1
python keyset_fold.py --dataset dunnhumby --fold_id 2
...
```
Repeat this process several times, you can get several keyset files under the keyset folder.

Keyset file name: {dataset}\_keyset_{index}.json

## Guidelines for each method
Our approach to reproducibility is to rely as much as possible on the artifacts provided by the user themselves, the following repositories have information about how to run each NBR method and the required packages.
* DREAM: https://github.com/yihong-chen/DREAM
* BEACON: https://github.com/PreferredAI/beacon
* CLEA: https://github.com/QYQ-bot/CLEA/tree/main/CLEA
* Sets2Sets: https://github.com/HaojiHu/Sets2Sets
* DNNTSP: https://github.com/yule-BUAA/DNNTSP
* TIFUKNN: https://github.com/HaojiHu/TIFUKNN
* UP-CF@r: https://github.com/MayloIFERR/RACF

We also provide our additional instructions if the original repository is not clear, as well as the parameters we use.

### G-TopFreq, P-TopFreq, GP-TopFreq
Three frequency based methods are under the folder "methods/g-p-gp-topfreq".
* Step 1: Check the file path of the dataset, or put the dataset into corresponding folder.
* Step 2: Using the following commands to run each method:
```
python g_topfreq.py --dataset dunnhumby --fold_id 0
...
python p_topfreq.py --dataset dunnhumby --fold_id 0
...
python gp_topfreq.py --dataset dunnhumby --fold_id 0
...
```
Predicted files are stored under folder: "g_top_freq", "p_top_freq", "gp_top_freq".

Predicted file name: {dataset}_pred{fold_id}.json

### Dream
Dream is under the folder "methods/dream".
* Step 1: Check the file path of the dataset in the config-param file "{dataset}conf.json"
* Step 2: Train and save the model using the following commands:
```
python trainer.py --dataset dunnhumby --fold_id 0 --attention 1
...
python trainer.py --dataset tafeng --fold_id 0 --attention 1
...
python trainer.py --dataset instacart --fold_id 0 --attention 1
...
```
* Step 3: Predict and save the results using the following commands:
```
python pred_results.py --dataset dunnhumby --fold_id 0
...
python pred_results.py --dataset tafeng --fold_id 0
...
python pred_results.py --dataset instacart --fold_id 0
...
```
Predicted file name: {dataset}_pred{fold_id}.json

### Beacon
Beacon is under the folder "methods/beacon".
* Step 1: Copy dataset to its folder, check the file path of the dataset.
* Step 2: Generate pre-computed correlation matrix using the following commands:
```
python cmatrix_generator.py --dataset dunnhumby --foldk 0
...
python cmatrix_generator.py --dataset tafeng --foldk 0
...
python cmatrix_generator.py --dataset instacart --foldk 0
...
```
* Step 3: Train model using the following commands:
```
python main_gpu.py --dataset dunnhumby --foldk 0 --train_mode True --emb_dim 64
...
python main_gpu.py --dataset dunnhumby --foldk 0 --train_mode True --emb_dim 64
...
python main_gpu.py --dataset dunnhumby --foldk 0 --train_mode True --emb_dim 64
...
```
* Step 4: Predict and save the results using the following commands:
```
python main_gpu.py --dataset dunnhumby --foldk 0 --prediction_mode True --emb_dim 64
...
python main_gpu.py --dataset tafeng --foldk 0 --prediction_mode True --emb_dim 64 
...
python main_gpu.py --dataset instacart --foldk 0 --prediction_mode True --emb_dim 64 
...
```
Predicted file name: {dataset}_pred{foldk}.json

### CLEA
CLEA is under the folder "methods/clea"
* Step 1: Copy dataset to its folder, check the file path of the dataset.
* Step 2: Pre-train several epochs using the following commands:
```
python new_main.py --dataset dunnhumby --foldk 0 --pretrain_epoch 20 --before_epoch 0 --epoch 10  --embedding_dim 64 --num_product 3920 --num_users 22530
...
python new_main.py --dataset tafeng --foldk 0 --pretrain_epoch 20 --before_epoch 0 --epoch 10 --embedding_dim 64 --num_product 11997 --num_users 13858
...
python new_main.py --dataset instacart --foldk 0 --pretrain_epoch 20 --before_epoch 0 --epoch 10 --embedding_dim 64 --num_product 13897 --num_users 19435
...
```
* Step 3: Further train the model and save the model using the following commands.
```
python new_main.py --dataset dunnhumby --foldk 0 --log_fire cleamodel --alternative_train_epoch 10 --alternative_train_epoch_D 10 --pretrain_epoch 2 --before_epoch 2 --epoch 30 --temp_learn 0 --temp 10  --embedding_dim 64 --num_product 3920 --num_users 22530
...
python new_main.py --dataset tafeng --foldk 0 --log_fire cleamodel --alternative_train_epoch 10 --alternative_train_epoch_D 10 --pretrain_epoch 2 --before_epoch 2 --epoch 30 --temp_learn 0 --temp 10 --num_product 11997 --num_users 13858
...
python new_main.py --dataset instacart --foldk 1 --log_fire cleamodel --alternative_train_epoch 10 --alternative_train_epoch_D 10 --pretrain_epoch 2 --before_epoch 2 --epoch 30 --temp_learn 0 --temp 10 --num_product 13897 --num_users 19435
...
```
* Step 4: Predict and save the results using the following commands:
```
python pred_results.py --dataset dunnhumby --foldk 0 --log_fire cleamodel --alternative_train_epoch 10 --alternative_train_epoch_D 10 --pretrain_epoch 2 --before_epoch 2 --epoch 30 --temp_learn 0 --temp 10  --embedding_dim 64 --num_product 3920 --num_users 22530
...
python pred_results.py --dataset tafeng --foldk 0 --log_fire cleamodel --alternative_train_epoch 10 --alternative_train_epoch_D 10 --pretrain_epoch 2 --before_epoch 2 --epoch 30 --temp_learn 0 --temp 10 --num_product 11997 --num_users 13858
...
python pred_results.py --dataset instacart --foldk 1 --log_fire cleamodel --alternative_train_epoch 10 --alternative_train_epoch_D 10 --pretrain_epoch 2 --before_epoch 2 --epoch 30 --temp_learn 0 --temp 10 --num_product 13897 --num_users 19435
...
```

Predicted file name: {dataset}_pred{foldk}.json

### Sets2Sets
Sets2Sets is under the folder "methods/sets2sets"
* Step 1: Copy dataset to its folder or check the file path of the dataset.
* Step 2: Train and save Sets2Sets model using the following commands:
```
python sets2sets_new.py dunnhumby 0 10 1
...
python sets2sets_new.py tafeng 1 10 1
...
python sets2sets_new.py instacart 2 10 1
...
```
* Step 3: Predict and save the results using the following commands:
```
python sets2sets_new.py dunnhumby 0 10 0
...
python sets2sets_new.py tafeng 1 10 0
...
python sets2sets_new.py instacart 2 10 0
...
```
Predicted file name: {dataset}_pred{foldk}.json

### DNNTSP
DNNTSP is under the folder "methods/dnntsp".
* Step 1: Go to config/parameter file, edit the following values: data, history_path, future_path, keyset_ path, item_embed_dim, items_total ... an example:
```
{
    "data": "Instacart",
    "save_model_folder": "DNNTSP",
    "history_path": "../../../jsondata/instacart_history.json",
    "future_path": "../../../jsondata/instacart_future.json",
    "keyset_path": "../../../keyset/instacart_keyset_0.json",
    "item_embed_dim": 32,
    "items_total": 13897,
    "cuda": 0,
    "loss_function": "multi_label_soft_loss",
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optim": "Adam",
    "weight_decay": 0
}
```
* Step 2: Train and save the models using the following command:
```
python train_main.py
```
* Step 3: Predict and save results using the following commands:
```
python pred_results.py --dataset dunnhumby --fold_id 0 --best_mode_path XXX
```
Note, DNNTSP will save several models during the training, an epoch model will be saved if it has higher performance than previous epoch, so XXX is the path of the last model saved during the training.

Predicted file name: {dataset}_pred{foldk}.json
### UP-CF
UP-CF is under the folder "methods/upcf".
* Step 1: Copy the dataset to its folder, and check the dataset path and keyset path.
* Step 2: Predict and save the results using the following commands:
```
python racf.py --dataset dunnhumby --foldk 0 --recency 25 --asymmetry 0.75 --locality 10
...
python racf.py --dataset tafeng --foldk 0 --recency 10 --asymmetry 0.75 --locality 10
...
python racf.py --dataset instacart --foldk 0 --recency 10 --asymmetry 0.75 --locality 100
...
``` 
Predicted file name: {dataset}_pred{foldk}.json

### TIFUKNN
TIFUKNN is under the folder "methods/tifuknn"
* Step 1: Predict and save the results using the following commands:
```
cd tifuknn
python tifuknn_new.py ../../jsondata/dunnhumby_history.json ../../jsondata/dunnhumby_future.json ../../keyset/dunnhumby_keyset_0.json 900 0.9 0.6 0.2 3 20
...
python tifuknn_new.py ../../jsondata/tafeng_history.json ../../jsondata/tafeng_future.json ../../keyset/tafeng_keyset_0.json 300 0.9 0.7 0.7 7 20
...
python tifuknn_new.py ../../jsondata/instacart_history.json ../../jsondata/instacart_future.json ../../keyset/instacart_keyset_0.json 900 0.9 0.7 0.9 3 20
```
Predicted file name: {dataset}_pred{foldk}.json

## Evaluation 
Once we got the reommended basket of the model/algorithm on all datasets, you can use our scripts in the evalution folder to evaluate performance w.r.t. repetition and exploration.

Note that, each method will save their results to their own pred folder. 

### Performance

* Step 1: Check the dataset, keyset, pred_file path in the code.
* Step 2: Evaluate the performance using the following commands:
```
cd evaluation
python model_performance.py --pred_folder XXX --fold_list [0, 1, 2, ...]
```
XXX is the folder where you put the predicted baskets, fold_list requires a list of all the keyset files you use in the experiments.

The results will be printed out in the terminal and saved to "eval_results.txt".

### Performance gain
* Step 1: Check the dataset, keyset, pred_file path in the code.
* Step 2: Evaluate the performance using the following commands:
 ```
cd evaluation
python performance_gain.py --pred_folder XXX --fold_list [0, 1, 2, ...]
```
XXX is the folder where you put the predicted baskets, fold_list requires a list of all the keyset files you use in the experiments.

The results will be printed out in the terminal.
