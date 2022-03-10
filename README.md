# Source Code and Appendix for "A Next Basket Recommendation Reality Check"

## Contents of this repository
* Source code and datasets.
* Descriptions of different dataset format.
* Pipelines about how to run and get results.
* A PDF file with the additional plots.

## Structure
* dataset: contains the .csv format dataset after preprocessing.
* jsondata: contains the .json format dataset.
* mergedata: contains the .json format dataset. 
* methods: contains the source code of different NBR methods and the original url repository link of these methods.
* keyset_fold.py: splits the datasets across users for train/validate/test.
* evaluation: scripts for evaluation.
    * metrics.py: the general metrics.
    * performance_gain.py: evaluate the contribution of repetition and exploration.
    * model_performance.py: evaluate the baskets' rep/expl ratio, and the recall, phr performance w.r.t. repetition and exploration.
* appendix: contains a PDF file with the additional plots.

## Pipeline
1. Select the datasets according to different methods. (Edit the entry and put datasets at the corresponding folder.)
2. Train the model and save the model. (Since we use the original implementations of the authors, we provide the original repository links, which contain the instructions of the environment setting, how to run each method, etc.)
3. Generate the predicted results via the trained model and save the results file.
4. Use the evaluation scripts to get the performance results.

### Take TIFUKNN and dunnhumby dataset as an example:
   [1] create the keyset folder and generate several keyset files (To perform user-level analysis, we use keyset files instead of random seeds to repeat experiments.)
   ```
   mikdir keyset
   python keyfold.py
   ```
   The keyset files will be saved at the keyset folder:
   keyset/dunnhumby_keyset_{index}.json
   [2] set hyper-parameters according to the source code instructions
   ```
   cd methods/tifuknn/
   python tifuknn_new.py ../../jsondata/dunnhumby_history.json ../../jsondata/dunnhumby_future.json ../../keyset/dunnhumby_keyset_0.json 900 0.9 0.6 0.2 3 20
python tifuknn_new.py ../../jsondata/dunnhumby_history.json ../../jsondata/dunnhumby_future.json ../../keyset/dunnhumby_keyset_1.json 900 0.9 0.6 0.2 3 20
python tifuknn_new.py ../../jsondata/dunnhumby_history.json ../../jsondata/dunnhumby_future.json ../../keyset/dunnhumby_keyset_2.json 900 0.9 0.6 0.2 3 20
   ```
   The predict results for each user will be saved at the method folder:
   > methods/tifuknn/dunnhumby_pred{index}.json
   
[3] run the evaluation script
   ```
   python model_performance.py
   ```
   Need to config the name of the pred_file and the index of the keyset file before running.
   
## Dataset format description
* jsondata: 

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }
> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}

* mergedata: 

> {uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}

* Predicted results:

> {uid1: [item, item, ..., item], uid2: [item, item, ..., item], ...}
