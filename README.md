# The repetition and exploration NBR

## Structure
* dataset: contains the .csv dataset after preprocess.
* jsondata: contains the .json format dataset. seperate into history file and future file.
* mergedata: contains the .json format dataset. Merged history file and future file.
* methods: contains the source code of different NBR methods, some with minor modification.
* keyset_fold.py: generate the users for train/validate/test.
* evaluation: script for evaluation rep-expl.
    * metrics.py: the general metrics.
    * performance_gain.py: evaluate the contribution of rep and expl.
    * model_performance.py: evaluate the basket rep/expl ratio, expl/rep recall, phr.

## Pipeline
1. Select the preprocessed dataset according to different methods. (Config the entry or put dataset at their folder.)
2. Train the model and save the model. 
3. Generate the predicted results and save the results file.
4. Use the evluation repetition and exploration script to get the results.


## Format description
* jsondata: 

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }
> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}

* mergedata: 

> {uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}
* Predict results:

> {uid1: [item, item, ..., item], uid2: [item, item, ..., item], ...}
