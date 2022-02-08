# Source Code and Appendix for "A Next Basket Recommendation Reality Check"

## Content of this repository
* Source code and datasets.
* Descriptions of different dataset format.
* Pipelines about how to run and get results.
* A PDF file with the additional plots, which can not be included in the paper due to space limitation.

## Structure
* dataset: contains the .csv format dataset after preprocessing.
* jsondata: contains the .json format dataset. Split into history file and future file.
* mergedata: contains the .json format dataset. Merged history file and future file.
* methods: contains the source code of different NBR methods and the original url repository link of these methods.
* keyset_fold.py: split the dataset across users for train/validate/test experiments.
* evaluation: scripts for evaluation.
    * metrics.py: the general metrics.
    * performance_gain.py: evaluate the contribution of repetition and exploration.
    * model_performance.py: evaluate the baskets' rep/expl ratio, and the recall, phr performance w.r.t. repetition and exploration.
* appendix: contains a PDF file with the additional plots described in the paper.

## Pipeline
1. Select the preprocessed datasets according to different methods. (Config the entry or put dataset at the corresponding folder.)
2. Train the model and save the model. 
3. Generate the predicted results and save the results file.
4. Use the evaluation repetition and exploration scripts to get the performance results.


## Formats description
* jsondata: 

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }
> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}

* mergedata: 

> {uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}

* Predict results:

> {uid1: [item, item, ..., item], uid2: [item, item, ..., item], ...}
