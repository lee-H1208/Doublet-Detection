# Doublet Detection

## Purpose

The goal of this project was to develop a lightweight, easy-to-understand machine learning algorithm using Python for detecting doublets in single-cell RNA sequencing data. 
In addition to building a functional model, this project served as a learning experience in applying both supervised and unsupervised learning techniques.

Over the course of two months, [I explored and implemented](##Sources) concepts from existing state-of-the-art doublet detection algorithms to inform and guide my own approach. 
Through this process, I deepened my understanding of key machine learning principles while building a practical tool from the ground up.

## Contents

- `\src\`: Main source directory for the doublet detection project  
  - `dbldec.py`: Entry point containing the full doublet detection pipeline.  
  - `dbldec_clustering.py`: Implements the simple clustering methods (Leiden clustering at multiple resolutions).  
  - `dbldec_density.py`: Calculates local cell densities and identifies density outliers.  
  - `dbldec_generate.py`: Provides artificial doublet generation methods, including cluster-based and scDblFinder-inspired approaches.  
  - `dbldec_naive.py`: Computes naive doublet scores based on co-expression features.  
  - `dbldec_utils.py`: Includes preprocessing functions and verbose-mode visualizations.  
  - `dbldec_xgbc.ipynb`: Jupyter notebook demonstrating the full pipeline from data input to prediction.  
  - `gbc_output.csv`: Output file containing doublet detection results on benchmark datasets.

## Datasets

The real datasets from the benchmarking computational doublet detection methods [paper](https://doi.org/10.1016/j.cels.2020.11.008) was used: [Link to datasets](https://zenodo.org/records/4062232#.X6GordD0laQ)


## Results

|Dataset        |Num Cells|AUPRC|AUROC|TN   |FP  |FN  |TP  |
|---------------|---------|-----|-----|-----|----|----|----|
|pbmc-1A-dm     |3298     |0.495|0.859|3059 |119 |41  |79  |
|pbmc-1B-dm     |3790     |0.411|0.779|3500 |160 |55  |75  |
|pbmc-1C-dm     |5270     |0.536|0.835|4685 |269 |106 |210 |
|pbmc-2ctrl-dm  |13913    |0.680|0.918|11458|857 |306 |1292|
|pbmc-2stim-dm  |13916    |0.663|0.913|11553|732 |358 |1273|
|pbmc-ch        |15272    |0.627|0.822|12022|705 |1039|1506|
|pdx-MULTI      |10296    |0.418|0.735|7823 |1156|617 |700 |
|cline-ch       |7954     |0.396|0.593|6154 |335 |1104|361 |
|HEK-HMEC-MULTI |10641    |0.485|0.776|9689 |463 |227 |262 |
|hm-12k         |12820    |0.833|0.987|11563|527 |59  |671 |
|hm-6k          |6806     |0.962|0.999|6459 |176 |0   |171 |
|HMEC-orig-MULTI|26426    |0.439|0.746|21719|1139|2154|1414|
|HMEC-rep-MULTI |10580    |0.558|0.663|6952 |346 |2342|940 |
|J293t-dm       |500      |0.123|0.496|444  |14  |37  |5   |
|mkidney-ch     |21179    |0.585|0.689|11206|2072|4348|3553|
|nuc-MULTI      |5578     |0.445|0.763|4526 |577 |222 |253 |

### Statistics

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 0.865   |
| Precision     | 0.570   |
| Recall (TPR)  | 0.495   |
| FNR           | 0.505   |
| FPR           | 0.068   |
| TNR           | 0.932   |
| AUPRC         | 0.541   |
| AUROC         | 0.786   |

## Algorithm

### 1. Basic Preprocessing
- Filter out empty genes and cells.
- Select only the top 1000 expressed genes.
- Log-normalize the data.
- Perform PCA with 30 principal components.

### 2. Perform “Simple Clustering”
- Run Leiden clustering at resolution 2.0.
- Run Leiden clustering at resolution 0.1.

### 3. Mark Density Outliers
- Calculate local density for each cell within clusters based on PCA coordinates.
- Define local density as inverse mean distance to the nearest 2% neighbors.
- Identify cells with density below a percentile threshold (scaled by dataset size) as outliers.
- Remove these low-density cells from the dataset before training and testing.

### 4. Generate Random Artificial Doublets
- Generate an equal number of artificial doublets as the number of cells in the dataset (ensuring balanced training dataset).
- This code was translated from R from scDblFinder, minor modifications were made to ensure performance in python.

### 5. Calculate Co-expression Doublet Scores
- This code was translated from R from scDblFinder, minor modifications were made to ensure performance.

### 6. Split Dataset (Training and Testing Set)
- Exclude density outliers from training and testing.
- Use log-normalized, PCA-transformed sparse matrix as input.
- Calculate and add features for classifier:
  - Naive doublet scores - inspired from other computational methods (RADO, DoubletFinder)
  - PCA coordinates
  - CXDS scores
  - Library sizes filtered between 5th and 95th percentiles - inspired by VAEda

### 7. Train XGBoost Classifier
- Learning rate: 0.01
- Number of rounds: 200
- Evaluation metrics: AUC-PR (Area Under Precision-Recall Curve), logloss

### 8. Prepare and Predict on Real Data
- Set classification threshold at 0.5.
- Return predictions (`doublet_preds`) and prediction probabilities (`doublet_probs`).

## Conclusions

Every dataset (besides the following: HMEC-orig-MULTI, HMEC-rep-MULTI, J293t-dm and hm-12k)
performed on par with our model.

The J293t-dm dataset should be interpreted cautiously since it contains only 500 doublets and may suffer from mislabeling, as suggested by consistently poor performance (AUPRC < 0.3) across other computational methods.

To my knowledge, hm-12k and hm-6k contain only heterotypic doublets, making it extremely difficult
to classify properly with randomly generated doublets. This also explains the lowered performance of
hm-12k with our algorithm.
- Additionally, I found that including the library sizes as a feature made the model overfit for hm-12k, but it was crucial to increase the performance of the pbmc datasets.

The biggest challenge was improving performance on the PBMC datasets. 
When I first began this project, my area under the precision-recall curve (AUPRC) hovered around 0.41, with the pbmc-1A-dm dataset performing particularly poorly at 0.2. 
I theorize that the relatively small number of cells in this dataset (~2000) caused random doublet generation to outperform heterotypic doublet generation, which led me to initially use a mixed doublet generation approach. 
Interestingly, larger PBMC datasets (e.g., pbmc-2ctrl-dm) also showed significantly better results when using randomly generated doublets

### Other statistical conclusions:
- The model achieves a solid overall accuracy of 86.5%, indicating it correctly classifies the majority of cells.
- The precision of 57% shows that when the model predicts a cell as a doublet, it is correct more than half the time.
- The recall (true positive rate) of 49.5% indicates the model detects about half of the true doublets, suggesting some doublets are missed.
- The false negative rate (50.5%) mirrors the recall, confirming a notable fraction of doublets are not being identified.
- The false positive rate is low at 6.8%, meaning few singlets are mistakenly labeled as doublets.
- A true negative rate of 93.2% reflects strong specificity, correctly identifying most singlets.

### Notes:
- We focused on maximizing AUPRC because it effectively balances precision and recall, which is especially important as doublets are "rare" creating imbalanced datasets.
- Maintaining a low false positive rate in doublet detection is crucial because mistakenly removing true singlet cells can lead to loss of valuable data and compromise the integrity of the analysis.

## Next Steps

1. Further reduce both the false positive rate and false negative rate to improve overall classification reliability.
2. Improve AUPRC performance on the hm-12k dataset, which exclusively contains labeled heterotypic doublets.
3. Investigate the cause of poor model performance on the HMEC-orig-MULTI and HMEC-rep-MULTI datasets to identify potential labeling inconsistencies or data-specific challenges.
4. Perform more systematic hyperparameter optimization to enhance model performance across diverse datasets.
5. Package the project for easy installation via pip, including proper documentation and versioning, to support wider adoption and reproducibility.

## Deployment

To run the doublet detection pipeline locally:

1. Clone the Repository
   ```bash
   git clone https://github.com/lee-H1208/Doublet-Detection.git
   cd Doublet-Detection/src
   ```
2. Install Required Dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Notebook
   ```bash
   jupyter notebook dbldec_xgbc.ipynb
   ```
4. Select a Dataset
   ```python
   sample_path = r'path_to_your_dataset'
   ```

## Sources

1. [Benchmarking computational doublet-detection methods for single-cell RNA sequencing data](https://doi.org/10.1016/j.cels.2020.11.008)
2. [Doublet identification in single-cell sequencing data using scDblFinder](https://doi.org/10.12688/f1000research.73600.2)
    - [Github](https://github.com/plger/scDblFinder/tree/devel)
3. [Vaeda computationally annotates doublets in single-cell RNA sequencing data](https://doi.org/10.1093/bioinformatics/btac720)
    - [Github](https://github.com/kostkalab/vaeda)
4. [Robust and Accurate Doublet Detection of Single-Cell Sequencing Data via Maximizing Area Under Precision-Recall Curve](https://www.biorxiv.org/content/10.1101/2023.10.30.564840v1)
    - [Github](https://github.com/poseidonchan/RADO)
