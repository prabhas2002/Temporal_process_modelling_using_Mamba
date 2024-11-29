

# Temporal Point Process Modeling with Mamba-TP: Enhanced Performance over THP

This repository contains the implementation of a novel approach for temporal point process modelling using **Mamba** as the backbone. The method, named **Mamba-TP**, modifies the architecture to improve performance compared to the Transformer Hawkes Process (THP). 

The results demonstrate significant improvements in predictive accuracy and efficiency across multiple datasets. The evaluation includes detailed comparisons with THP, along with visualizations of the results.


## **Directory Structure**

```
pwd/
├── Mamba/                 # Mamba-TP implementation
│   ├── preprocess.py      # Preprocessing for temporal point process datasets
│   ├── main.py            # Mamba architecture adapted for TPP
│   └── run.sh             # Script to run with multiple parameters
│
├── THP/                   # THP implementation
│   ├── preprocess.py      # Preprocessing for temporal point process datasets
│   ├── main.py            # THP implementation
│   └── run.sh             # Script to run with multiple parameters
│
├── requirements.txt       # Dependencies for the project
└── results.ipynb          # Notebook with evaluation metrics and comparison plots
```

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/prabhas2002/Temporal_process_modelling_using_Mamba/
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### Running Models
- Navigate to either the `Mamba/` or `THP/` directory.
- Execute the `run.sh` script to train and evaluate models with various parameters:
  ```bash
  cd Mamba
  bash run.sh
  ```

### Viewing Results
Open the `results.ipynb` notebook to explore:
- Evaluation metrics: **log-likelihood**, **accuracy**, **RMSE**.
- Comparative plots showcasing the performance of Mamba-TP vs. THP.

---

## **Evaluation Metrics**
- **Log-Likelihood**: Measures the probability of observed events under the model.
- **Accuracy**: Assesses the correctness of event type predictions.
- **RMSE**: Evaluates the error in predicting event timestamps.

---


