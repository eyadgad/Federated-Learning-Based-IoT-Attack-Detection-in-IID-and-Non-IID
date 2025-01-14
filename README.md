# [A Robust Federated Learning Approach for Combating Attacks Against IoT Systems Under Non-IID Challenges](https://doi.org/10.1109/SmartNets61466.2024.10577749)

---

## 📖 Project Overview
This repository implements a **Federated Learning (FL)** framework for **IoT attack detection** under challenging **non-IID data distributions**. By utilizing the **CICIoT2023 dataset**, we explore the performance of three FL algorithms: **FedAvg**, **FedProx**, and **Scaffold**. This work addresses statistical heterogeneity in IoT data, a common bottleneck in FL systems, and proposes robust methods to ensure scalability and effectiveness in **privacy-preserving machine learning** for resource-constrained IoT networks.
---

## 🗂️ Repository Structure
This repository is organized as follows:

### 📁 Data:
- **CICIoT2023 Dataset**: A large-scale dataset for IoT attack detection. Access it [here](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

### 📁 Code:
- **`utils.py`**: Essential helper functions for metrics calculation, logging, and visualization.
- **`preprocess.py`**: Scripts for loading, cleaning, and splitting data into IID and non-IID partitions.
- **`models.py`**: Implementation of machine learning models used in the FL pipeline.
- **`federated_utils.py`**: Core FL functionalities, including client updates, model aggregation, and Scaffold optimizations.
- **`train.py`**: Main training pipeline for federated models (FedAvg, FedProx, Scaffold).
- **`requirements.txt`**: A list of all necessary dependencies for the project.

---

## 🎯 Results
Our experiments yielded the following performance on the **CICIoT2023 dataset**:

### Experimental Results Summary

| **Algorithm**  | **Setting**    | **Peak Accuracy** | **Minimum Loss** | **Observations**                                                                 |
|-----------------|---------------|-------------------|------------------|---------------------------------------------------------------------------------|
| **FedAvg**      | IID           | **86.33%**        | **0.53**         | Fluctuations in accuracy and loss across 100 rounds.                           |
| **FedProx**     | IID (mu=0.01) | **93.29%**        | **0.16**         | Consistent improvement; higher mu values showed slight accuracy reduction.     |
| **Scaffold**    | IID           | **96.16%**        | **N/A**         | Superior performance with high accuracy; consistent zero loss needs validation.|
| **FedAvg**      | Non-IID       | **28.88%**        | **12.54**        | Struggles significantly with non-IID data.                                     |
| **FedProx**     | Non-IID (mu=0.01) | **58.46%**    | **1.58**         | Marginal improvement over FedAvg.                                              |
| **FedProx**     | Non-IID (mu=0.04) | **71.88%**    | **1.10**         | Best adaptability in non-IID data, showing substantial improvement.            |
| **Scaffold**    | Non-IID       | **18.78%**        | **N/A**     | Limited performance under non-IID conditions.                                  |


---

## ⚙️ Prerequisites
- **Python**: Version 3.8 or higher
- **Libraries**: To set up the environment, run the following:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage
### Data Preprocessing

The `preprocess.py` script prepares the **CICIoT2023 dataset** for federated learning experiments. Key steps include:

- **Loading and Sampling**: Loads and balances data across categories like `DDoS`, `DoS`, `Mirai`, and others, adding `BenignTraffic` for class balance.
- **Feature Scaling and Encoding**: Encodes labels using `LabelEncoder` and scales features using `StandardScaler`.
- **Data Partitioning**: Splits data into server, training, and testing sets for both **IID** and **Non-IID** configurations across clients.
- **Batch Preparation**: Generates PyTorch `DataLoader` objects for efficient training.

Run the script:
```bash
python preprocess.py
```

Ensure the CICIoT2023 dataset is downloaded from [CICIoT2023 Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html) and placed in the specified directory (`DATASET_DIRECTORY`).

### Training the Models

The `train.py` script evaluates **FedAvg**, **FedProx**, and **SCAFFOLD** federated learning algorithms on **IID** and **Non-IID** partitions of the **CICIoT2023 dataset**. Key experiments include:

- **FedAvg**: Baseline algorithm tested with IID and Non-IID data.
- **FedProx**: Evaluated with multiple `mu` values (`0.01`, `0.04`, `0.1`, `0.2`, `0.3`) for both data types.
- **SCAFFOLD**: Addresses gradient variance, tested on IID and Non-IID distributions.

#### To run:
```bash
python train.py
```

Results, including global models and performance metrics, are saved in the `output/` directory for analysis.

---

## 📄 References
If you use this work in your research, please cite:

```plaintext

@INPROCEEDINGS{10577749,
  author={Gad, Eyad and Fadlullah, Zubair Md and Fouda, Mostafa M.},
  booktitle={2024 International Conference on Smart Applications, Communications and Networking (SmartNets)}, 
  title={A Robust Federated Learning Approach for Combating Attacks Against IoT Systems Under Non-IID Challenges}, 
  year={2024},
  pages={1-6},
  doi={10.1109/SmartNets61466.2024.10577749}
}

```

---

## ✉️ Contact
For questions about the dataset or implementation, contact **Eyad Gad** at [egad@uwo.ca](mailto:egad@uwo.ca).

---
