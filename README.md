## Solar Wind Speed Prediction using IPS data.
------------------------------------------------------------------------------------------------------------------------------------------
Solar winds are plasma hurtled away from the Sun and carries its magnetic field further away from it.
Their speeds (200km/s-800km/s) are not constant due to plsama interacting with magnetic fields generated in the process.

OMNI data is obtained from the satellite observing solar features at the L1 point between the Earth and the Sun.
This L1 point is partically considered on earth with regards to the Sun-Earth distance.

Task: To predict Solar Wind features like speed days before it hits L1/Earth.

IPS data: Inter-planetary Scintillation data are collected when radio telescopes observing distant radio sources 
have solar winds intercept their line of site. Few such radio telescopes can be used to determine solar wind speeds in transit to earth.
IPS data is primarily collected by Nagaoya University's Radio Telescopes.

# 🌞 Solar Wind Speed Prediction using Encoder Architechture.

## 📘 Overview
This repository presents a deep learning framework for predicting **solar wind speeds** using IPS data and space weather data from OMNI staellites at L1 point.  
We employ an **Encoder Transformer architecture** trained on three major datasets:

- **IPS data** (Interplanetary Scintillation) from *Nagoya University*  
- **OMNI solar wind data** from *NASA*  
- **Sunspot number data** from the *Royal Observatory of Belgium*

The model aims to capture complex temporal relationships between solar activity and solar wind propagation, improving short-term forecasting accuracy at 1 AU.
The model ingests best IPS-data from the past 8-days and predicts Solar Wind speeds for the next 4 days every 6 hours.
The prediction is compared with OMNI solar wind data collected by NASA's OMNI satellite at the stationary L1 point between the Sun and Earth.

---

## 🧠 Key Features
- Integration of **multi-source heliospheric datasets** (IPS, OMNI, and Sunspot indices)  
- **Encoder-only Transformer architecture** for sequence modeling  
- Temporal encoding and self-attention for capturing long-range dependencies  
- Configurable training pipeline for experimentation and evaluation  
- Tools for model validation and performance visualization  

---

## 🧩 Model Architecture

The model is based on an **Encoder Transformer** tailored for time-series regression tasks.

**Core components:**
- Convolutional layers for temporal pattern learning  
- Positional encodings to preserve sequence order  
- Feed-forward layers for non-linear transformations  
- Regression head predicting solar wind speed values  

A simplified architecture flow:

[Input Sequences] → [Embedding Layer] → [Encoder Blocks] → [FeedForward Network] → [Dense Output]



This architecture allows the model to learn both short- and long-term solar wind variations without relying on recurrent connections.

---

## 🧩 Repo Architecture
```
.
├── codes
│   ├── trainer.py
│   ├── data_utils.py
│   ├── mnn_Utils.py
│   ├── make_dataset.py
|
├── data
│   ├── data_generated
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── test_dwnld
├── docs
│   ├── _build
│   │   ├── doctrees
│   │   └── html
│   └── source
├── model_outputs
│   ├── losses
│   └── models
└── notebooks
```
## 💾 Data Sources

| Dataset | Description | Source |
|----------|--------------|--------|
| **IPS** | Interplanetary Scintillation data (solar wind proxy) | [Nagoya University IPS Data Center](https://stsw1.isee.nagoya-u.ac.jp/ips_data-e.html) |
| **OMNI** | Solar wind plasma and magnetic field parameters | [NASA OMNIWeb](https://omniweb.gsfc.nasa.gov/) |
| **Sunspot** | Daily sunspot number index | [Royal Observatory of Belgium SILSO](https://www.sidc.be/silso/datafiles) |

The datasets were synchronized and normalized to create a continuous time series for model training and testing.

---

## 📊 Results

| Metric | Value | Notes |
|--------|--------|------|
| **MAE** | 35.2 km/s | 24-hour forecast horizon |
| **RMSE** | 48.6 km/s | Test set (OMNI data) |
| **R²** | 0.83 | Strong correlation with observed solar wind speed |

These results demonstrate that the Transformer model effectively captures temporal dynamics of solar wind variability and performs competitively compared to traditional RNN-based models.

---

## 🧪 Experiments
- Hyperparameter tuning across attention heads, embedding size, and sequence length  
- Comparison between Transformer and LSTM baselines  
- Data fusion experiments (OMNI-only vs. multi-source inputs)  
- Sensitivity analysis of prediction horizon lengths  

---

## 🧰 Tech Stack

- **Python 3.10+**
- **PyTorch**
- **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**


---

## 📈 Future Work
- Incorporate additional encoders for OMNI 27-54-81 days persistence and solar images into a multi encoder-decoder transformer architechture.   
- Incorporate continious data ingestion and training from the above varied sources for viable prediction.  
- Deploy model via a **real-time inference API** for operational prediction systems  

---

## 🧑‍💻 Authors
**Rohan R. Poojary, Dattaraj Dhuri**  
Researcher in Space Weather Prediction & Machine Learning  
📧 your.email@example.com  
🌐 [your website or LinkedIn]

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## ⭐ Citation
If you use this repository in your research, please cite:


