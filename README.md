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

## 🧠 What is the Model Learning!
The IPS-data records some kind of solar wind speeds having deduced it from the disturbances caused to the radio signals emanating from distant sources. In effect the IPS data observes the Solar Winds as it propagates away from the sun driven by the complex Magneto-Hydrodynamics(MHD) of the solar plasma and the encompanying magnetic fields. 

The complications involved in this indirect measurement and the extremely complicated MHD driven propagation makes it hard to foretell the stae of the Solar Wind plasma at a later state. We therefore expect the model to learn to translate and decipher what the Solar Wind speeds would be for the next 4-days given the best IPS-data from the past 8-days.


---

## 🧩 Repo Architecture
```
.
├── codes
│   ├── trainer.py        # Implements training, validation and test.
│   ├── data_utils.py     # Generates data sets from which train, val and test sets are drawn.
│   ├── mnn_Utils.py      # Neural netwrok module
│   ├── make_dataset.py   # Extends torch.utils.data.Dataset class for from torch.utils.data.DataLoader
|
├── data
│   ├── data_generated
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── test_dwnld        # Input data for IPS-data, OMNI-data and sun-spot-data 
├── docs
│   ├── _build
│   │   ├── doctrees
│   │   └── html
│   └── source
├── model_outputs         # Outputs from model training 
│   ├── losses
│   └── models
└── notebooks
```
## 💾 Raw Data Sources

| Dataset | Description | Source |
|----------|--------------|--------|
| **IPS** | Interplanetary Scintillation data (solar wind proxy) | [Nagoya University IPS Data Center](https://stsw1.isee.nagoya-u.ac.jp/vlist/) |
| **OMNI** | Solar wind plasma and magnetic field parameters | [NASA OMNIWeb](https://omniweb.gsfc.nasa.gov/) |
| **Sunspot** | Daily sunspot number index | [Royal Observatory of Belgium SILSO](https://www.sidc.be/silso/datafiles) |

The datasets were synchronized and normalized to create a continuous time series for model training and testing.

---

## 🧹 Data Processing
The IPS data from the above source needs to be thoroughly cleaned before being used for any model training. As this is human processed data it suffers from large periods of variations in its availibility. The IPS-data forms the **time-series** input with features without the target values. This data is also combined with the Sun-spot data to add an additional feature to the input. The output or the target data is the periodic Solar Wind speed measurements taken by NASA's OMNI satellite stationed at the L1 point from where it takes few minutes for the solar winds to reach earth. 

### data_utils.py
This is used to clean, process, combine and select the best data satisfying certain criterion from the above raw data sources.
The main task for data processing is to generate the **best possible** IPS data from the past 8-days with relevant features for any given hour for which the OMNI data (as target) is available for the next 4-days from that hour. 

data_utils.py produces data with a row length of 13x32 + 2X16 as follows:

Input features (total 13) in  the given order for 32 IPS-data points from the past 8-days:
| input Feature       | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| dist          | Radial distance, of P-point, from the sun (AU)           |
| hla           | Heliocentric latitude of P-point (deg.)    |
| hlo           | Heliocentric longitude of P-point (deg.)                  |
| gla           | Heliographic latitude of P-point (deg.)                                  |
| glo           | Heliographic latitude of P-point (deg.)                  |
| carr          | Carrington rotation number of P-point                                            |
| v             | Solar wind velocity (km/s), IPS deduction.                                        |
| er            | The error in velocity estimation (km/s)                            |
| sc-indx       | Scintillation level (in arbitrary unit) observed at either Fuji or Kiso station.                                |
| time          | Time of obs. - time_trgt, standardized.                                  |
| day_total     | Total Sun-spots observed (Sun-spot data)      |
| time_trgt     | Target time-stamp.                            |
| input         | Boolean (0, 1): 1 if input is present     |

Output features in the given order for 32 smoothened OMNI data points for the following 4-days starting from time_trgt:
| Output Feature       | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| v         | Solar wind velocity (km/s)  at L1             |
| time            | time-stamp (not used)  |

---

## 📊 Results

| Metric | Value | Notes |
|--------|--------|------|
| **MAE** | --| 96-hour forecast horizon |
| **RMSE** |-- | Under analysis |
| **R²** | -- | Under analysis |

It is expected that this Transformer model provides a base model at least as good as if not better than the Transformer model used to predict the Solar-wind time-series data using the OMNI data with 27, 54 and 81 days persistence along with the solar images. 

---

## 🧪 Experiments
- Currently using CNN based encoder only architecture to test viability of deep neiral networks to learn wind speeds.  
- Possible enhancements may inclued Multi-head attention layers instead of CNNs.
- Data fusion experiments (OMNI-only vs. multi-source inputs).  
- Sensitivity analysis of prediction horizon lengths.

---

## 🧰 Tech Stack

- **Python 3.11.6**
- **PyTorch**, with xpu support
- **NumPy**, **Pandas**, **Matplotlib**, **Scipy**

The whole project is actually managed by a uv environment (see[uv package manager](https://docs.astral.sh/uv/)) with the above requirements.txt generated from it. 
For a full description of the installing intel-xpu drivers compatible with the above packages visit [setup link](https://roninrp.github.io/install_1.html)

---

## 📈 Future Work
- Incorporate additional encoders for OMNI 27-54-81 days persistence and EUV solar images into a multi encoder-decoder transformer architechture extending recent work 🔗[A Multimodal Encoder–Decoder Neural Network for Forecasting Solar Wind Speed at L1](https://iopscience.iop.org/article/10.3847/1538-4365/adf436/meta)
- Incorporate continious data ingestion and training from the above varied sources for viable prediction.  
- Deploy model via a **real-time inference API** for operational prediction systems  

---

## 🧑‍💻 Authors
**Rohan R. Poojary, Dattaraj Dhuri**  
Researchers in Space Weather Prediction & Machine Learning in **Prof. Shravan Hanasoge's group**.

📧 rohan.poojary@protonmail.com  
🌐 [personal website: https://roninrp.github.io/](https://roninrp.github.io/)

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## ⭐ Citation
If you use this repository in your research, please cite:


