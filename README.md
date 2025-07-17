# KIDCM
This repository contains the official code and datasets for **A Knowledge-informed Dynamic Correlation Modeling Framework for Lane-level Traffic Flow Prediction** 


**Link** https://www.sciencedirect.com/science/article/pii/S1566253525004002

**Authors: ** Ruiyuan Jiang, Shangbo Wang, Wei Ma, Yuli Zhang, Pengfei Fan, Dongyao Jia 

# How to run
**1. Download the dataset**
The dataset of I-24 Motion is available at: https://i24motion.org/data
The trajectory data can be aggregated to any fine-grained traffic data in any time and space through macro.py.  
**2. Process and Run**
1. Diffirent API LLM can be find at LLM_apikey.py.
2. Use the expected generated traffic information in space_modeling.py for modeling traffic dynamics.
3. Generate the expression and transfer it to KIDCM.py.
4. Employ T-GCN model to achieve the weights for surrogate model.
5. Save the weights and bias to the expected directory and run KIDCM.py for prediction tasks.
