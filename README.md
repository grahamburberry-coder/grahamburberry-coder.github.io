# Data Scientist Portfolio

## Portfolio Contents:

**Core Skills**: Google Colab, Python, Pandas, Matplotlib, Numpy, Seaborn, Tensorflow, RE, NLTK, Torch, pyLDAvis

| No. | Title | Problem | Skills/Techniques |
| :--- | :--- | :--- | :--- |
| **1** | **Detecting the Anomalous Activity of a Ship’s Engine** | Apply critical thinking and ML (Machine Learning) concepts to design and implement a robust anomaly detection model. | EDA (Exploratory Data Analysis), Descriptive Statistics, Data Visualization, Anomaly Detection (Statistical), IQR (Interquartile Range) Method, Anomaly Detection (Machine Learning), One-Class SVM (Support Vector Machine), Isolation Forest, Feature Scaling, PCA (Principal Component Analysis) |
| **2** | **Customer Segmentation with Clustering** | Apply critical thinking and ML concepts to design and implement clustering models to perform customer segmentation and improve marketing efforts. | Preprocessing, Feature Engineering, Outlier Detection, Duplicate Handling, Aggregation, Recency Calculation, Frequency Calculation, CLV (Customer Lifetime Value) Calculation, Customer Age Calculation, Feature Scaling, Column Transformer, Pipeline, EDA, Data Visualization, Histograms, Box Plots, Pair Plots, Correlation Matrix, Clustering, Elbow Method, Silhouette Score, Hierarchical Clustering, Dendrogram, K-Means Clustering, Dimension Reduction, PCA, t-SNE, Statistical Analysis, Model Evaluation |
| **3** | **Predict Student Dropout** | Use supervised learning techniques to predict whether a student will drop out. Precise information protected under NDA. | Predictive Modeling, Data Preprocessing, Feature Engineering, Ordinal Encoding, One-Hot Encoding, Missing Value Imputation/Removal, Data Splitting, XGBoost, Neural Networks, Hyperparameter Tuning, Model Evaluation, Accuracy, Precision, Recall, F1 Score, AUC (Area Under Curve), Confusion Matrix, Data Merging, Scikit-learn, TensorFlow/Keras, Data Analysis |
| **4** | **Applying NLP (Natural Language Processing) for Topic modelling in a Real-Life Context** | Analyse gym-company's review data to uncover key drivers that provide actionable insights for enhancing customer experience. Precise information protected under NDA. | NLP, Topic Modelling, BERTopic, Gensim, LDA (Latent Dirichlet Allocation), Emotion Analysis, Data Cleaning, Data Analysis, Data Visualization, Text Preprocessing, Word Clouds, Hugging Face, LLM (Large Language Model,Falcon-7b-instruct), Data Wrangling, NLTK (Natural Language Toolkit) |
| **5** | **Using Time Series Analysis for Sales and Demand Forecasting** | Identify sales patterns that demonstrate seasonal trends or any other traits, providing insights to inform reordering, restocking, and reprinting decisions for various books. |  |


## Project 1: Detecting the Anomalous Activity of a Ship’s Engine
**Date**: 14/06/2025

**Skills/Techniques:**
- Pandas
- Matplotlib
- EDA (Exploratory Data Analysis)
- Descriptive Statistics
- Data Visualization
- Anomaly Detection (Statistical)
- IQR (Interquartile Range) Method
- Anomaly Detection (Machine Learning)
- One-Class SVM (Support Vector Machine)
- Isolation Forest
- Feature Scaling
- PCA (Principal Component Analysis)

**Introduction:**
A ship must operate smoothly and without issues to complete its tasks effectively. A poorly maintained ship engine can cost companies dearly, especially in supply-chain industries, as it will lead to transport inefficiencies, increased fuel consumption, increased risks of malfunctions, and potential safety hazards. The best approach to determine if the ship’s engine was poorly maintained without taking apart the engines themselves is to collect as much data as possible and determine if there was an anomaly. For this project, I was provided with a dataset regarding the functionalities of the ship’s engine, and the task was to determine if any anomalies were detected.

**Methods:**
The dataset contained 19,535 data entries and was organised into six features: engine rpm (revolutions per minute), lubrication oil pressure, fuel pressure, coolant pressure, lubrication oil temperature, and coolant temperature. The ideal methods for identifying anomalies within the dataset are machine learning (ML) algorithms, specifically one-class SVM and Isolation Forest, which detect anomalies within specified parameters, and the statistical method IQR for identifying outliers that fall beyond the upper and lower bound limits. Due to previous experience, the proportion of anomalies was expected to be within 1 – 5%; therefore, the defined parameters were maintained within that range.

The data was first analysed with EDA to identify any missing or duplicate values, descriptive statistics of the data, and visualised data using histograms and boxplots for ease of observation for the distribution and extreme values. Following this, the IQR method was used. The anomalies were determined based on the outliers beyond the higher bound limit (Q3 + 1.5(IQR)) and lower bound limit (Q1 – 1.5(IQR)) for each feature independently, and a binary indication for an outlier observed across two or more features. The ML methods, one-class SVM and Isolation Forest, were used and tested, with their parameters maintained within their expected 1 – 5% predictions (nu=0.01 – 0.05 and contamination=0.01 – 0.05, respectively), as well as other parameters (gamma and n_estimators, respectively) for better optimisation. After PCA was performed, the data was visualised in a 2D perspective. Results were analysed for further observation and the anomalies were determined.

**Results:**
Based on the information displayed in Figure 1 below, anomalies are evident in the dataset. The most statistically significant outliers appear beyond the upper bound limit (above Q3) in the boxplot graphs, particularly for the features Engine RPM and Coolant Temperature. The outliers shown for the features Lub Oil Pressure, Fuel Pressure, Coolant Pressure, and Lub Oil Temperature are close to the upper and lower (below Q1) bound limits. Still, it is difficult to determine if they are statistically significant based on this alone.

**Figure 1:** Boxplots for the ship engine features.

<img width="154" height="100" alt="image" src="https://github.com/user-attachments/assets/dea601de-9d37-4ec8-bb79-fe1f89aa2c42" /><img width="147" height="100" alt="image" src="https://github.com/user-attachments/assets/5cbc44fa-4f27-4b65-a5e5-7824d4fc65db" /><img width="150" height="100" alt="image" src="https://github.com/user-attachments/assets/5cead160-8cbb-4670-841f-1b400cfd5e4d" />
<img width="148" height="100" alt="image" src="https://github.com/user-attachments/assets/192451af-9576-4cf9-bccc-0e8b6e192c9b" /><img width="153" height="100" alt="image" src="https://github.com/user-attachments/assets/2eaba461-b5f5-453b-aacf-cbc234b614e2" /><img width="151" height="100" alt="image" src="https://github.com/user-attachments/assets/03042f13-a416-4539-abcb-4d416db9a776" />

Based on the IQR method, 2.16% or 422 of the samples were calculated as outliers within the dataset. At least two features displayed anomalies across the results, the most evident being Engine RPM, Fuel Pressure and Lub Oil Temp. The observed anomalies identified with IQR appear similar to those detected through the boxplot graphs in Figure 1, though the most significant observed feature was the Engine RPM and Lub Oil Temp features.

Based on the observed results from one-class SVM, anomalies can be seen in the data. However, compared to the IQR statistical method and depending on the established parameters, there are either too many anomalies (1043 at 5%) or too few (189 at 1%) to observe. Unlike with IQR, it is not possible to indicate which features should be considered for further observation. Some anomalies are visually apparent, such as the datapoint [x=11, y=4.5] in Figure 2(A). Nevertheless, it remains difficult to determine whether most of these anomalies fall within an acceptable error range among the clusters. The parameters that are closest to the abnormalities detected in the IQR were at the 2% outlier prediction, as shown in Figure 2(A), with 400 detected anomalies.




