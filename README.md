# Data Scientist Portfolio

## Portfolio Contents

**Core Skill:** Python

| No. | Title | Problem | Skills/Techniques |
| :--- | :--- | :--- | :--- |
| **1** | **Detecting the Anomalous Activity of a Ship’s Engine** | Apply critical thinking and ML (Machine Learning) concepts to design and implement a robust anomaly detection model. | EDA (Exploratory Data Analysis), Descriptive Statistics, Data Visualization, Anomaly Detection (Statistical), IQR (Interquartile Range) Method, Anomaly Detection (Machine Learning), One-Class SVM (Support Vector Machine), Isolation Forest, Feature Scaling, PCA (Principal Component Analysis) |
| **2** | **Customer Segmentation with Clustering** | Apply critical thinking and ML concepts to design and implement clustering models to perform customer segmentation and improve marketing efforts. | Preprocessing, Feature Engineering, Outlier Detection, Duplicate Handling, Aggregation, Recency Calculation, Frequency Calculation, CLV (Customer Lifetime Value) Calculation, Customer Age Calculation, Feature Scaling, Column Transformer, Pipeline, EDA, Data Visualization, Histograms, Box Plots, Pair Plots, Correlation Matrix, Clustering, Elbow Method, Silhouette Score, Hierarchical Clustering, Dendrogram, K-Means Clustering, Dimension Reduction, PCA, t-SNE, Statistical Analysis, Model Evaluation |
| **3** | **Predict Student Dropout** | Use supervised learning techniques to predict whether a student will drop out. Precise information protected under NDA. | Predictive Modeling, Data Preprocessing, Feature Engineering, Ordinal Encoding, One-Hot Encoding, Missing Value Imputation/Removal, Data Splitting, XGBoost, Neural Networks, Hyperparameter Tuning, Model Evaluation, Accuracy, Precision, Recall, F1 Score, AUC (Area Under Curve), Confusion Matrix, Data Merging, Scikit-learn, TensorFlow/Keras, Data Analysis |
| **4** | **Applying NLP (Natural Language Processing) for Topic modelling in a Real-Life Context** | Analyse gym-company's review data to uncover key drivers that provide actionable insights for enhancing customer experience. Precise information protected under NDA. | NLP, Topic Modelling, BERTopic, Gensim, LDA (Latent Dirichlet Allocation), Emotion Analysis, Data Cleaning, Data Analysis, Data Visualization, Text Preprocessing, Word Clouds, Hugging Face, LLM (Large Language Model, Falcon-7b-instruct), Data Wrangling, NLTK (Natural Language Toolkit) |
| **5** | **Using LLM and GenAI for Sales and Demand Forecasting** |  (Group Project). Precise information protected under NDA. | Team leader,  |


## Project 1: Detecting the Anomalous Activity of a Ship’s Engine
### Date: June, 2025

### Skills/Techniques
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
  
### Introduction
A ship must operate smoothly and without issues to complete its tasks effectively. A poorly maintained ship engine can cost companies dearly, especially in supply-chain industries, as it will lead to transport inefficiencies, increased fuel consumption, increased risks of malfunctions, and potential safety hazards. The best approach to determine if the ship’s engine was poorly maintained without taking apart the engines themselves is to collect as much data as possible and determine if there was an anomaly. For this project, I was provided with a dataset regarding the functionalities of the ship’s engine, and the task was to determine if any anomalies were detected.

### Methods
The dataset contained 19,535 data entries and was organised into six features: engine rpm (revolutions per minute), lubrication oil pressure, fuel pressure, coolant pressure, lubrication oil temperature, and coolant temperature. The ideal methods for identifying anomalies within the dataset are machine learning (ML) algorithms, specifically one-class SVM and Isolation Forest, which detect anomalies within specified parameters, and the statistical method IQR for identifying outliers that fall beyond the upper and lower bound limits. Due to previous experience, the proportion of anomalies was expected to be within 1 – 5%; therefore, the defined parameters were maintained within that range.

The data was first analysed with EDA to identify any missing or duplicate values, descriptive statistics of the data, and visualised data using histograms and boxplots for ease of observation for the distribution and extreme values. Following this, the IQR method was used. The anomalies were determined based on the outliers beyond the higher bound limit (Q3 + 1.5(IQR)) and lower bound limit (Q1 – 1.5(IQR)) for each feature independently, and a binary indication for an outlier observed across two or more features. The ML methods, one-class SVM and Isolation Forest, were used and tested, with their parameters maintained within their expected 1 – 5% predictions (nu=0.01 – 0.05 and contamination=0.01 – 0.05, respectively), as well as other parameters (gamma and n_estimators, respectively) for better optimisation. After PCA was performed, the data was visualised in a 2D perspective. Results were analysed for further observation and the anomalies were determined.

### Results
Based on the information displayed in Figure 1 below, anomalies are evident in the dataset. The most statistically significant outliers appear beyond the upper bound limit (above Q3) in the boxplot graphs, particularly for the features Engine RPM and Coolant Temperature. The outliers shown for the features Lub Oil Pressure, Fuel Pressure, Coolant Pressure, and Lub Oil Temperature are close to the upper and lower (below Q1) bound limits. Still, it is difficult to determine if they are statistically significant based on this alone.

**Figure 1:** Boxplots for the ship engine features.

<img width="354" height="300" alt="image" src="https://github.com/user-attachments/assets/dea601de-9d37-4ec8-bb79-fe1f89aa2c42" /><img width="347" height="300" alt="image" src="https://github.com/user-attachments/assets/5cbc44fa-4f27-4b65-a5e5-7824d4fc65db" /><img width="350" height="300" alt="image" src="https://github.com/user-attachments/assets/5cead160-8cbb-4670-841f-1b400cfd5e4d" />
<img width="348" height="300" alt="image" src="https://github.com/user-attachments/assets/192451af-9576-4cf9-bccc-0e8b6e192c9b" /><img width="353" height="300" alt="image" src="https://github.com/user-attachments/assets/2eaba461-b5f5-453b-aacf-cbc234b614e2" /><img width="351" height="300" alt="image" src="https://github.com/user-attachments/assets/03042f13-a416-4539-abcb-4d416db9a776" />

Based on the IQR method, 2.16% or 422 of the samples were calculated as outliers within the dataset. At least two features displayed anomalies across the results, the most evident being Engine RPM, Fuel Pressure and Lub Oil Temp. The observed anomalies identified with IQR appear similar to those detected through the boxplot graphs in Figure 1, though the most significant observed feature was the Engine RPM and Lub Oil Temp features.

Based on the observed results from one-class SVM, anomalies can be seen in the data. However, compared to the IQR statistical method and depending on the established parameters, there are either too many anomalies (1043 at 5%) or too few (189 at 1%) to observe. Unlike with IQR, it is not possible to indicate which features should be considered for further observation. Some anomalies are visually apparent, such as the datapoint [x=11, y=4.5] in Figure 2(A). Nevertheless, it remains difficult to determine whether most of these anomalies fall within an acceptable error range among the clusters. The parameters that are closest to the abnormalities detected in the IQR were at the 2% outlier prediction, as shown in Figure 2(A), with 400 detected anomalies.

**Figure 2:** 2D visualisations of 2% anomalies identified by A) one-class SVM, and B) Isolation Forest after using two principal components through PCA.

**A)**
<img width="437" height="379" alt="image" src="https://github.com/user-attachments/assets/d56d2f8b-96bd-4a5a-9a45-0cbbaf2f258b" />
**B)**
<img width="447" height="379" alt="image" src="https://github.com/user-attachments/assets/e5a20bb5-2483-4633-abea-4d38f2747da7" />

Similar to the one-class SVM results, there are anomalies in the data through the Isolation Forest method. Similarly, compared to the IQR statistical method and depending on the established parameters, there are either too many anomalies (977 at 5%) or too few (196 at 1%) to observe. However, unlike with one-class SVM, not all outliers were considered anomalies, such as the datapoint [x=11, y=4.5] in Figure 2(B) compared to Figure 2(A). The parameters that are closest to the abnormalities detected in the IQR were at the 2% outlier prediction, as shown in Figure 2(B), with 391 detected anomalies.

### Conclusion
If the task was to identify what anomalies, if any, were present in the given data, then the ML models one-class SVM and Isolation Forest would have been the best methods. Based on observations of this dataset, it would be preferable to use the one-class SVM since not all outliers were recognised as anomalies under the Isolation Forest ML model, as observed in Figure 2.

If the task was to detect anomalies and narrow down which feature or features to observe further, potentially determining a causal connection, then combining the IQR statistical method with examining boxplot graphs would be the ideal approach. As observed in Figure 1 and further analysis with IQR, this data clearly indicates that there are anomalies, and from a business perspective, I would highly suggest to determine why there are multiple anomalies in the ship’s engine RPMs and whether there is a relationship or causation with Fuel Pressure and Lub Oil Temp.


## Project 2: Customer Segmentation with Clustering
### Date: June, 2025

### Skills/Techniques
- Preprocessing
- Feature Engineering
- Outlier Detection
- Duplicate Handling
- Aggregation
- Recency Calculation
- Frequency Calculation
- CLV (Customer Lifetime Value) Calculation
- Customer Age Calculation
- Feature Scaling
- Column Transformer
- Pipeline
- EDA (Exploratory Data Analysis)
- Histograms
- Box Plots
- Pair Plots
- Correlation Matrix
- Clustering
- Elbow Method
- Silhouette Score
- Hierarchical Clustering
- Dendrogram
- K-Means Clustering
- Dimension Reduction
- PCA (Principal Component Analysis)
- t-SNE (distributed Stochastic Neighbor Embedding)
- Statistical Analysis
- Model Evaluation

### Introduction
The retail industry emphasises understanding and serving customers. Customer segmentation—dividing a customer base into distinct groups based on characteristics—is essential for this. Grouping customers helps businesses tailor marketing, improve product development, enhance satisfaction, boost retention, optimise pricing, and allocate resources effectively. This report presents a customer segmentation analysis on a transnational e-commerce dataset. Its aim is to develop a robust customer segmentation model using clustering (𝑘) techniques, allowing the e-commerce company to understand its diverse customer base better and enhance marketing efficiency.

### Methods
The raw dataset contained 951,668 transactional records with twenty features. The dataset was initially examined for quality issues before aggregating the data to 68,300 customers. This was accomplished by grouping the data by 'Customer ID' and engineering customer-specific features: frequency (total number of unique orders per customer), recency (number of days between the customer’s latest and most recent order), total revenue (sum of all orders placed per customer), average unit cost, total profit, and customer birthdate. Based on the aggregated data, two additional features were engineered: customer lifetime value (CLV; total profit per customer with positive frequency orders) and customer age (based on the latest order date and birthdate). EDA was performed on the engineered features, including generating histograms and box plots to understand the distribution of each feature and identify potential outliers. A correlation matrix heatmap was created to visualise the relationships between the features.

Before clustering the data, the engineered numerical features were scaled using StandardScaler. This was essential for distance-based clustering algorithms like K-Means to prevent larger-scaled features from dominating the distance calculations. To determine the optimal number of clusters (𝑘), the machine learning (ML) methods Elbow and Silhouette Score were used. Elbow analyses the within-cluster sum of squares (inertia) for various 𝑘 values, identifying an "elbow" point where the decrease in inertia slows. Conversely, the Silhouette Score assesses how well each data point fits its assigned cluster compared to others, with higher scores indicating better-defined clusters. Both methods were applied to the scaled customer features for a range of 𝑘 values (2 – 10). Hierarchical clustering was also performed on the scaled data to visualise the potential cluster structure through a dendrogram.

Finally, K-Means clustering was applied to the full dataset of engineered customer features using a chosen optimal 𝑘. To visualise the clusters in a 2D space, the ML methods PCA for dimensional linearity and t-SNE were used to reduce the scaled features. Results were analysed for further observation and conclusions.

### Results
The EDA revealed that features Frequency and Total Revenue were highly correlated, which was to be expected, and there was a negative correlation between Recency with both Frequency and Total Revenue, as displayed in Figure 1(B). The box plots displayed in Figure 1(A) displayed outliers in features Frequency, Recency, Total Revenue, Average Unit Cost, and CLV, but not in Customer Age.

**Figure 1:** Box plots (A) and correlation-matrix heatmap (B) of the engineered features.

**A)** <img width="689" height="492" alt="image" src="https://github.com/user-attachments/assets/11d27d41-b6c6-4970-a14a-a25cd90e133a" />

**B)** <img width="608" height="486" alt="image" src="https://github.com/user-attachments/assets/8dc53d91-6a45-4a37-9019-fcc4e4254e1f" />

As can be seen in Figure 2, through the Elbow method, a less distinct elbow could be determined, suggesting that the reduction in inertia slows down around 𝑘=3 or 4. The Silhouette Score analysis indicated that 𝑘=6 yielded the highest score, but scores for 𝑘=3 – 5 were also relatively close and reasonably balanced. To balance distinct segments with interpretability for marketing, 𝑘=4 was chosen as the optimal number of clusters for K-Means analysis, although 𝑘=3 was initially explored.

**Figure 2:** Elbow and Silhouette Score methods for optimal clusters (𝑘).
<img width="868" height="405" alt="image" src="https://github.com/user-attachments/assets/835589db-d97e-46ab-815e-4cc3c0ca7c8f" />

While hierarchical clustering on a sample provided a visual representation of data structure (Figure 3), the computational limitations made it less practical for determining 𝑘 on the full dataset compared to the Elbow and Silhouette methods. Due to memory constraints with the large dataset size, the dendrogram was generated on a sample of 30,000 data points without displaying individual leaf labels for readability. 

**Figure 3:** Hierarchical dendrogram of clusters, based on 30,000 samples.
<img width="897" height="454" alt="image" src="https://github.com/user-attachments/assets/a345feb3-b1ad-4203-b3cc-09b88aefdc30" />

**Figure 4:** Box plots illustrating features by K-Means clusters (𝑘=4).
<img width="897" height="627" alt="image" src="https://github.com/user-attachments/assets/9f873a4a-c10f-4595-847e-055e93a6229f" />

K-means clustering using 𝑘=4 and the displayed boxplot (Figure 4) suggest four customer segments: high-value loyal customers (high Frequency, Total Revenue, and CLV), churn risk customers (high Recency), premium product buyers (high Average Unit Cost), and new/low activity customers (low values across all features).

**Figure 5:** 2D visualisation of PCA (A) and t-SNE (B) clustered data (𝑘=4).

**A)** <img width="460" height="275" alt="image" src="https://github.com/user-attachments/assets/649804c3-9ddd-4e86-8c62-1f0b06e8821d" /> **B)** <img width="482" height="277" alt="image" src="https://github.com/user-attachments/assets/81f450c5-9395-43c5-bd0c-3c88f30448b8" />

Similar to hierarchical clustering limitations, PCA was applied to the entire dataset, while t-SNE focused on a sample of 30,000 datapoints. The PCA scatter plot (Figure 5[A]) of the first two components showed separation between clusters, notably distinguishing Cluster 0 (High-Value) and Cluster 3 (Lapsed) from Clusters 1 and 2. However, Clusters 1 and 2 overlapped significantly, indicating less linear separability based on principal components. The t-SNE scatter plot (Figure 5[B]) revealed that the four clusters were more distinctly separated than in PCA, with less overlap, suggesting they may not be linearly separable but form more cohesive groups in high-dimensional space.

### Conclusion
Based on statistical methods through K-Means of clustering, the business should concentrate its efforts on increasing profits and saving costs based on high-value loyal customers, recent and engaged customers, older customers who purchase higher-cost items, and lapsed or low-activity customers. The analysis of cluster characteristics through box plots provided actionable insights into the unique profiles of each segment. Based on these results, from a business standpoint, it would be recommended to implement targeted marketing campaigns, focus on retention with a priority on retaining loyal customers with exclusive offers and loyalty programs, develop re-engagement strategies with previous customers, tailor product recommendations based on older customers who purchase higher-cost item segments for personalised product recommendations, and continuously monitor and refine segments. For example, through a calculated distribution of customers by continent, Europe and North America had highest number of customers per continent.


## Project 3: Predict Student Dropout
### Date: August, 2025

### Skills/Techniques
- Predictive Modeling
- Data Preprocessing
- Feature Engineering
- Ordinal Encoding
- One-Hot Encoding
- Missing Value Imputation/Removal
- Data Splitting
- XGBoost
- Neural Networks
- Hyperparameter Tuning
- Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- AUC (Area Under Curve)
- Confusion Matrix
- Data Merging
- Scikit-learn
- TensorFlow/Keras
- Data Analysis

### Introduction
Student dropout poses significant challenges for educational institutions, causing financial losses, reputational damage, and obstructing students' growth. Predicting dropout risk is vital for timely interventions to boost retention and success. This project develops and evaluates supervised learning models to forecast dropouts at various academic stages using data from a study group. The goal is to identify key dropout factors and find the most predictive stages. **To maintain anonymity, the specifics and details are protected under a signed NDA and will remain private.**

### Methods
This project utilised datasets in three distinct stages:

Stage 1: Initial applicant and course info.
Stage 2: Student and engagement data.
Stage 3: Academic performance.

Each dataset was pre-processed before machine learning (ML). 'LearnerCode' was removed as irrelevant; ‘Age' was calculated from 'DateofBirth'; 'CompletedCourse' was converted to a binary 'Target' ('Yes'=1, 'No'=0). Columns with over 200 unique values or more than 50% missing data were discarded. Missing values were handled differently per stage: for Stage 2, rows with less than 2% missing data were removed; for Stage 3, missing values in 'AssessedModules', 'PassedModules', 'FailedModules', 'AuthorisedAbsenceCount' were imputed as 0.0. Other missing values were addressed based on the 2% threshold.

The 'CourseLevel' column was treated as ordinal data and encoded numerically based on a predefined order ('Foundation', 'International Year One', 'International Year Two', 'Pre-Masters'). All other remaining categorical columns were transformed using one-hot encoding, except for the more notable numerical columns in stages 2 and 3.

The models used were XGBoost and a Neural Network. XGBoost is a powerful gradient boosting algorithm for structured data, while the Neural Network was a multi-layer perceptron (MLP) built with TensorFlow/Keras. It featured an input layer, three dense layers with ReLU activation, and an output layer with sigmoid activation for binary classification. The pre-processed data from each stage, plus a final merged dataset, was split into 80-20 training and testing sets with stratification to maintain target variable distribution and address class imbalance. MLs were trained on the training data and evaluated on the unseen testing data. GridSearchCV tuned hyperparameters for XGBoost (XGBClassifier) and Neural Network (KerasClassifier), aiming to optimise performance with AUC as the metric.

### Results
**Table 1:** XGBoost model performance metrics (A) and confusion matrices (B) summary across stages.

**A)**
| | Stage 1 (Untuned) | Stage 1 (Tuned)	| Stage 2 (Untuned)	| Stage 2 (Tuned)	| Stage 3 (Merged) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy**	| 0.8953	| 0.8925	| 0.8954	| 0.8946	| 1.0000 |
| **Precision**	| 0.9202	| 0.9186	| 0.9231	| 0.9242	| 1.0000 |
| **Recall**	| 0.9601	| 0.9585	| 0.9574	| 0.9550	| 1.0000 |
| **F1 Score**	| 0.9397	| 0.9381	| 0.9399	| 0.9394	| 1.0000 |
| **AUC**	| 0.8784	| 0.8798	| 0.8908	| 0.8891	| 1.0000 |

**B)**

| Stage 1 (Untuned)	| Stage 1 (Tuned)	| Stage 2 (Untuned)	| Stage 2 (Tuned)	| Stage 3 (Merged) |
| :---: | :---: | :---: | :---: | :---: |
| array([[ 396,  355], [ 170, 4091]])	| array([[ 389,  362], [ 177, 4084]])	| array([[ 383,  339], [ 181, 4068]])	| array([[ 389,  333], [ 191, 4058]])	| array([[ 722,    0], [   0, 4249]])

---

**Table 2:** Neural Network model performance metrics (A) and confusion matrices (B) summary across stages.

**A)**

| | Stage 1 (Untuned)	| Stage 1 (Tuned)	| Stage 2 (Untuned)	| Stage 2 (Tuned)	| Stage 3 (Merged) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Accuracy	| 0.8851	| 0.8927	| 0.8851	| 0.8851	| 1.0000 | 
| Precision	| 0.9229	| 0.9132	| 0.9191	| 0.9197	| 1.0000 |
| Recall	| 0.9437	| 0.9655	| 0.9492	| 0.9485	| 1.0000 |
| F1 Score	| 0.9332	| 0.9386	| 0.9339	| 0.9338	| 1.0000 |
| AUC	0.8429	| 0.8740	| 0.8300	| 0.8239	| 1.0000 |

**B)**

| Stage 1 (Untuned)	| Stage 1 (Tuned)	| Stage 2 (Untuned)	| Stage 2 (Tuned)	| Stage 3 (Merged) |
| :---: | :---: | :---: | :---: | :---: |
| array([[ 415,  336], [ 240, 4021]])	| array([[ 360,  391], [ 147, 4114]])	| array([[ 367,  355], [ 216, 4033]])	| array([[ 370,  352], [ 219, 4030]])	| array([[ 722,    0], [   0, 4249]]) |

---

*Stage 1 analysis:*

The performance of both XGBoost and Neural Network models on Stage 1 data, shown in Tables 1 and 2, indicates reasonable initial predictive ability. Hyperparameter tuning caused minor performance changes: slight decreases in accuracy, precision, recall, and F1 score, with a small increase in AUC. Confusion matrices (Tables 1B and 2B) reveal slightly more false positives and negatives after tuning, suggesting default parameters were already effective. XGBoost generally outperformed the Neural Network, especially in recall and F1 score, with minimal gains from tuning. Overall, XGBoost is marginally better for Stage 1, and tuning did not significantly improve model performance.

**Figure 1:** Top 10 Most Important Features (Tuned XGBoost - Stage 1 Data).

<img width="2018" height="978" alt="image" src="https://github.com/user-attachments/assets/b209f656-b673-4cdd-993e-5966dce99bf8" />

---

**Figure 2:** Loss curves for the Neural Network model trained on Stage 1 data.

<img width="627" height="474" alt="image" src="https://github.com/user-attachments/assets/4f0a0934-c346-48c8-850b-dc35a24f3a3b" />

The analysis also highlighted Nationality as the top 3 key features in the Stage 1 XGBoost model (Figure 1), which warrants further investigation by the study group to understand potential underlying factors or biases. The Neural Network loss curves (Figure 2) also provided insights into the training process, showing the model's learning progression.

*Stage 2 Analysis:*

Including Stage 2 data modestly improved the XGBoost model's performance, with slight increases in Accuracy, Precision, F1 Score, and AUC (Table 1A), despite minor metric changes after hyperparameter tuning. The tuned XGBoost showed more false positives and fewer false negatives (Table 1B). The Stage 2 Neural Network benefited more from Tuning, with notable gains in Accuracy, Recall, F1 Score, and AUC (Table 2A), though Precision slightly decreased (Table 2B). 

Comparing tuned models, XGBoost slightly outperformed Neural Network in Accuracy, Precision, and F1 Score, while Neural Network had a higher Recall and XGBoost a marginally higher AUC. Both models improved AUC from Stage 1. Overall, Stage 2 models, especially XGBoost, performed better due to additional predictors like absence counts, with the Neural Network improving notably after tuning.

*Stage 3 (Merged) Analysis:*

Models trained on the merged dataset, including all features and the 'Completedcourse' column, achieved perfect metrics and zero false positives and negatives, indicating perfect classification. However, this likely results from data leakage, most likely from including academic performance data (AssessedModules, PassedModules, FailedModules) and the 'CompletedCourse' target variable. This leakage lets the model 'see' outcomes during training, leading to overly optimistic results, as shown by perfect confusion matrices. Therefore, these models are unreliable for predicting dropout before outcome data is available and are unsuitable for early intervention.

### Conclusion
Models trained on Stage 3 data showed perfect performance, indicating data leakage and unreliability for real-world dropout prediction. Focusing on Stage 1 and 2 results, the tuned XGBoost consistently slightly outperformed the Neural Network, especially in AUC. Including Stage 2 data improved both models, suggesting engagement and absence info adds predictive power. Hyperparameter tuning yielded minor Stage 1 gains but more in Stage 2 for Neural Network. XGBoost still slightly outperformed in Stage 2.

In conclusion, the tuned XGBoost model trained on Stage 2 data is the most promising for early dropout prediction. It balances performance and uses early available information, enabling timely interventions like academic support, counselling, or mentoring to improve retention.

Further Recommendations:
•	Recognising key features like Nationality and absence counts can improve support programs.
•	Investigate how nationality predicts outcomes to inform strategies addressing systemic issues and developing culturally sensitive support.
•	Conduct further feature engineering based on domain expertise and importance analysis.
•	Explore handling class imbalance to enhance the model's ability to identify dropouts.
•	Evaluate the impact of intervention strategies on reducing dropout rates.

## Project 4: Applying NLP (Natural Language Processing) for Topic modelling in a Real-Life Context
### Date: September, 2025

### Skills/Techniques
- NLP
- Topic Modelling
- BERTopic
- Gensim
- LDA (Latent Dirichlet Allocation)
- Emotion Analysis
- Data Cleaning
- Data Analysis
- Data Visualization
- Text Preprocessing
- Word Clouds
- Hugging Face
- LLM (Large Language Model, Falcon-7b-instruct)
- Data Wrangling
- NLTK (Natural Language Toolkit)

### Introduction
In today's competitive environment, understanding and responding to customer feedback is essential for business success. Negative customer reviews, in particular, serve as a vital source of unfiltered insights that can reveal operational inefficiencies, product flaws, or service issues. This report presents a detailed analysis of negative customer reviews using advanced topic modelling techniques. The main goal of this project was to systematically identify, categorise, and interpret the most common themes of dissatisfaction within a dataset to offer practical insights for improving operations and enhancing the customer experience. The project encountered several computational challenges, which were successfully overcome to employ a strong, comparative analysis utilising multiple topic modelling approaches for a deeper understanding of the customer feedback landscape. **To maintain anonymity, the specifics and details are protected under a signed NDA and will remain private.**

### Methods
The analysis utilised a multi-stage approach, applying various Natural Language Processing (NLP) models to the combined negative reviews from Google and Trustpilot Reviews after standard preprocessing (tokenisation, stop-word, and numerical removal).

*Stage 1: Traditional Topic Modelling*
BERTopic was initially applied to the entire negative review dataset to identify primary themes and clusters. The outputs of BERTopic were then compared with Gensim's LDA (set to 10 topics), which served as the methodological baseline to confirm the high-level themes.

*Stage 2: Emotion and Focused Analysis*
Emotion analysis was performed using a BERT model to classify the dominant emotion of each negative review, isolating a subset where "anger" was the top emotion. A second, focused BERTopic run was then executed specifically on these angry reviews to identify more granular themes related to intense dissatisfaction.

*Stage 3: LLM-Driven Granularity Refinement*
To address the limitations of "broad catch-all" clusters from Stage 1, a final, high-resolution analysis was performed. The Falcon-7b-instruct LLM (Large Language Model) was used to extract three main topics from a sample of 100 reviews. This granular output was clustered using a secondary BERTopic analysis. The LLM was then utilised a second time to generate actionable business suggestions based on these refined themes. The execution of this stage required utilising an A100 GPU and 8-bit quantisation to manage model resources.

### Results
*General Negative Review Analysis (BERTopic and LDA):*
Initial data investigation revealed key distributional and linguistic characteristics. As shown in Figure 1, 'gym' and 'one' were the most frequent words from both datasets, indicating the general topic focus, though similarities and differences in word frequency existed, necessitating further analysis.

A <img width="444" height="332" alt="image" src="https://github.com/user-attachments/assets/3eb877eb-8dd3-4521-ad09-d97c248b3af1" />
B <img width="445" height="333" alt="image" src="https://github.com/user-attachments/assets/04071507-c6ea-400d-8cce-8b888db447c9" />

Figure 1: Word clouds of tokenised words from customer reviews from Google (A) and Trustpilot (B).

Analysis of negative review counts confirmed that several Capital-City locations consistently appeared among the top 20 most problematic sites across both platforms, with specific locations detailed in Table 1. This location data guided the focus toward high-frequency problem areas.

Table 1: Top 20 locations with the highest number of negative reviews from Google and Trustpilot.
|	| Negative Google Reviews	|	| Negative Trustpilot Reviews |	|
| :---: | :---: | :---: | :---: | :---: |
| |	Location |	Negative Review Count	| Location	| Negative Review Count |
| 1	| Capital-City Part 1	| 59	| City 10	| 50 |
| 2	| Capital-City Part 2	| 26	| 345	| 45 |
| 3	| Capital-City Part 3	| 26	| Capital-City Part 4	| 23 |
| 4	| Capital-City Part 4	| 25	| Capital-City Part 1	| 22 |
| 5	| Capital-City Part 5	| 24	| City 11	| 20 |
| 6	| Capital-City Part 6	| 22	| Capital-City Part 12	| 18 |
| 7	| Capital-City Part 7	| 21	| Capital-City Part 11	| 18 |
| 8	| City 1 Part 1	| 21	| Capital-City Part 8	| 16 |
| 9	| City 2	| 20	| City 12	| 16 |
| 10	| City 3	| 19	| Capital-City Part 13	| 16 |
| 11	| City 4	| 19	| Capital-City Part 14	| 16 |
| 12	| City 5	| 19	| City 13	| 16 |
| 13	| City 6	| 18	| City 14	| 15 |
| 14	| Capital-City Part 8	| 18	| Capital-City Part 15	| 15 |
| 15	| Capital-City Part 9	| 18	| Capital-City Part 5	| 15 |
| 16	| City 7	| 17	| Capital-City Part 16	| 15 |
| 17	| Capital-City Part 10	| 17	| City 15	| 14 |
| 18	| City 8	| 17	| City 1 Part 1	| 14 |
| 19	| Capital-City Part 11	| 16	| City 1 Part 2	| 14 |
| 20	| City 9	| 16	| City 16	| 14 |

The initial BERTopic model on the full negative dataset and the Gensim LDA baseline showed broad agreement, revealing 'catch-all' clusters of customer dissatisfaction: **General Facility, Membership and Access, Crowding and Equipment, Social Interactions, and Facility Conditions**. The presence of 'gibberish' clusters confirmed the limitation of the preprocessing step for these traditional models.

*Emotion Analysis and Focused Findings:*
Analysis of reviews classified as 'angry' provided more focused feedback, confirming the persistence of core themes but with greater detail: **Access Issues (PINS/app), Hygiene and Odour, Locker Room Problems, and Temperature Concerns**. This stage demonstrated that high emotional intensity directly correlates with specific, actionable feedback.

*LLM-Driven Topic Refinement:*
The hierarchical LLM approach successfully addressed the ambiguity of the general clusters by extracting and clustering 299 phrases into 8 distinct, high-impact operational themes (see Table 2). The LLM analysis successfully validated the general areas identified by LDA but provided the necessary granular detail. For instance, the LLM separated the broad 'Social Interactions' theme into two distinct issues: Customer Service (unhelpful) and Staff Attitude (unprofessional).

Table 2: LLM-Driven BERTopic Results (8 Core Negative Themes).
| Topic ID	| Frequency Count	| Key Theme	| Representative Words	| Focus of Customer Dissatisfaction |
| :---: | :---: | :---: | :---: | :---: |
| 1	| 52	| Cleanliness/Hygiene	| Dirty, rooms, changing, water, hygiene, showers	| Facility Cleanliness and Maintenance |
| 2	| 50	| Gym Environment	| Gym, environment, crowding, atmosphere, poor, noise	| Operational Atmosphere and Overcrowding |
| 3	| 38	| Poor Customer Service	| Service, customer, poor, friendly, experience, support	| Failure in Customer Support |
| 4	| 28	| Membership/Billing Errors	| Payment, account, fees, membership, bank, subscription	| Administrative and Financial Issues |
| 5	| 20	| Equipment Issues	| Machines, equipment, broken, cable, maintenance, old	| Equipment Availability and Quality |
| 6	| 19	| Staff Attitude	| Staff, attitude, communication, unprofessional, rude	| Employee Conduct and Professionalism |
| 7	| 17	| Quality of Offerings	| Quality, products, service, value, offering, class	| Perceived Quality of Products/Classes |
| 8	| 13	| Logistics/Access	| Parking, free, crowding, area, logistics, availability	| Parking and Facility Access |

Crucially, the LLM successfully filtered out the noise found in the LDA and initial BERTopic runs, resulting in exact themes. The two highest frequency topics were Cleanliness / Hygiene (Topic 1) and Gym Environment (Topic 2). The final output of the LLM translated these 8 validated and refined themes into strategic recommendations: **Hygiene Priority, Service Training, Billing Clarity, Proactive Maintenance, and Environment Management.**

### Conclusion
Customer complaints consistently focused on app usability, facilities, and service. The project's multi-method approach established a clear path from broad dissatisfaction to specific solutions:

•	Traditional models (BERTopic and LDA) established the scope of issues (**Facility, Membership, Staff**).
•	Emotion Analysis confirmed the most intense complaints related to **Access** and **Hygiene**.
•	The LLM-driven analysis provided the final, granular resolution, confirming the high-priority themes while eliminating the ambiguity of initial clusters.

The LLM analysis successfully confirmed the general areas identified by the LDA baseline but provided the necessary resolution to identify the most critical operational failures: **Hygiene** and **Facility Maintenance** and systemic **Customer Service** and **Billing** failures.

*Recommendations:*
1	**Prioritise top themes (LLM validation):** Based on the LLM's high-resolution clustering, focus operational spending on **Hygiene, Equipment Maintenance,** and **Staff Training** (attitude and service).
2	**Address billing systemically:** Implement changes to the membership payment system to increase clarity and reduce errors, directly addressing administrative friction confirmed across all analyses.
3	**Localised action:** Use the negative review frequency data (Table 1) to target problematic locations (such as Capital-City Part 1) for pilot programs implementing these changes.

## Project 5: Using LLM and GenAI for Sales and Demand Forecasting
### Date: December, 2025

### Skills/Techniques
- 
