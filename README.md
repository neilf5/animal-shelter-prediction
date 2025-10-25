  
**Introduction**

**Why was the project undertaken?**

Most animal shelters see a large volume of animals yearly and work to achieve the best possible outcomes while commonly operating with limited resources and capacity constraints. Two critical areas that inform strategic decision-making are understanding which factors influence an animal's likelihood of survival and identifying what affects the duration of time animals spend in the shelter before reaching an outcome.

Understanding survival patterns is essential for shelters to allocate their resources effectively and provide appropriate care based on each animal's needs. By analyzing intake data, including factors such as the animal's condition upon arrival, presence of illness or injury, age, and other demographic characteristics, shelters can identify which animals may require immediate intervention or specialized care. This knowledge allows shelters to make informed decisions about medical treatment prioritization, foster placement, and resource allocation to maximize positive outcomes.

Equally important is understanding the duration animals remain in shelters before reaching an outcome. Minimizing time in the shelter is beneficial for animal well-being, as prolonged stays can be stressful. Additionally, shorter stays increase the shelter's capacity to help more animals. Research consistently shows that returning animals to loving homes as quickly as possible serves the animals' best interests and the shelter's mission.

**What was the research question, the tested hypothesis, or the purpose of the research?**

This project addresses two primary research questions: 

1. Which animal characteristics and intake conditions are most strongly associated with survival outcomes?   
2. What factors predict the duration of time an animal spends in the shelter before reaching an outcome? 

The analysis will provide shelters with actionable insights to recognize where additional resources or different care approaches may improve outcomes. This project aims to assemble these patterns in an accessible, data-driven format that shelter staff can utilize to make informed decisions about medical care prioritization, resource allocation, and operational strategies, ultimately working toward supporting the survival and swift placement of the animals in their care.

While this is a challenging and heavy dataset, we think there is real potential for impact if this data can be analyzed and understood at a deeper level, leading to actionable recommendations for shelter operations.

**Selection of Data**

**What is the source of the dataset? Characteristics of data?**

The dataset is sourced from the City of Long Beach's Animal Shelter public data. The dataset includes 51,648 records from January 1, 2017 to October 14, 2025\. 

The dataset includes categorical variables such as Animal Type, Primary Color, Sex, Intake Condition, Intake Type, and Outcome Type. It also includes numeric variables such as DOB, Intake Date, Latitude, and Longitude.

**Any munging or feature engineering?**

All entries that were not cats or dogs were removed from the dataset, as cats and dogs represent the large majority of shelter intakes and are more directly comparable than wildlife or other animal types, as they are not considered wildlife or exotic animals. 

Records with missing values were removed using dropna(), as complete information on date of birth and age at intake was essential for the survival and duration analyses.

An "Age at Intake" feature was created by calculating the difference between the intake date and the date of birth. This provides the animal's age at the time of shelter entry and is used within our models. During data validation, entries with negative ages at intake were identified and removed from the dataset, as these were assumed to be data entry errors. 

For the Animal Type categorical variable, pd.get\_dummies was used to create a binary indicator where 0 represents cats and 1 represents dogs. This change enables the variable to be used effectively in models later on. This same approach was used for the intake status and is also used within our models.  
**Methods**

**Libraries and Tools:**

* **pandas:** Used extensively for data loading, cleaning, manipulation, and preliminary data exploration (e.g., filtering, creating new columns, handling missing values, one-hot encoding).  
* **numpy:** Used for numerical operations, particularly in calculating RMSE and handling arrays for model training.  
* **matplotlib.pyplot and seaborn:** Employed for data visualization to understand the distribution of features, relationships between variables, and model evaluation (e.g., bar charts, KDE plots, scatter plots, confusion matrices).  
* **scikit-learn:** The primary library for machine learning tasks, providing tools for:  
  * **Model Selection:** `train_test_split` for dividing data into training and testing sets.  
  * **Preprocessing:** `StandardScaler` for feature scaling to ensure that features contribute equally to the models, and `OneHotEncoder` for converting categorical variables into a format suitable for machine learning algorithms.  
  * **Modeling:** Implementing various regression and classification algorithms.  
  * **Evaluation:** Providing metrics such as `mean_absolute_error`, `r2_score`, `mean_squared_error`, `classification_report`, `confusion_matrix`, `accuracy_score`, and `roc_auc_score`.

Three main predictive tasks were addressed: predicting how long the animal is in the shelter with `intake_duration` (regression), predicting whether an animal has an outcome where they have died with `outcome_is_dead` (classification), and predicting `stay_category` (classification) which is a categorization of their stay length. 

**Regression Modeling (Predicting intake\_duration):**  
 Due to skewed data and outliers, intake\_duration was clipped at the 99th percentile.

* **Linear Regression & KNN:** Baseline models; both performed poorly (RMSE \> 54).

* **Random Forest:** Chosen for ability to handle non-linear relationships and feature interactions. Improved after clipping (RMSE \= 34.9).

* **Gradient Boosting (n\_estimators=200, lr=0.03, subsample=0.8, max\_depth=5):** Best performance (RMSE \= 32.98), more effectively captured complex patterns.

**Classification (Predicting outcome\_is\_dead):**

* **Logistic Regression:** Used with scaled features to predict death likelihood. The threshold was lowered (0.1) to prioritize recall for identifying at-risk animals. Evaluation included accuracy, ROC AUC, and confusion matrix. Feature importance was derived from model coefficients.

**Classification (Predicting stay\_category):**

* **KNN Classifier (n=7):** Predicted stay categories (‘0–45’, ‘46–65’, ‘65+’ days) using one-hot encoded categorical and numerical features. Evaluated via classification report and confusion matrix.

**Results**

**Regression Modeling (Predicting Intake Duration)**

We were not able to achieve very good results with predicting intake\_duration based on the available features. We saw the best results with the Gradient Boosting Regression model, with an RMSE of 32.98 and R² of 0.141, which means this model explained about 14% of the variance in intake\_duration. This suggests that while available features capture some predictive signal, much of the variation in intake duration likely depends on unobserved or complex factors. 

### **Classification Modeling (Predicting whether animal will die)**

A Logistic Regression model was trained to predict whether an animal's `outcome_is_dead`. The model was evaluated using a classification report, accuracy, ROC AUC, and confusion matrix.

* **Initial Model (Default Threshold \= 0.5):**  
  * Accuracy: 0.8923  
  * ROC AUC Score: 0.8747  
  * Classification Report:  
    * Class 'False' (Not Dead): Precision 0.90, Recall 0.98, F1-score 0.94  
    * Class 'True' (Dead): Precision 0.80, Recall 0.38, F1-score 0.52  
  * Confusion Matrix: Showed many False Negatives (animals that died but were predicted as not dead).

Interpretation: The model is very good at predicting that an animal will *not* die (high recall for 'False'). However, for predicting that an animal *will* die (class 'True'), the precision is high (80% of predicted deaths are correct), but the recall is low (only 38% of actual deaths are identified). In the context of an animal shelter, a low recall for predicting death is concerning as it means many at-risk animals are missed.

* **Model with Adjusted Threshold (Threshold \= 0.10):** To prioritize identifying more animals at risk of death, the classification threshold was lowered to 0.10.  
  * Accuracy: 0.7290 (Lower than default threshold)  
  * ROC AUC Score: 0.8747 (Unaffected by threshold change)  
  * Classification Report:  
    * Class 'False' (Not Dead): Precision 0.968, Recall 0.704, F1-score 0.815  
    * Class 'True' (Dead): Precision 0.343, Recall 0.868, F1-score 0.492  
  * Confusion Matrix: Showed a significant reduction in False Negatives and an increase in False Positives.

Interpretation: By lowering the threshold, the recall for the 'True' class (predicting death) dramatically increased to 86.8%, meaning the model now identifies a much larger proportion of animals that will actually die. This comes at the cost of precision for the 'True' class (now 34.3%), meaning a higher number of animals predicted to die will actually survive (False Positives). For a shelter prioritizing intervention for at-risk animals, this trade-off might be acceptable, as catching most potential deaths (high recall) is more critical than minimizing false alarms (high precision).

Analyzing the coefficients from the Logistic Regression model for predicting `outcome_is_dead` provided insights into the most influential features:

* **Sex\_Spayed and Sex\_Neutered:** Have large negative coefficients, indicating that spayed/neutered animals are significantly less likely to have a 'dead' outcome compared to intact animals.  
* **Animal Type\_DOG:** Has a negative coefficient, suggesting dogs are less likely to have a 'dead' outcome compared to cats.  
* **Intake Condition\_INJURED SEVERE and Intake Condition\_ILL SEVERE:** Have large positive coefficients, strongly indicating that animals admitted with severe injuries or illnesses are much more likely to have a 'dead' outcome.  
* **Age at Intake:** Has a positive coefficient, suggesting older animals are more likely to have a 'dead' outcome.

These findings align with expectations – animals that are intact, cats, severely injured, severely ill, or older face higher risks of not surviving in the shelter.

### **Classification Modeling (Predicting Stay Category)**

A K-Nearest Neighbors Classifier was used to predict the `stay_category` (short: 0-45 days, medium: 46-65 days, long: 65+ days).

* **KNC Model Performance:**  
  * Classification Accuracy: 0.87  
  * Classification Report:  
    * Class '0-45 days': Precision 0.88, Recall 0.98, F1-score 0.93  
    * Class '46-65 days': Precision 0.22, Recall 0.03, F1-score 0.06  
    * Class '65+ days': Precision 0.32, Recall 0.08, F1-score 0.13  
  * Confusion Matrix: Showed good performance for the '0-45 days' category but poor performance for the '46-65 days' and '65+ days' categories, with many animals from these categories being misclassified as '0-45 days'.

Interpretation: The model is highly accurate at predicting short stays ('0-45 days'), likely due to the large number of animals in this category. However, it struggles significantly to identify medium or long stays, resulting in very low recall and F1-scores for these categories. Due to the very low numbers of animals in the longer stay categories, we were not able to achieve more accurate modeling for these animals. We would like to have more data points and features for these animals. 

**Discussion**

These results matter because they describe both the issues and solutions that exist with respect to keeping animals in an urban environment. Our brief examination of the dataset implies that there is a clear difference between cats and cats and dogs feral compared to those which are likely someone’s pet. 

The vast majority of animals which were spayed and neutered left the shelter after a short stay, while animals which were not spayed or neutered had a higher likelihood of dying in the shelter system. This makes sense, based on our own reasoning, however we do not have enough data to support our findings over a larger area. We only had data from a small area in one of the largest metropolitan areas of the USA. This will obviously lead to questions of whether the trends we saw in the data are supported by data collected in neighboring areas. In the future, it would be better to have more organized data from the various authorities in the area where it is collected. 

We were only looking at the LA metropolitan area, but it is unfortunate that there wasn’t any sort of collective data even for that specific area. It would be helpful for there to be a larger dataset, which would allow us to look at other geographic areas. If we could compare the data from an inner city area to areas closer to the rural-urban border, we might have a better idea of potential outcomes for animals that are brought into a shelter.

I think that the tools we used for this project were adequate, however I am pretty sure that we could come to a more satisfying conclusion if we included more advanced tools like keras or pytorch. Overall, I am pleased with how well the tools that were available to us performed.  

**Summary**

Our most significant finding from examining the data is that most animals taken in are released in less than a month, provided that they are not a feral animal. This means that the majority of animals are either reunited with their humans, or find a new home. 

A minority of animals are kept in the shelter for more than one month, which might result in an extended stay without being adopted or claimed. This makes sense, considering that the data was collected from a major metropolitan area. 

Among the cases where animals did not get adopted out or rescued, we concluded that the majority of these cases were stray animals that were either sick or injured, and brought into a shelter by a member of the community. This could potentially explain our observation of intake conditions being weighed much less heavily than being spayed or neutered for the outcome of an animal. 

If animals that arrive in a distressed condition appear to be a pet, or some other cared for animal, they are more likely to receive care and be released or claimed. Animals that were not spayed or neutered may have been more likely to be perceived as feral animals, which likely did not receive as much medical care, and may have been euthanized.
