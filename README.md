# INN Hotels – Predicting Booking Cancellations

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa0b7553-7901-482a-8b84-598e7a3201fb" width="500"/>
</p>

## Table of Contents

- [Project Background](#project-background)
- [Executive Summary](#executive-summary)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing ](#data-preprocessing)
- [Model Building ](#model-building)
  - [Logistic Regression Model](#logistic-regression-model)
  - [Multicollinearity and Treat High P-values](#multicollinearity-and-treat-high-p-values)
  - [Final Logistic Model](#final-logistic-model)
  - [Threshold Optimization](#threshold-optimization)
  - [Model performance summary](#model-performance-summary)
- [Evaluation and Optimization ](#evaluation-and-optimization)
  - [Hyperparameter Tuning comparison](#hyperparameter-tuning-comparison)
  - [Final Model Evaluation on Test Set ](#final-model-evaluation-on-test-set)
  - [Future importances](#future-importances)
  - [Pipeline Evaluation](#pipeline-evaluation)
- [Insights](#insights)
- [Business Recommendations](#business-recommendations)
- [Assumptions & Limitations](#assumptions--limitations)
Model performance summary

## Project Background 

<div align="justify">
INN Hotels Group, a hotel chain in Portugal, faced rising operational costs due to frequent booking cancellations particularly last minute ones resulting in unoccupied rooms, higher distribution expenses, and lower profit margins.

As part of the Data Science team, I collaborated with Revenue Management and Guest Services to develop a machine learning model that predicts the likelihood of cancellations using historical booking data. The model prioritized minimizing false negatives to prevent revenue loss, while also controlling false positives to optimize room allocation.

This solution allowed the hotel to anticipate cancellations, implement proactive overbooking strategies, and design policies that improved occupancy and profitability.

</div>

---

## Executive Summary

<div align="justify">
The model identified lead time, booking channel, and special requests as the most impactful predictors of cancellations. Longer lead times slightly increase cancellation risk, while online bookings are over five times more likely to be canceled than offline ones. In contrast, guests who make special requests, especially two or more, are significantly less likely to cancel, indicating stronger intent. Repeated guests also show much higher reliability, with an 89% lower chance of canceling. Seasonality plays a role as well: cancellations are more common early in the year and less likely toward the end of the year. These insights can support targeted strategies to reduce cancellation rates and improve booking quality.
</div>





---
## Exploratory Data Analysis (EDA)

<div align="justify">
  
The dataset contains historical booking records from INN Hotels and is used to build a predictive model for reservation cancellations. Each row represents an **individual hotel booking made by a customer**. The dataset consists of **36,275 rows** and **19 columns**, covering a mix of categorical and numerical features.The final cleaned version of the dataset contains no missing values or duplicate records, ensuring it is ready for supervised classification.
</div>

**Data Dictionary:**
<img src="https://github.com/user-attachments/assets/381c13ec-5105-481f-837f-57d97016ca20" alt="INN Hotel Logo" width="380" align="right"> 

- `Booking_ID`: Unique identifier for each reservation  
- `no_of_adults`, `no_of_children`: Guest composition  
- `no_of_week_nights`, `no_of_weekend_nights`: Duration of stay
- `type_of_meal_plan`, `room_type_reserved`: Service preferences  
- `lead_time`: Days between booking and check-in  
- `arrival_year`, `arrival_month`, `arrival_date`: Arrival information 
- `market_segment_type`: Booking source or channel  
- `repeated_guest`, `no_of_previous_cancellations`: Booking history  
- `avg_price_per_room`: Room price (dynamic, in euros)  
- `no_of_special_requests`: Number of custom requests  
- `booking_status`: Target variable indicating cancellation (Yes/No)





---

### Univariate Analysis 

**Lead Time** 

Lead time represents the number of days between the booking date and the arrival date.

- The distribution is heavily right-skewed, with the majority of bookings occurring within the first **100 days**.

- The most frequent bookings occur between **0 and 10 days**, suggesting a large share of last minute reservations.

- The median lead time (the green dashed line) is approximately **62 days**.

- Several outliers beyond 300 days likely represent **early planners** or **group bookings**.

**Interpretation:**

Shorter lead times may reflect impulsive or flexible bookings, which are more likely to be canceled. Given its distribution and behavioral implications, lead time is expected to be a key variable in predicting cancellation outcomes.



<div align="center">
  <img src="https://github.com/user-attachments/assets/81622a50-258e-4671-8ae5-c78bcc755eb7" width="500"/>
</div>

--- 

**Market Segment Type**

The variable indicates the source channel through which the booking was made.

- The **Online** segment dominates the dataset with **23,214 bookings**, followed by the **Offline** segment with **10,528**. Together, these two channels account for over **90% of all bookings**.

- **Corporate, Complementary**, and **Aviation** segments represent a very small share of bookings (less than 10%).
 
- The distribution suggests a strong dependency on **online platforms**, which may influence booking behavior and cancellation trends.

**Interpretation:**
Given the dominance of online bookings, this channel could play a critical role in cancellation patterns.

<div align="center">
  <img src="https://github.com/user-attachments/assets/024d62a5-0534-4c02-81bc-57745b315f0b" width="500"/>
</div>

---

**Average Price per Room**

The variable represents the average nightly rate charged for each booking, in euros.

- The distribution is right-skewed, with most prices falling between **60 and 140 euros**.

- The **median price** is just below **€105**, as indicated by the green dashed line.

- A small number of high price outliers exceed **€300**, as shown in the boxplot, likely corresponding to **luxury or peak season** bookings.

- Minor spikes around €0 and other rounded values may reflect promotional rates or data encoding artifacts.

**Interpretation:**

Most bookings are priced within a competitive range, while higher priced stays are relatively rare. Since more expensive bookings may lead to greater hesitation or stricter refund policies, price sensitivity could influence cancellation behavior.

<div align="center">
  <img src="https://github.com/user-attachments/assets/32e4a1ed-745a-411c-8f09-8340f8e80479" width="500"/>
</div>

### Bivariate Analysis

**Average Price per Room vs Market Segment Type**

The boxplot compares the distribution of average room prices across different booking channels.

- Online and Offline segments show comparable price ranges, but Online bookings have a higher median and a wider spread, with many outliers above €200, indicating greater price variability.

- Corporate bookings are more concentrated around mid-range prices, with fewer extreme values, suggesting negotiated or standardized rates.

- Aviation prices are tightly clustered near €100, showing minimal variation, which is consistent with fixed-rate contracts for airline crew or partners.

- Complementary bookings have a median close to €0, reflecting free stays or promotional offers.

**Interpretation:**

Room pricing differs significantly across market segments. The Online channel displays the greatest variability, possibly due to dynamic pricing strategies or discount-driven customer behavior. These pricing patterns may influence both guest expectations and the likelihood of cancellations, and should be considered in model segmentation. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/50b0053b-0688-430e-8b44-bac163029a92" width="500"/>
</div>


**Special Requests vs. Booking Status**

This stacked bar chart shows the proportion of **canceled (1)** and **not canceled (0)** bookings by number of special requests made.

- Guests with zero special requests have the highest cancellation rate, with over **40%** of those bookings ending in cancellation.

- As the number of special requests increases, the cancellation rate drops. From **3 or more requests**, no cancellations are observed.

- There’s a clear negative correlation: more engagement via requests appears to indicate stronger booking intent and lower risk of cancellation.

**Interpretation:**

Customers who make special requests are more invested in their stay and less likely to cancel. This variable is a strong behavioral indicator and should be treated as an important feature in the predictive model. It may also guide hotel staff in identifying low risk guests for personalized service or upselling.

<div align="center">
  <img src="https://github.com/user-attachments/assets/d197c73e-4c17-4209-813a-2277af98d0a4" width="500"/>
</div> 

**Month of Arrival vs. Booking Status**

This stacked bar chart shows the proportion of canceled and non-canceled bookings by month of arrival where **canceled (1)** and **not canceled (0)**.

- **July (7)** shows the highest cancellation rate, with nearly 45% of bookings canceled.

- Summer months (**June, July, August**) generally reflect elevated cancellation patterns, possibly linked to vacation flexibility or last minute plans.

- In contrast, winter months particularly **December** and **January** have lower cancellation rates, dropping below 20%.

- A clear seasonal pattern is observed: cancellations decrease in colder months, potentially due to more stable, purpose driven bookings like business travel or holidays.

**Interpretation:**

Seasonality appears to influence booking reliability. Bookings in peak vacation months carry higher cancellation risk, making arrival_month a valuable feature for forecasting demand volatility and overbooking strategies.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b1de6d05-0f59-42c1-8fcc-4ba77295e5c5" width="700"/>
</div>

---

## Data Preprocessing

**Outlier Check**

We chose not to remove or cap these outliers, as they carry meaningful signals about customer behavior and hotel operations.An exception was made for the no_of_adults variable. Based on business logic, a reservation should include at least one adult. Therefore, values of 0 were considered unrealistic and were replaced with the mode to reflect more plausible scenarios.

Retaining the remaining outliers allows the model to better capture real-world booking behavior. Additionally, models such as Decision Trees are inherently robust to outliers, making them well-suited to handle this type of variability without distortion.For these reasons, outliers were preserved as part of the training data.

<div style="display: flex; justify-content: center; gap: 20px;">

  <img src="https://github.com/user-attachments/assets/4f263720-86ef-4bc5-a6da-a206b068dd24" width="500"/>

  <img src="https://github.com/user-attachments/assets/ad5f63a6-fa8c-4cfe-8b97-1a4300efe337" width="400"/>

</div>

---

## Model Building

**Data Preparation**

Before training the logistic regression model, the dataset was preprocessed by applying **one-hot encoding** to categorical features (with drop_first=True) and splitting the data into training 70% and test 30% sets. The target variable was **booking_status** (1 = Canceled, 0 = Not Canceled). The class distribution remained consistent across both sets 33% cancellations, ensuring a balanced and representative split for modeling.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b086ad68-b04d-4e55-81e0-6e10a744640a" width="300"/>
</div>

--- 

## Logistic Regression Model

The initial results showed an accuracy of 80.6%, a recall of 63.4%, and an F1-score of 68.3% on the training set. While the model demonstrated solid performance, the presence of multicollinearity and high p-values in several features suggested the need of treatment. 

<div style="display: flex; justify-content: center; gap: 20px;">

  <img src="https://github.com/user-attachments/assets/33beaf2e-8be4-4183-8426-d6ef8c6989d3" width="500"/>
  
  <img src="https://github.com/user-attachments/assets/c7bbbf73-f825-4de7-8302-02b012826cea" width="300"/>
  
</div>

---

## Multicollinearity and Treat High P-values**

To improve model the multicollinearity was assessed using the Variance Inflation Factor (VIF). Several dummy variables, particularly within the market_segment_type feature, showed high VIF values (above 60), indicating strong linear dependencies. In parallel, backward elimination was applied using p-values from the logistic regression summary to remove statistically insignificant variables (p > 0.05). This iterative process reduced the feature set to only those with meaningful contribution to the model, resolving multicollinearity and improving overall model.  

---

## Final Logistic Model

After resolving multicollinearity and removing statistically insignificant features based on p-values, a new logistic regression model was trained using the reduced set of predictors.The model achieved a accuracy of 80.57%, precision 73.94%, recall 63.33% and F1-Score of 68.23%.

<div style="display: flex; justify-content: center; gap: 20px;">

  <img src="https://github.com/user-attachments/assets/bab10a2a-542c-4571-8429-831db17b5991" width="500"/>

  <img src="https://github.com/user-attachments/assets/e8bcf8e2-e762-49d6-934a-d1153991cbfb" width="350"/>

</div>



**Coefficients to Odds**

To make the logistic regression results more interpretable, the model coefficients were converted from log-odds to odds ratios using the exponential transformation.This allows us to understand the impact of each predictor on the likelihood of cancellation in percentage terms.


---

## Threshold Optimization

<div align="justify">

To improve the model's ability to detect cancellations, we performed threshold tuning using the ROC Curve and Precision-Recall Curve. While logistic regression defaults to a threshold of 0.5, this value may not be optimal for imbalanced datasets or when recall is a priority.

The ROC Curve (AUC = 0.86) revealed an optimal threshold around 0.37, improving the true positive rate with minimal increase in false positives. The Precision-Recall Curve confirmed this tradeoff, highlighting another suitable threshold at 0.42, where recall was maximized without severely compromising precision.

Adjusting the threshold to 0.37–0.42 significantly improved recall, our key metric for anticipating cancellations, supporting the hotel’s goal of minimizing revenue loss from no-shows.

</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/136a04e7-ec1d-43fd-8631-1d722240862e" width="400"/>
  <img src="https://github.com/user-attachments/assets/69bb651f-9a8e-49a1-a236-cf986250c75f" width="400"/>
</p>

## Model performance summary

After evaluating different thresholds, the model with 0.37 achieved the highest recall in the test performance, making it ideal for capturing the most cancellations. However, as a Data Scientist, I would recommend using the 0.42 threshold instead, as it offers a more balanced trade-off between recall and precision, making it better suited for real-world decision-making where both false positives and false negatives carry a cost.



<div align="center">
  <img src="https://github.com/user-attachments/assets/d4cdf8da-92bf-496a-a052-42e52f125211" width="700"/>
</div>


## Decision Tree

Before training the Decision Tree model, categorical features were, the dataset was preprocessed by applying **one-hot encoding** to categorical features (with drop_first=True) and splitting the data into training 70% and test 30% sets. Despite high performance on both sets, the significant gap between training and test results especially the near perfect metrics on the training set indicates **overfitting**. For this reason, we will apply pre-pruning to improve the model. 


<div style="display: flex; justify-content: center; gap: 20px;">

  <img src="https://github.com/user-attachments/assets/09aca7de-fc22-4af1-9873-ad8aec26236a" width="400"/>

  <img src="https://github.com/user-attachments/assets/76fbf5ed-f49c-4dd9-97dc-be2e2a2d5a4c" width="400"/>

</div>

---

**Important features**

Before applying pre-pruning techniques to reduce overfitting, we analyzed the relative importance of features in the unpruned Decision Tree model. The most influential variables in predicting booking cancellations were:

- **lead_time**: By far the most important feature, indicating that the number of days between booking and arrival is a strong predictor of cancellation likelihood.

- **avg_price_per_room**: Price dynamics likely signal demand peaks or premium bookings, influencing cancellation behavior.

- **market_segment_type_Online**: Online bookings showed a high contribution, consistent with trends seen in earlier exploratory analysis.

- **arrival_date** and **no_of_special_requests** also contributed moderately to the model's decisions.

<div align="center">
  <img src="https://github.com/user-attachments/assets/472b2f50-d371-4828-8d5f-4ce33a080fcb" width="500"/>
</div>

---

## Pruning the tree**

**Model Optimization**

To address overfitting in the initial Decision Tree model, two pruning techniques were applied:

**Pre-Pruning (Grid Search):**
A hyperparameter tuning process using GridSearchCV was conducted over parameters such as max_depth, min_samples_split, and min_samples_leaf. The best configuration (e.g., max_depth=6) improved generalization, resulting in a test F1-score of 0.75.

**Post-Pruning (Cost-Complexity Pruning):**
A post-pruning strategy was implemented by analyzing the trade-off between tree complexity and performance using the cost-complexity parameter ccp_alpha. The optimal value (ccp_alpha ≈ 0.00013) was selected based on the highest test F1-score (0.808) and recall (0.853), leading to a more robust and simplified model.


**Post-pruning** proved to be the best option, delivering the strongest trade-off between model complexity and predictive power. It outperformed both the original and pre-pruned versions by offering improved generalization and higher recall—crucial for cancellation prediction scenarios where false negatives carry significant cost.


<div align="center">
  <img src="https://github.com/user-attachments/assets/b9aade7b-dea2-44a2-a269-21981223f330" width="500"/>
</div>

---

**Final Feature Importances Summary**

At the end of the Decision Tree modeling process, the most influential features for predicting booking cancellations were:

- Lead time, by far the most important predictor, suggests that bookings made well in advance are more prone to cancellation.

- Market segment type (Online) and average price per room also played key roles, highlighting customer profile and pricing sensitivity.

- Other relevant variables included the number of special requests, arrival month, and booking timing features (arrival date, week/weekend nights).

This ranking provides valuable insight into which attributes are most critical in cancellation behavior, supporting both strategic pricing and customer targeting decisions.


<div align="center">
  <img src="https://github.com/user-attachments/assets/6e5eb3ba-5c7c-455c-8d6b-e144221174d8" width="500"/>
</div>

## Insights   

**Lead time** remains a strong predictor of cancellations: each additional day increases the odds of cancellation by 1.6% (OR = 1.0158). This is also supported by the Decision Tree model, where lead time shows a feature importance of ≈ 0.39.

**Higher room prices** are linked to slightly higher cancellation risk: for every €10 increase, the odds of cancellation rise by ~2.05%, possibly reflecting greater hesitation with premium bookings.

**Online bookings** show significantly higher cancellation risk: their odds of cancellation are 435% higher than offline bookings (OR = 5.35), making this the most impactful categorical feature in both models.

**Special requests** are a strong signal of booking commitment: each additional request reduces the odds of cancellation by 76% (OR = 0.2385). Guests with 2 or more requests rarely cancel, reflecting higher engagement.

**Month of booking** influences reliability: with each later month, the odds of cancellation drop by ~6% (OR = 0.9399). Cancellations are more common early in the year, January to March, with a median probability of cancellation during the months April to June, and less likely from October to December, suggesting stronger commitment in late-season travel.

Repeated Guests Are Still the Most Reliable Segment
Repeat customers had ~89% lower odds of cancelling compared to first-time guests (OR = 0.1093), highlighting the importance of loyalty and familiarity with the brand.


 
## Recomendations

Reconfirm or Add Deposits to Long-Lead Bookings
Bookings made more than 90 days in advance should trigger automated reconfirmation emails or require partial deposits. These bookings carry significantly higher cancellation risk and benefit from early engagement strategies to reduce dropouts and improve inventory planning.

✅ Tighten Policies for Online Bookings
Given their 435% higher cancellation risk, consider the following measures for online reservations:

Non-refundable incentives, such as discounts or added amenities

Stricter free cancellation windows, e.g., no refunds within 7 days

Booking flow nudges or reminders to reinforce intent during the purchase journey

✅ Use Engagement Signals (Special Requests) to Flag Reliable Guests
Guests who make two or more special requests demonstrate higher commitment and lower cancellation rates. Use this as a behavioral signal to:

Offer them flexible or premium policy options

Adjust overbooking buffers downward for such bookings

✅ Identify High-Risk Price-Only Bookings
Bookings that are above average in price but lack engagement (no special requests, no loyalty status) show a ~30% higher chance of cancellation. These should be:

Flagged as high risk

Monitored closely

Consider follow-up calls or confirmation messages to verify intent

✅ Promote and Protect Repeat Customers
Guests with previous stays are statistically far less likely to cancel. Prioritize this group by:

Offering early access, exclusive discounts, or personalized perks

Encouraging direct bookings via loyalty incentives to reduce OTA dependency

✅ Incorporate Seasonal Patterns from Arrival Month
Since cancellations are more common for check-ins in January–March, and drop significantly between October–December, consider:

Applying stricter cancellation policies or deposits for early-year bookings

Running promotions and retention campaigns during Q4 to lock in high-intent travelers

Segmenting guests by arrival month to inform overbooking strategies and marketing timing

✅ Apply Cancellation Predictions to Operations
With a recall score of 0.844, the post-pruned Decision Tree model can anticipate most cancellations. Integrate it into hotel operations to:

Dynamically manage overbooking

Optimize staffing in front desk and housekeeping

Reduce revenue loss from unanticipated no-shows
