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
- [Decision Tree](#decision-tree)
  - [Important features](#important-features)
  - [Pruning the tree](#pruning-the-tree)
  - [Final Feature Importances Summary](#final-feature-importances-summary)
- [Insights](#insights)
- [Business Recommendations](#business-recommendations)
- [Assumptions & Limitations](#assumptions--limitations)

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
  <img src="https://github.com/user-attachments/assets/b0ed055e-4804-456f-9023-ba0f1aa4d201" width="500"/>
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

**Merging Low Volume Market Segments**

To streamline the analysis and improve model performance, the **market_segment_type** variable was simplified by grouping the **Corporate, Complementary**, and **Aviation** segments into a single category labeled Offline. These segments represented a very small portion of the dataset, and combining them with the existing Offline group helped reduce sparsity, simplify interpretation, and ensure more balanced comparisons across segments.

**Outlier Check**

I chose not to treat these outliers, as they carry meaningful information about customer behavior and hotel operations.An exception was made for the **no_of_adults** variable. Based on business logic, a reservation should include at least one adult. Therefore, values of 0 were considered unrealistic and were replaced with the mode to reflect more plausible scenarios.

Another treatment was applied to the **lead_time** variable. Based on industry standards, bookings made more than 10 months in advance (300 days) are rare typically representing less than 1% of total reservations. I follow this approach to limit the influence of extreme values. 

Retaining the remaining outliers allows the model to better capture real world booking behavior. Additionally, models such as Decision Trees are inherently robust to outliers, making them well suited to handle this type of variability without distortion.For these reasons, outliers were preserved as part of the training data.


<p align="center">
  
  <img src="https://github.com/user-attachments/assets/4f263720-86ef-4bc5-a6da-a206b068dd24" width="500"/>

</div>

---

## Model Building

**Data Preparation**

Before training the logistic regression model, the dataset was preprocessed by applying one-hot encoding to categorical features and then split into a **70% training** and **30% test** set. The target variable was booking_status, where **1 = Canceled** and **0 = Not Canceled**.

The class distribution was consistent across both sets, with approximately **33% canceled bookings** and **67% non canceled bookings**. While not fully balanced, the dataset does not present a severe imbalance either.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b086ad68-b04d-4e55-81e0-6e10a744640a" width="300"/>
</div>

--- 

## Logistic Regression Model

The initial results showed an accuracy of 80.6%, a recall of 63.4%, and an F1-score of 68.3% on the training set. While the model demonstrated solid performance, the presence of multicollinearity and high p-values in several features suggested the need of treatment. 

<div style="display: flex; justify-content: center; gap: 20px;">

  <img src="https://github.com/user-attachments/assets/33beaf2e-8be4-4183-8426-d6ef8c6989d3" width="500"/>
  
  <img src="https://github.com/user-attachments/assets/8f0b54fd-364c-49a3-ad64-8bb916e9a39b" width="300"/>
  
</div>

---

## Multicollinearity and Treat High P-values  

To improve the interpretability and stability of the logistic regression model, multicollinearity was assessed using the VIF. Most features showed acceptable VIF values below 5, indicating low multicollinearity.

In parallel, **backward elimination** was applied using p-values from the logistic regression model, until all remaining predictors were statistically significant. This resulted in a refined set of variables strongly associated with cancellation risk.

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

---

## Decision Tree

Before training the Decision Tree model, categorical features were, the dataset was preprocessed by applying **one-hot encoding** to categorical features and splitting the data into training 70% and test 30% sets. Despite high performance on both sets, the significant gap between training and test results especially the near perfect metrics on the training set indicates **overfitting**. For this reason, we will apply pre-pruning to improve the model. 


<div style="display: flex; justify-content: center; gap: 20px;">

  <img src="https://github.com/user-attachments/assets/09aca7de-fc22-4af1-9873-ad8aec26236a" width="400"/>

  <img src="https://github.com/user-attachments/assets/76fbf5ed-f49c-4dd9-97dc-be2e2a2d5a4c" width="400"/>

</div>

---

## Important features

Before applying pre-pruning techniques to reduce overfitting, we analyzed the relative importance of features in the unpruned Decision Tree model. The most influential variables in predicting booking cancellations were:

- **lead_time**: By far the most important feature, indicating that the number of days between booking and arrival is a strong predictor of cancellation likelihood.

- **avg_price_per_room**: Price dynamics likely signal demand peaks or premium bookings, influencing cancellation behavior.

- **market_segment_type_Online**: Online bookings showed a high contribution, consistent with trends seen in earlier exploratory analysis.

- **arrival_date** and **no_of_special_requests** also contributed moderately to the model's decisions.

<div align="center">
  <img src="https://github.com/user-attachments/assets/472b2f50-d371-4828-8d5f-4ce33a080fcb" width="500"/>
</div>

---

## Pruning the tree

**Model Optimization**

To reduce overfitting in the initial Decision Tree model, two pruning strategies were applied:

- **Pre-Pruning (Grid Search):** We tuned hyperparameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` using `GridSearchCV`. The best configuration (e.g., `max_depth = 6`) improved test performance, achieving an **F1-score of 0.75**.

- **Post-Pruning (Cost-Complexity Pruning):** We applied pruning based on the `ccp_alpha` parameter to simplify the tree while maintaining performance. The optimal value (`ccp_alpha ≈ 0.00013`) resulted in a **higher F1-score (0.808)** and **recall (0.853)** on the test set.

**Post-pruning delivered the best results**, offering a better balance between model complexity and accuracy. It outperformed both the original and pre-pruned models, especially in terms of **recall**, which is critical for identifying booking cancellations early.


<div align="center">
  <img src="https://github.com/user-attachments/assets/b9aade7b-dea2-44a2-a269-21981223f330" width="500"/>
</div>

---

## Final Feature Importances Summary 

At the end of the Decision Tree modeling process, the most influential features for predicting booking cancellations were:

- **Lead time**, by far the most important predictor, suggests that bookings made well in advance are more prone to cancellation.

- **Market segment type (Online)** and **average price per room** also played key roles, highlighting customer profile and pricing sensitivity.

- Other relevant variables included the **number of special requests, arrival month**, and **booking timing features** (arrival date, week/weekend nights).

This provides insight into which attributes are most critical in cancellation behavior, supporting both strategic pricing and customer targeting decisions.


<div align="center">
  <img src="https://github.com/user-attachments/assets/6e5eb3ba-5c7c-455c-8d6b-e144221174d8" width="500"/>
</div>

---

## Insights   

- **Lead time** (days between booking and arrival) is a **strong predictor of cancellations**. For every additional day in advance that a booking is made, the chance of cancellation increases by about **1.6%**. Bookings made far in advance especially **beyond 60 days** are more likely to be canceled.

- **Higher room prices** are associated with a small increase in cancellations. On average, for every **€10** increase in the room rate, the chance of cancellation rises by about **2%**.

- **Online bookings** are much more likely to be canceled than offline ones. In fact, bookings made online are about **five times more likely** to be canceled, making this the most influential customer segment in the model.

- **Special requests** strongly reduce the likelihood of cancellation. Each additional request **lowers the chance by about 76%**, and guests with two or more requests almost **never cancel**.

- **Month of arrival**. Cancellations are more frequent from **January to March**, while bookings for **October to December** are the most stable, showing much lower cancellation rates around **6% lower** with each later month.

- **Repeat guests** are highly reliable, with nearly **89%** fewer cancellations compared to first time customers. This highlights the importance of **building guest loyalty**.

---
 
## Business Recommendations

**Reconfirm or Add Deposits** for Long-Lead Bookings
Bookings made more than **90 days in advance** have a significantly higher cancellation risk.
To reduce dropouts and improve inventory planning:

- Send **automated reconfirmation emails** closer to arrival.

- Require **partial deposits** for high risk advance bookings.

---

**Tighten Policies for Online Bookings**
Online bookings are about **5 times more likely to be canceled** than offline ones. 
Consider:

- Offering **non refundable discounts** or added perks.

- Limiting **free cancellation to 7+ days before arrival**.

- Adding **reminders or nudges** during the booking process to reinforce intent.

---

**Use Special Requests to Flag Reliable Guests**
Guests with **2 or more special requests** rarely cancel and show high engagement. 
Use this behavior to:

- Offer them **flexible cancellation policies** or upgrade options.

- **Reduce overbooking buffers** for these low risk reservations.

---

**Promote and Protect Repeat Guests**
Repeat customers are **89% less likely to cancel** compared to first time guests. 
Strengthen this segment by:

- Offering **exclusive discounts** or early access.

- Encouraging **direct bookings** through loyalty perks to reduce OTA dependence.

---

**Identify High-Risk Bookings with No Engagement**
High price bookings without special requests or loyalty status are about **30% more likely to be canceled**. 
These should be:

- **Flagged as high risk** in the booking system.

- Monitored with **confirmation emails or follow up calls** to verify intent.

---

## Assumptions & Limitations

**Assumptions**

<div align="justify">

**Cancellations are binary and final**: The dataset assumes that once a booking is marked as canceled, it remains canceled. No partial or rebooked scenarios were included.

**Booking channel grouping**: Segments such as Corporate, Complementary, and Aviation were grouped under Offline based on booking characteristics and volume, assuming they reflect direct or negotiated channels rather than online platforms.

**Outliers reflect real behavior**: Except for unrealistic values, outliers were retained under the assumption that they represent legitimate, though rare, guest behavior. Since we do not have access to internal business rules, booking policies, or operational exceptions from the INN Hotels chain, we assume these extreme values could be valid in certain scenarios. Without further context, we chose not to discard them to preserve potentially meaningful variation in the data.

</div>
