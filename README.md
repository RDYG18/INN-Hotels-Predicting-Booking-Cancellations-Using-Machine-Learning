# INN Hotels – Predicting Booking Cancellations

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa0b7553-7901-482a-8b84-598e7a3201fb" width="500"/>
</p>



## Company Context  

As a Data Scientist at **INN Hotels**, a hotel group operating across Portugal, I was brought in to address an operational challenge affecting revenue: a high rate of booking cancellations. These cancellations often happen at the last minute and are facilitated by flexible online policies, which while attractive to guests, lead to revenue loss and increased operational costs for the hotel.

The **Revenue Management and Operations teams** raised concerns after noticing an uptick in unoccupied rooms that were previously reserved. These no-shows and cancellations impact multiple areas: unutilized room inventory, last-minute discounting, and resource allocation for guest arrangements. With most reservations now occurring through online channels, the hotel needed a data-driven solution to anticipate cancellations and take preventive action.

---

## Objective  

To build a predictive model that identifies in advance which bookings are likely to be canceled, allowing INN Hotels’ **Revenue Management and Operations teams** to proactively manage inventory, reduce financial losses, and improve operational planning. Specifically, the project aims to:

- ¿What are the main factors that influence whether a booking is likely to be canceled?  
- ¿Can we accurately predict which reservations are at high risk of cancellation before check-in?  
- ¿How can the insights from the model inform overbooking strategies, refund policies, and room reallocation decisions?

The insights generated will help the Revenue team adjust pricing and inventory decisions, while the Operations team can better allocate resources and minimize disruptions caused by last-minute cancellations.

---
## Data Structure 

The dataset contains historical booking records from INN Hotels and is used to build a predictive model for reservation cancellations. Each row corresponds to a **single hotel booking made by a customer**. The dataset includes **36,275 rows** and **19 columns**, combining both categorical and numerical variables. The final cleaned version contains **no missing values or duplicate entries**, making it suitable for supervised classification tasks.

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
# Exploratory Data Analysis (EDA)

### Univariate Analysis 

**Lead Time** 

The variable measures the number of days between the booking date and the arrival date.

- The distribution is right-skewed, with most bookings made within 0 to 50 days. The mode is around 0–10 days, indicating a **high volume of last-minute reservations**.

- The median is 62 days, but the boxplot reveals a long right tail and several outliers beyond 300 days, likely from **early planners** or **group booking**s.

**Interpretation:**

Short lead times may reflect impulsive bookings, potentially linked to higher cancellation risk. This feature is likely to play a significant role in predicting cancellations and should be examined against booking_status.

<div align="center">
  <img src="https://github.com/user-attachments/assets/81622a50-258e-4671-8ae5-c78bcc755eb7" width="500"/>
</div>

--- 

**Market Segment Type**

- The variable indicates the source channel through which the booking was made.

- The majority of reservations came from the Online segment 23,214 bookings, followed by Offline 10,528. Together, they represent over 90% of the customers.

- Given their limited volume, both Corporate bookings and the Complementary and Aviation channels will be discussed further in the dedicated sections below.

- The distribution suggests a strong dependency on online platforms, which may influence booking behavior and cancellation trends.

**Interpretation:**
Given the dominance of online bookings, this channel could play a critical role in cancellation patterns. It should be further explored in a bivariate analysis with booking_status to assess whether certain channels are more prone to cancellations than others.

<div align="center">
  <img src="https://github.com/user-attachments/assets/024d62a5-0534-4c02-81bc-57745b315f0b" width="500"/>
</div>

---

**Average Price per Room**

The avg_price_per_room represents the average nightly rate charged for each booking, in euros.

- The distribution is slightly right-skewed, with most room prices concentrated between 60 and 140 euros.

- The median price is just under 105 euros, while a small number of high-value outliers extend beyond 300 euros, as seen in the boxplot.

- There are also minor spikes near 0 and at common rounded values, possibly from promotional rates or encoding artifacts.

**Interpretation:**

Most bookings fall within a competitive pricing range, with higher prices occurring less frequently. Price sensitivity may influence cancellation behavior, especially for higher-rate bookings. For this reason this variable could be relevant for modeling how price impacts cancellation likelihood and revenue recovery strategies.

<div align="center">
  <img src="https://github.com/user-attachments/assets/32e4a1ed-745a-411c-8f09-8340f8e80479" width="500"/>
</div>

### Bivariate Analysis

**Average Price per Room vs Market Segment Type**

The boxplot compares the distribution of avg_price_per_room across different market_segment_type categories.

- Online and Offline segments show similar price ranges, but Online bookings have higher median prices and more extreme high end outliers above 200 EUR, suggesting broader price variation in that channel.

- Corporate bookings are more concentrated, with moderate prices and fewer outliers.

- Aviation prices cluster tightly around ~100 EUR with minimal variation, likely reflecting fixed rate contracts.

- Complementary bookings have a median close to 0 EUR, consistent with free or promotional stays.

**Interpretation:**

Room pricing varies significantly across booking channels. The Online segment shows the widest variability, which may reflect flexible pricing algorithms or discount driven behavior. These pricing differences may influence both customer profiles and cancellation likelihood, and should be considered when segmenting the model.

<div align="center">
  <img src="https://github.com/user-attachments/assets/50b0053b-0688-430e-8b44-bac163029a92" width="500"/>
</div>

**Special Requests vs. Booking Status**

This stacked bar chart shows the proportion of canceled (1) and not canceled (0) bookings by number of special requests made.

- Guests with zero special requests have the highest cancellation rate, with over 40% of those bookings ending in cancellation.

- As the number of special requests increases, the cancellation rate drops. From 3 or more requests, no cancellations are observed.

- There’s a clear negative correlation: more engagement via requests appears to indicate stronger booking intent and lower risk of cancellation.

**Interpretation:**

Customers who make special requests are more invested in their stay and less likely to cancel. This variable is a strong behavioral indicator and should be treated as an important feature in the predictive model. It may also guide hotel staff in identifying low-risk guests for personalized service or upselling.

<div align="center">
  <img src="https://github.com/user-attachments/assets/d197c73e-4c17-4209-813a-2277af98d0a4" width="500"/>
</div> 

**Month of Arrival vs. Booking Status**

This stacked bar chart shows the proportion of canceled and non-canceled bookings by month of arrival where canceled (1) and not canceled (0).

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

**My interpretation:**
- High lead times may correspond to early-planned group or seasonal reservations.
- Extreme price values could reflect peak-season rates or luxury room types.
- Multiple special requests or prior cancellations could indicate loyal but demanding customers.
- Outlier counts in previous bookings may come from frequent guests or business travelers

<div style="display: flex; justify-content: center; gap: 20px;">

  <img src="https://github.com/user-attachments/assets/4f263720-86ef-4bc5-a6da-a206b068dd24" width="500"/>

  <img src="https://github.com/user-attachments/assets/ad5f63a6-fa8c-4cfe-8b97-1a4300efe337" width="400"/>

</div>

---

## Modeling Logistic Regression

**Data Preparation**

Before training the logistic regression model, the dataset was preprocessed by applying **one-hot encoding** to categorical features (with drop_first=True) and splitting the data into training 70% and test 30% sets. The target variable was **booking_status** (1 = Canceled, 0 = Not Canceled). The class distribution remained consistent across both sets 33% cancellations, ensuring a balanced and representative split for modeling.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b086ad68-b04d-4e55-81e0-6e10a744640a" width="300"/>
</div>

--- 

## Building Logistic Regression Model

The initial results showed an accuracy of 80.6%, a recall of 63.4%, and an F1-score of 68.3% on the training set. While the model demonstrated solid performance, the presence of multicollinearity and high p-values in several features suggested the need of treatment. 

<div style="display: flex; justify-content: center; gap: 20px;">

  <img src="https://github.com/user-attachments/assets/33beaf2e-8be4-4183-8426-d6ef8c6989d3" width="500"/>
  
  <img src="https://github.com/user-attachments/assets/c7bbbf73-f825-4de7-8302-02b012826cea" width="300"/>
  
</div>

**Test for multicollinearity and treat high p-values**

To improve model the multicollinearity was assessed using the Variance Inflation Factor (VIF). Several dummy variables, particularly within the market_segment_type feature, showed high VIF values (above 60), indicating strong linear dependencies. In parallel, backward elimination was applied using p-values from the logistic regression summary to remove statistically insignificant variables (p > 0.05). This iterative process reduced the feature set to only those with meaningful contribution to the model, resolving multicollinearity and improving overall model.  

**Final Model**

After resolving multicollinearity and removing statistically insignificant features based on p-values, a new logistic regression model was trained using the reduced set of predictors.The model achieved a accuracy of 80.57%, precision 73.94%, recall 63.33% and F1-Score of 68.23%.

<div style="display: flex; justify-content: center; gap: 20px;">

  <img src="https://github.com/user-attachments/assets/bab10a2a-542c-4571-8429-831db17b5991" width="500"/>

  <img src="https://github.com/user-attachments/assets/e8bcf8e2-e762-49d6-934a-d1153991cbfb" width="350"/>

</div>

**Coefficients to Odds**

To make the logistic regression results more interpretable, the model coefficients were converted from log-odds to odds ratios using the exponential transformation.This allows us to understand the impact of each predictor on the likelihood of cancellation in percentage terms.


---

**Threshold Optimization and Model Refinement**

<div align="justify">

To improve the model's ability to correctly identify canceled bookings, we explored threshold tuning using both the ROC Curve and the Precision-Recall Curve. Logistic regression models predict probabilities, and by default, a **threshold of 0.5** is used to classify outcomes. However, this threshold may not yield the best balance between precision and recall particularly in imbalanced datasets or use cases where one metric is more critical.

First we plotted the ROC curve, which showed an AUC of 0.86, indicating strong separability between canceled and non canceled bookings. By analyzing the curve, we identified an **optimal threshold** of approximately **0.37**, where the true positive rate was significantly improved without drastically increasing the false positive rate.

To further validate the threshold, we used the **Precision Recall Curve**, which is useful when dealing with imbalanced classes like ours. The curve clearly showed the tradeoff between precision and recall, and we identified an additional **optimal threshold** at **0.42**, where recall was maximized without sacrificing too much precision.

By adjusting the threshold from the default 0.5 to a more optimal value 0.37–0.42, the **model's recall** our key metric in this business case **improved substantially**, allowing the hotel to better anticipate booking cancellations, which is a key priority.


</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/136a04e7-ec1d-43fd-8631-1d722240862e" width="400"/>
  <img src="https://github.com/user-attachments/assets/69bb651f-9a8e-49a1-a236-cf986250c75f" width="400"/>
</p>

**Model performance summary**

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
