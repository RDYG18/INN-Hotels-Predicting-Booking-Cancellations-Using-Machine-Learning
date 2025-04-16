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

**Summary of the key fields:**
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

<div align="center">
  <img src="https://github.com/user-attachments/assets/81622a50-258e-4671-8ae5-c78bcc755eb7" width="600"/>
</div>

lead_time measures the number of days between the booking date and the arrival date.

The distribution is right-skewed, with most bookings made within 0 to 50 days. The mode is around 0–10 days, indicating a high volume of last-minute reservations.

The median is ~62 days, but the boxplot reveals a long right tail and several outliers beyond 300 days, likely from early planners or group bookings.

This skew suggests the potential need for log transformation or binning before modeling.

Interpretation:
Short lead times may reflect impulsive bookings, potentially linked to higher cancellation risk. This feature is likely to play a significant role in predicting cancellations and should be examined against booking_status.


