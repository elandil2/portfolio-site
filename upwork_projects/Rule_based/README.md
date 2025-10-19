# Rule-Based Customer Segmentation and Revenue Prediction

## Project Overview

This project demonstrates a comprehensive rule-based customer segmentation system developed for **Based_Rules Company**, an imaginary gaming analytics firm. The system analyzes customer purchase data to create level-based customer personas and predict potential revenue from new customers.

## Business Problem

Based_Rules Company needed to understand their gaming customers better by creating level-based customer definitions (personas) using demographic features. The goal was to segment customers and predict how much revenue new customers with specific profiles might generate for the company.

**Example Use Case**: Determine how much revenue a 25-year-old male iOS user from Turkey might generate.

## Dataset Description

The dataset (`persona.csv`) contains transaction records from Based_Rules Company's gaming platform with the following features:

- **PRICE**: Customer's purchase amount (USD)
- **SOURCE**: Device type (android/ios)
- **SEX**: Customer gender (male/female)
- **COUNTRY**: Customer's country (USA, BRA, TUR, FRA, DEU, CAN)
- **AGE**: Customer's age

**Note**: The dataset is not deduplicated - multiple transactions per customer are possible.

## Methodology

### 1. Data Exploration and Analysis
- Comprehensive EDA with visualizations
- Statistical analysis of distributions and relationships
- Correlation analysis between variables

### 2. Customer Persona Creation
- Age categorization: 0-18, 19-23, 24-30, 31-40, 41-70
- Level-based persona format: `COUNTRY_SOURCE_SEX_AGE_CATEGORY`
- Example: `TUR_IOS_MALE_19_23`

### 3. Revenue Prediction Model
- Group customers by personas and calculate average revenue
- Segment customers into A/B/C tiers based on revenue potential
- Rule-based prediction system for new customers

### 4. Segmentation Strategy
- **Segment A**: High-value customers (top 33% by revenue)
- **Segment B**: Medium-value customers (middle 33%)
- **Segment C**: Low-value customers (bottom 33%)

## Key Findings

### Data Insights
- Total transactions: 5,000
- Unique customers: ~3,500 (estimated)
- Price range: $9 - $59
- Most common countries: Brazil, USA, Turkey
- Device preference: Android slightly more popular

### Revenue Segmentation
- **Segment A**: $35+ average revenue (High value)
- **Segment B**: $25-35 average revenue (Medium value)
- **Segment C**: <$25 average revenue (Low value)

### Top Performing Personas
1. FRA_ANDROID_FEMALE_24_30: $45.43
2. TUR_IOS_MALE_24_30: $45.00
3. TUR_IOS_MALE_31_40: $42.33

## Technologies Used

- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **Scikit-learn** - Machine learning framework
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Light gradient boosting framework
- **CatBoost** - Categorical boosting framework
- **Jupyter Notebook** - Development environment

## Project Structure

```
├── kural_tabanli_siniflandirma.py    # Main analysis script
├── persona.csv                       # Raw transaction data
├── README.md                         # Project documentation
├── *.png                             # Generated visualizations
└── index.html                        # Interactive presentation
```

## Usage

### Running the Analysis

```python
# Install required packages
pip install pandas numpy matplotlib seaborn

# Run the analysis
python kural_tabanli_siniflandirma.py
```

### Making Predictions

The system includes a prediction function for new customers:

```python
prediction = predict_customer_revenue('TUR', 'ios', 'male', 25, persona_df)
print(prediction)
```

## Sample Predictions

### Example 1: 33-year-old Turkish woman, Android user
- **Profile**: TUR_ANDROID_FEMALE_31_40
- **Expected Revenue**: $28.50
- **Segment**: B

### Example 2: 35-year-old French woman, iOS user
- **Profile**: FRA_IOS_FEMALE_31_40
- **Expected Revenue**: $32.75
- **Segment**: B

### Example 3: 25-year-old American man, Android user
- **Profile**: USA_ANDROID_MALE_24_30
- **Expected Revenue**: $41.20
- **Segment**: A

## Business Impact

This segmentation system enables Based_Rules Company to:

1. **Target High-Value Segments**: Focus marketing on A-segment customers
2. **Personalized Pricing**: Adjust pricing strategies by customer profile
3. **Revenue Forecasting**: Predict income from new customer acquisitions
4. **Resource Allocation**: Optimize customer support and development efforts
5. **Market Expansion**: Identify profitable customer demographics for expansion

## Machine Learning Exploration

Multiple ML algorithms (Random Forest, XGBoost, LightGBM, CatBoost) were explored to test predictive capabilities. The analysis revealed that demographic features alone have limited predictive power for exact prices in gaming contexts, where price depends more on product tier selection. This led to the conclusion that **rule-based segmentation is the optimal approach** for this business problem.

## Future Enhancements

- Real-time prediction API development
- Customer lifetime value (CLV) analysis
- A/B testing framework for pricing strategies
- Integration with CRM and marketing platforms
- Behavioral analytics and purchase pattern tracking
- Automated segment updating based on new data

## Author

Developed as part of a data science portfolio project demonstrating rule-based customer segmentation and predictive analytics capabilities.

## License

This project is for educational and portfolio purposes.