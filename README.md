![image](https://github.com/user-attachments/assets/1746928c-b069-4044-98f4-3219f1d70412)

Access our Streamlit [Dashboard Website](https://ua-reviews.streamlit.app/)

# United Airlines Review (Phase 1)

**End-to-end Analytics Project for United Airlines:** Analyze Customer Experience
This project will simulate a data team at United Airlines, from ETL to Business Intelligence and Data Science. We will extract real-time data from [Air Inequality](https://www.airlinequality.com/airline-reviews/united-airlines), and perform 4 types of modern analytics to find insightful recommendations.

**Example Analysis**
![image](https://github.com/user-attachments/assets/bfdfbd3b-16e3-4d7c-b478-797e1f05cdcf)


# United Airlines Review (Phase 2)

**RAG chatbot for United Airlines:** Analyze Customer Review with a RAG chatbot
The **RAG Chatbot for United Airlines** utilizes cutting-edge technology stacks like **Chroma DB** and **LangChain**. **Chroma DB** serves as the vector database, enabling efficient storage and retrieval of customer review embeddings. **LangChain** powers the chatbot's integration of retrieval and generative AI, allowing for contextually accurate responses. This combination ensures fast, intelligent analysis of customer reviews, enhancing United Airlines ability to address feedback and improve customer satisfaction.

**Example RAG response**
![image](https://github.com/user-attachments/assets/79e1406e-7acb-410d-8e1a-813ea0bc59df)

## About the data

### Data Source
[Air Inequality](https://www.airlinequality.com/airline-reviews/united-airlines) is a robust platform for assessing United Airlines service quality through diverse customer reviews and ratings. Providing insights into cabin comfort, amenities, and overall satisfaction, enables customers to share feedback. However, the data's reliance on customer surveys may introduce bias.


### Cleaned Data
- `Id`: Order of each review, where smaller numbers represent later reviews. (Ordinal, Int)
- `Date review`: The date when the review was conducted. (Ordinal, Date)
- `Day review`: The day of the week when the review was conducted. (Discrete, Int)
- `Month review`: The month when the review was conducted. (Ordinal, String)
- `Month review num`: The numerical representation of the month when the review was conducted. (Discrete, Int)
- `Year review`: The year when the review was conducted. (Discrete, Int)
- `Verified`: Indicates whether the review was successfully verified or not. (Nominal, Boolean)
- `Name`: Name of the passenger who provided the review. (Nominal, String)
- `Month fly`: The month of the flight date. (Ordinal, String)
- `Month fly num`: The numerical representation of the month of the flight date. (Discrete, Int)
- `Year fly`: The year of the flight date. (Discrete, Int)
- `Month year fly`: The month and year of the flight date. (Ordinal, Date)
- `Country`: Nationality of the passenger. (Nominal, String)
- `Aircraft_1`: The type of aircraft used for the first flight. (Nominal, String)
- `Aircraft_2`: The type of aircraft used for the second flight. (Nominal, String)
- `Aircraft_3`: The type of aircraft used for the third flight. (Nominal, String)
- `Is_return`: Can we go back? (Boolean)
- `Multi_leg`: Have we arrived already, and are we still flying ? (Boolean)
- `Type`: Purpose of the flight (e.g., business, leisure). (Nominal, String)
- `Seat Type`: Class of the seat (e.g., Economy, Business). (Ordinal, String)
- `Origin`: Departure location of the passenger. (Nominal, String)
- `Destination`: Final destination of the passenger. (Nominal, String)
- `Transit`: Location where the passenger was transited. (Nominal, String)
- `Seat Comfort`: Passenger’s evaluation of seat comfort on a scale of 1 to 5. (Ordinal, Int)
- `Cabin service`: Passenger’s evaluation of the cabin service on a scale of 1 to 5. (Ordinal, Int)
- `Ground service`: Passenger’s evaluation of the ground service on a scale of 1 to 5. (Ordinal, Int)
- `Wifi`: Passenger’s evaluation of the on-board wifi connection on a scale of 1 to 5. (Ordinal, Int)
- `Money value`: Passenger's evaluation of how the flight experience corresponds to the money paid on a scale of 1 to 5. (Ordinal, Int)
- `Score`: Average of the scores given by the passenger on seat comfort, cabin service, ground service, wifi, and money value. (Continuous, Float)
- `Experience`: Overall rating of the flight experience categorized into poor, fair, and good. (Ordinal, String)
- `Recommended`: Indicates whether the passenger would recommend the same flight experience. (Nominal, Boolean)
- `Review`: Detailed feedback provided by the passenger regarding their flight experience. (Nominal, String)
 
## Project Flow Chart

## Project Steps
### 1. Extract - Transform - Load (ETL) - Airline Quality ETL Pipeline

This repository contains the implementation of an Extract, Transform, Load (ETL) pipeline that periodically scrapes customer reviews for United Airlines from AirlineQuality.com. The data is processed and used for analytical and machine-learning purposes. The pipeline is designed within the AWS Cloud environment, leveraging a combination of AWS S3 for robust, scalable, and efficient data handling.

![image](https://github.com/user-attachments/assets/975a0b38-a8e7-4b44-bd28-846b02d46547)

#### 1.1 Data Extraction (extract.py):
The `Extract.py` script scrapes real-time reviews from the United Airlines page on the Airline Quality website using BeautifulSoup and requests. It iterates through specified pages, extracting review details like date, customer name, country, review body, and ratings on various aspects of the airline service. The extracted data is stored in a pandas DataFrame and saved to a CSV file named `raw_data.csv`.

#### 1.2 Data Cleaning (data_cleaning.py):
The `data_cleaning.py` script preprocesses data using pandas. It removes parentheses from country names, splits review bodies into 'review' and 'verified' columns, and parses date information. The script standardizes column names, reorders them, and saves the cleaned data to `clean_data.csv`.

#### 1.3 Feature Engineering and Transformation (feature_engineering.py)
The `clean_data.py` script preprocesses data for analysis. It calculates an overall score, cleans the 'route' column into 'origin', 'destination', and 'transit', splits aircraft types, standardizes names, and categorizes customer experience. It also converts 'Yes' and 'No' values to boolean and reorders columns. This script streamlines data cleaning and expansion, saving results to the `clean_data_expand.csv`.

#### 1.4 Why This Pipeline?
The purpose of this ETL pipeline is to automate the collection and preprocessing of valuable airline customer feedback data. Analyzing customer reviews can reveal insights into overall customer satisfaction, service quality, and areas needing improvement. By scheduling the pipeline to run weekly, we can track changes in customer sentiment over time, allowing for timely data-driven decisions.

### 2. Data Analysis
#### 2.1: Exploratory Data Analysis (EDA.ipynb)
The `EDA.ipynb` file explored the United Airlines flight dataset through data preprocessing, general analysis of null values and score distributions, service rating analysis with visualizations, sentiment analysis on reviews, and correlation heatmaps. It investigated poor ground experiences for economy class, highlighting issues at London/Heathrow. For non-economy, it focused on food and seat comfort, comparing ratings across segments like recommended/non-recommended flights. Time series analysis visualized score and review count trends over monthly/yearly periods. The analysis uncovered data quality insights, service performance factors, experience elements impacting ratings, and potential seasonal effects to guide further modeling efforts.
 
* Hypothesis: Economy type tends to care more about Staff, while non-economy care more about Food and Seat comfort
  * For first class and economy class: Seat and comfort are not up to expectations
 
#### 2.2: Review Analysis (review_analysis.ipynb)
Review/Sentiment Analysis in Python

### 3. Predictive Modelling and Feature Engineering
#### 3.1 Feature importance selection
The MoneyValueModel determines the top 3 most important factors that affect a customer's MoneyValue:
* Economy Type: Staff
* Non-Economy: Food, Seat comfort

This will further confirm our Hypothesis when doing EDA. 
#### 3.2 Classify London Staff Review 
In this model, we will use Natural Language Processing (NLP) to classify London Staff problems into 3 categories: Staff's attitude, Lack of Staff, and Others

### 4. Streamlit app building
Here's an overview dashboard app 
* [Review app](https://ua-reviews.streamlit.app/)
