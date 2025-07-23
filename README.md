# Pricing Strategy Analysis

## Summary
This project analyzes e-commerce sales data from the Kaggle dataset "Unlock Profits with E-commerce Sales Data". It performs:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) including price distribution and demand curves
- Price elasticity estimation at both SKU and category levels
- Simulation of various pricing and demand scenarios to forecast revenue changes
- Visualization of key findings

## Setup Instructions
1. Clone or unzip the repository.
2. Ensure you have Python 3.7+ installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Setup your kaggle key and username for the kaggle setup hub to download the data
```python
KAGGLE_USERNAME = ""                 # can be blank if key-only tokens work
KAGGLE_KEY      = ""
```
4. When you will run the code, it will automatically download the dataset
   ```
   .
   ├── data/
   │   └── Amazon Sale Report.csv
   └── main.py
   ```

## How to Run
Run the analysis script:
```bash
python main.py
```
The script will output key tables to the console and generate plots in the working directory.

