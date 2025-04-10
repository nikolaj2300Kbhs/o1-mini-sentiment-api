from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import requests
import time
import os

# Folder path
input_folder = '/content/drive/My Drive/Chat GPT 4o files Jan to Oct 2024'

# Debug: List files in the folder to confirm
print("Files in folder:")
for file in os.listdir(input_folder):
    print(file)

# Load historical files
box_ratings = pd.read_excel(f'{input_folder}/Total box rating Jan-Oct24.xlsx')
product_info = pd.read_excel(f'{input_folder}/Product Information total.xlsx')
product_info_alt = pd.read_excel(f'{input_folder}/Product information Jan to October 2024 incl. product description (2).xlsx')
brand_avg = pd.read_excel(f'{input_folder}/brand average review (1).xlsx')
category_avg = pd.read_excel(f'{input_folder}/Category rating (1).xlsx')
box_content = pd.read_excel(f'{input_folder}/Box content Classic Jan-October 2024 (1).xlsx')

# Debug: Print column names before renaming
print("\nbox_ratings columns before renaming:", box_ratings.columns.tolist())
print("box_content columns before renaming:", box_content.columns.tolist())
print("product_info columns before renaming:", product_info.columns.tolist())
print("brand_avg columns before renaming:", brand_avg.columns.tolist())
print("category_avg columns before renaming:", category_avg.columns.tolist())

# Debug: Print a sample of the new data in box_content
print("\nSample of box_content (first 5 rows):")
print(box_content.head())

# Standardize column names
box_ratings.rename(columns={'Box sku': 'box_sku', 'average rating': 'average_box_score'}, inplace=True)
box_content.rename(columns={'Box sku': 'box_sku'}, inplace=True)
product_info.rename(columns={'SKU': 'product_sku'}, inplace=True)
product_info_alt.rename(columns={'SKU': 'product_sku'}, inplace=True)
brand_avg.rename(columns={'brand': 'Brand', 'Average rating': 'brand_avg_rating'}, inplace=True)
category_avg.rename(columns={'Average rating': 'category_avg_rating'}, inplace=True)

# Debug: Print column names after renaming
print("\nbox_ratings columns after renaming:", box_ratings.columns.tolist())
print("box_content columns after renaming:", box_content.columns.tolist())
print("product_info columns after renaming:", product_info.columns.tolist())
print("brand_avg columns after renaming:", brand_avg.columns.tolist())
print("category_avg columns after renaming:", category_avg.columns.tolist())

# Merge historical data
merged_data = pd.merge(box_ratings, box_content, on='box_sku', how='left')
merged_data = pd.merge(merged_data, product_info, on='product_sku', how='left')
merged_data = pd.merge(merged_data, product_info_alt, on='product_sku', how='left', suffixes=('', '_alt'))
merged_data = pd.merge(merged_data, brand_avg, on='Brand', how='left')
merged_data = pd.merge(merged_data, category_avg, on='Category', how='left')

# Debug: Print merged_data columns before calculating global averages
print("\nmerged_data columns:", merged_data.columns.tolist())

# Fill missing ratings
global_brand_avg = 4.07
global_category_avg = category_avg['category_avg_rating'].mean()
merged_data['brand_avg_rating'] = merged_data['brand_avg_rating'].fillna(global_brand_avg)
merged_data['category_avg_rating'] = merged_data['category_avg_rating'].fillna(global_category_avg)

# Summarize historical data
historical_summary = []
for box_sku, group in merged_data.groupby('box_sku'):
    summary = (f"Box {box_sku}: {len(group)} products, Total Retail Value: €{group['Retail price €'].sum():.2f}, "
               f"Unique Categories: {group['Category'].nunique()}, Full-size: {(group['Size_category'] == 'Full size').sum()}, "
               f"Premium (>€20): {(group['Retail price €'] > 20).sum()}, Total Weight: {group['Weight'].sum():.2f}g, "
               f"Avg Brand Rating: {group['brand_avg_rating'].mean():.2f}, Avg Category Rating: {group['category_avg_rating'].mean():.2f}, "
               f"Historical Score: {group['average_box_score'].iloc[0] if pd.notna(group['average_box_score'].iloc[0]) else 'None'}")
    historical_summary.append(summary)
historical_data = '; '.join(historical_summary)

# Load future box data from Google Sheet (exported as .xlsx)
future_box_data = pd.read_excel(f'{input_folder}/New Box Data.xlsx')

# Standardize future box column names to match historical data
future_box_data.rename(columns={
    'Box sku': 'box_sku',
    'product SKU': 'product_sku',
    'Retail Price (€)': 'Retail price €',
    'Product size': 'Size_category'
}, inplace=True)

# Extract future box info
future_box_sku = future_box_data['box_sku'].iloc[0]
future_products = future_box_data['product_sku'].tolist()
future_box_info = (f"Future Box {future_box_sku}: {len(future_products)} products, "
                   f"Total Retail Value: €{future_box_data['Retail price €'].sum():.2f}, "
                   f"Unique Categories: {future_box_data['Category'].nunique()}, "
                   f"Full-size: {(future_box_data['Size_category'] == 'Full size').sum()}, "
                   f"Premium (>€20): {(future_box_data['Retail price €'] > 20).sum()}, "
                   f"Total Weight: {future_box_data['weight'].sum():.2f}g")

# API URL for Gemini 2.0 Flash
url = 'https://gemini-sentiment-api.onrender.com/predict_box_score'  # Replace with your actual Render URL

# Predict future box score
try:
    response = requests.post(url, json={'historical_data': historical_data, 'future_box_info': future_box_info})
    print("HTTP Status Code:", response.status_code)
    print("Raw API response:", response.text)
    result = response.json()
    print("Parsed API response:", result)
    predicted_score = result['predicted_box_score']
    print(f"{future_box_sku}: Gemini 2.0 Flash Predicted Score {predicted_score}")
except Exception as e:
    print(f"Error predicting future box with Gemini 2.0 Flash: {e}")
    raise Exception("API call failed—cannot proceed without predicted_score")

# Save result
results = [{'box_sku': future_box_sku, 'gemini_2_0_flash_predicted_box_score': predicted_score}]
results_df = pd.DataFrame(results)
results_df.to_csv(f'{input_folder}/gemini_2_0_flash_future_box_scores.csv', index=False)
print(f"Done—Gemini 2.0 Flash results saved to {input_folder}/gemini_2_0_flash_future_box_scores.csv!")
