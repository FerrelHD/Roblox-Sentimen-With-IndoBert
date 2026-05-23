import pandas as pd
import os

def run_analysis(input_file='data/processed/roblox_sentiment.csv'):
    """
    Perform time-based sentiment analysis:
    - Group by Month (YYYY-MM)
    - Group by Sentiment
    - Calculate counts and percentages
    """
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} tidak ditemukan.")
        return None
    
    print(f"Menganalisis tren dari {input_file}...")
    df = pd.read_csv(input_file)
    
    # 1. Convert 'at' (date) to datetime and extract YYYY-MM
    df['at'] = pd.to_datetime(df['at'])
    df['month'] = df['at'].dt.to_period('M').astype(str)
    
    # 2. Group by month and sentiment
    monthly_sentiment = df.groupby(['month', 'sentiment']).size().reset_index(name='count')
    
    # 3. Calculate total reviews per month for percentage
    monthly_total = df.groupby(['month']).size().reset_index(name='total')
    
    # 4. Merge and calculate percentage
    analysis_df = pd.merge(monthly_sentiment, monthly_total, on='month')
    analysis_df['percentage'] = (analysis_df['count'] / analysis_df['total']) * 100
    
    print("\nHasil Analisis Tren (Preview):")
    print(analysis_df.head(10))
    
    # Save results to a summary file
    summary_path = 'data/processed/sentiment_analysis_summary.csv'
    analysis_df.to_csv(summary_path, index=False)
    print(f"\nHasil analisis tren disimpan di {summary_path}")
    
    return analysis_df

if __name__ == "__main__":
    run_analysis()
