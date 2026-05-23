import pandas as pd
import time
from datetime import datetime
from google_play_scraper import reviews, Sort
import os

def scrape_roblox_reviews(target_count=50000, start_date="2026-01-01", end_date="2026-04-30"):
    """
    Scrape reviews for Roblox from Google Play Store.
    
    Args:
        target_count (int): Maximum number of reviews to collect.
        start_date (str): The earliest date for reviews (YYYY-MM-DD).
        end_date (str): The latest date for reviews (YYYY-MM-DD).
    """
    app_id = 'com.roblox.client'
    all_reviews = []
    continuation_token = None
    
    # Convert dates to datetime objects for comparison
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"Memulai scraping reviews untuk {app_id}...")
    print(f"Target: {target_count} data dari {start_date} hingga {end_date}")

    count = 0
    while count < target_count:
        # Fetch reviews
        result, continuation_token = reviews(
            app_id,
            lang='id', # Bahasa Indonesia
            country='id', # Indonesia
            sort=Sort.NEWEST, # Get newest first to check date range
            count=1000, # Max per request
            continuation_token=continuation_token
        )
        
        if not result:
            print("Tidak ada review lagi yang ditemukan.")
            break
            
        for review in result:
            review_at = review['at']
            
            # Check if review is within our target date range
            if review_at < start_dt:
                print(f"Mencapai tanggal sebelum {start_date}. Berhenti.")
                count = target_count # To break outer loop
                break
            
            if review_at <= end_dt:
                all_reviews.append(review)
                count += 1
                
                # Backup every 2000 data
                if count % 2000 == 0:
                    temp_df = pd.DataFrame(all_reviews)
                    temp_df.to_csv(f'data/raw/roblox_raw_backup_{count}.csv', index=False)
                    print(f"Backup data: {count} data tersimpan.")
            
            if count >= target_count:
                break
        
        print(f"Progress: {count}/{target_count} data terkumpul.")
        
        if not continuation_token:
            break
            
        # Delay to avoid rate limiting
        time.sleep(2)
    
    # Save final result
    df = pd.DataFrame(all_reviews)
    output_path = 'data/raw/roblox_raw.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Scraping selesai. Total data: {len(df)}. Tersimpan di {output_path}")

if __name__ == "__main__":
    # Note: In a real scenario, we might want to handle exceptions
    scrape_roblox_reviews()
