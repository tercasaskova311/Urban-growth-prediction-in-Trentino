"""
Complete script to download Sentinel-2 data for Trentino urban analysis (2018-2025)
Fixed and tested version
"""

import ee
from .sentinel2_downloader import Sentinel2Downloader
import pandas as pd
from datetime import datetime
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Replace with your Google Earth Engine project ID
PROJECT_ID = 'youtubeapi-455317'

# Trentino bounding box [min_lon, min_lat, max_lon, max_lat]
# Updated to match your config.yaml
TRENTINO_BBOX = [10.4, 45.7, 12.0, 47.1]

# Time period
START_DATE = '2018-01-01'
END_DATE = '2025-09-30'

# Cloud cover threshold (%)
CLOUD_COVER = 20  # Matching your config

# Bands to export (important for urban analysis)
URBAN_BANDS = [
    'B2',    # Blue
    'B3',    # Green
    'B4',    # Red
    'B8',    # NIR (Near-Infrared)
    'B11',   # SWIR (Shortwave Infrared)
    'NDVI',  # Vegetation Index
    'NDBI',  # Built-up Index (KEY for urban analysis)
    'NDWI'   # Water Index
]

# Export resolution (meters)
SCALE = 20  # Using 20m for faster processing and smaller files

# Temporal aggregation
AGGREGATION_PERIOD = 'quarter'  # Options: 'month', 'quarter', 'year'

# Google Drive folder for exports
EXPORT_FOLDER = 'Trentino_Sentinel2'

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

def main():
    print("="*70)
    print("SENTINEL-2 DATA ACQUISITION FOR TRENTINO URBAN ANALYSIS")
    print("="*70)
    
    # Step 1: Initialize Google Earth Engine
    print("\n[1/7] Initializing Google Earth Engine...")
    try:
        ee.Initialize(project=PROJECT_ID)
        print("✓ GEE initialized successfully")
        print(f"  Project: {PROJECT_ID}")
    except Exception as e:
        print(f"✗ Error initializing GEE: {e}")
        print("\nTroubleshooting:")
        print("1. Run: earthengine authenticate")
        print("2. Make sure your project ID is correct")
        print("3. Check you have Earth Engine enabled: https://console.cloud.google.com/")
        return
    
    # Step 2: Create downloader instance
    print("\n[2/7] Setting up Sentinel-2 downloader...")
    downloader = Sentinel2Downloader(
        bbox=TRENTINO_BBOX,
        start_date=START_DATE,
        end_date=END_DATE
    )
    print(f"✓ Downloader configured")
    print(f"  Region: Trentino, Italy")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Bbox: {TRENTINO_BBOX}")
    
    # Step 3: Get collection info
    print("\n[3/7] Checking available imagery...")
    try:
        info = downloader.get_collection_info()
        print(f"✓ Collection info retrieved")
        print(f"  Available images: {info['count']}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Step 4: Get image collection
    print("\n[4/7] Retrieving and filtering image collection...")
    collection = downloader.get_image_collection(cloud_cover=CLOUD_COVER)
    
    try:
        count = collection.size().getInfo()
        print(f"✓ Found {count} images with <{CLOUD_COVER}% cloud cover")
        
        if count == 0:
            print("✗ No images found!")
            print("  Try increasing CLOUD_COVER threshold (currently {CLOUD_COVER}%)")
            return
        
        if count < 10:
            print(f"⚠ Warning: Only {count} images found. Consider:")
            print("  - Increasing CLOUD_COVER threshold")
            print("  - Expanding the date range")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Step 5: Create temporal composites
    print(f"\n[5/7] Creating {AGGREGATION_PERIOD}ly composites...")
    try:
        aggregated = downloader.aggregate_temporal(collection, period=AGGREGATION_PERIOD)
        n_composites = aggregated.size().getInfo()
        print(f"✓ Created {n_composites} {AGGREGATION_PERIOD}ly composites")
        
        # Show what periods we have
        if n_composites > 0:
            first = ee.Image(aggregated.first())
            first_date = ee.Date(first.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            print(f"  First composite: {first_date}")
            
            last = ee.Image(aggregated.toList(n_composites).get(n_composites-1))
            last_date = ee.Date(last.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            print(f"  Last composite: {last_date}")
            
    except Exception as e:
        print(f"✗ Error creating composites: {e}")
        return
    
    # Step 6: Export composites
    print(f"\n[6/7] Exporting to Google Drive...")
    print(f"  Folder: {EXPORT_FOLDER}")
    print(f"  Bands: {', '.join(URBAN_BANDS)}")
    print(f"  Resolution: {SCALE}m")
    print(f"  Format: GeoTIFF")
    print()
    
    composite_list = aggregated.toList(n_composites)
    tasks = []
    
    for i in range(n_composites):
        try:
            image = ee.Image(composite_list.get(i))
            
            # Get timestamp
            timestamp = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            composite_count = image.get('composite_count').getInfo()
            
            # Create filename
            filename = f'trentino_s2_{AGGREGATION_PERIOD}_{timestamp}'
            
            # Start export
            task = downloader.export_to_geotiff(
                image=image,
                filename=filename,
                scale=SCALE,
                bands=URBAN_BANDS,
                folder=EXPORT_FOLDER
            )
            
            tasks.append((filename, task))
            print(f"  ✓ {filename} ({composite_count} images)")
            
        except Exception as e:
            print(f"  ✗ Error exporting composite {i+1}: {e}")
    
    print(f"\n✓ Successfully started {len(tasks)} export tasks!")
    
    # Step 7: Extract time series samples
    print(f"\n[7/7] Extracting time series for sample locations...")
    
    cities = {
        'Trento': (11.1210, 46.0664),
        'Rovereto': (11.0404, 45.8906),
        'Pergine': (11.2354, 46.0631),
        'Arco': (10.8861, 45.9188)
    }
    
    for city_name, coords in cities.items():
        try:
            print(f"  {city_name:15} ...", end=' ', flush=True)
            
            ts_data = downloader.get_time_series(
                point=coords,
                bands=['NDVI', 'NDBI', 'NDWI', 'B4', 'B8']
            )
            
            # Process and save
            features = ts_data.get('features', [])
            if features:
                data = []
                for feature in features:
                    props = feature['properties']
                    if 'time' in props:
                        # Only add if we have valid data
                        if props.get('NDBI') is not None:
                            data.append({
                                'timestamp': props['time'],
                                'date': pd.to_datetime(props['time'], unit='ms'),
                                'NDVI': props.get('NDVI'),
                                'NDBI': props.get('NDBI'),
                                'NDWI': props.get('NDWI'),
                                'B4_Red': props.get('B4'),
                                'B8_NIR': props.get('B8')
                            })
                
                if data:
                    df = pd.DataFrame(data).sort_values('date')
                    csv_filename = f'timeseries_{city_name.lower()}.csv'
                    df.to_csv(csv_filename, index=False)
                    print(f"✓ Saved {len(df)} records to {csv_filename}")
                else:
                    print("⚠ No valid data points")
            else:
                print("⚠ No features returned")
                
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
    
    # Print final summary
    print("\n" + "="*70)
    print("EXPORT SUMMARY")
    print("="*70)
    print(f"Region: Trentino, Italy")
    print(f"Coordinates: {TRENTINO_BBOX}")
    print(f"Time period: {START_DATE} to {END_DATE}")
    print(f"Total source images: {count}")
    print(f"Composites created: {n_composites} ({AGGREGATION_PERIOD}ly)")
    print(f"Bands per image: {len(URBAN_BANDS)}")
    print(f"Spatial resolution: {SCALE}m")
    print(f"Export destination: Google Drive/{EXPORT_FOLDER}/")
    print()
    
    # Estimate file sizes
    area_km2 = ((TRENTINO_BBOX[2] - TRENTINO_BBOX[0]) * 111) * \
               ((TRENTINO_BBOX[3] - TRENTINO_BBOX[1]) * 111)
    pixels_per_image = (area_km2 * 1e6) / (SCALE ** 2)
    mb_per_image = (pixels_per_image * len(URBAN_BANDS) * 4) / (1024**2)  # 4 bytes per float32
    total_size_gb = (mb_per_image * n_composites) / 1024
    
    print(f"Estimated size per image: ~{mb_per_image:.0f} MB")
    print(f"Total estimated size: ~{total_size_gb:.1f} GB")
    print("="*70)
    
    print("\n📊 NEXT STEPS:")
    print()
    print("1. Monitor exports at: https://code.earthengine.google.com/tasks")
    print("   - Tasks will show as RUNNING, then COMPLETED")
    print("   - Each task takes 10-30 minutes depending on size")
    print()
    print("2. Find exported files in Google Drive:")
    print(f"   - Look in folder: {EXPORT_FOLDER}/")
    print("   - Files will be named: trentino_s2_quarter_YYYY-MM-DD.tif")
    print()
    print("3. Download files to your computer")
    print()
    print("4. Load data for ML training using:")
    print("   - rasterio: for loading GeoTIFF files")
    print("   - numpy: for array operations")
    print("   - scikit-learn or pytorch: for ML models")
    
    print("\n💡 ML TIPS:")
    print("- NDBI (band 7) is your PRIMARY urban indicator")
    print("- Create change detection features: NDBI_t2 - NDBI_t1")
    print("- Consider normalizing spectral bands")
    print("- Use temporal features: mean, std, trend over time")
    
    print("\n⚠ TROUBLESHOOTING:")
    if count < 50:
        print("- Low image count: increase CLOUD_COVER threshold")
    print("- If exports fail: reduce SCALE (try 30m)")
    print("- Check GEE quota: https://code.earthengine.google.com/")
    
    print("\n" + "="*70)
    print("✓ Script completed successfully!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Script interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        print("Please check your configuration and try again")