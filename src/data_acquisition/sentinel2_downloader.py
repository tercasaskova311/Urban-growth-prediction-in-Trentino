"""
Sentinel-2 data acquisition module using Google Earth Engine.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np
from datetime import datetime

if TYPE_CHECKING:
    import ee

try:
    import ee
    HAS_EE = True
except ImportError:
    HAS_EE = False
    ee = None  # type: ignore
    print("Warning: Google Earth Engine (ee) not installed. Sentinel-2 features will be limited.")


class Sentinel2Downloader:
    """
    Class for downloading and processing Sentinel-2 imagery from Google Earth Engine.
    """
    
    def __init__(self, bbox: List[float], start_date: str, end_date: str):
        """
        Initialize the Sentinel-2 downloader.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        if not HAS_EE:
            raise ImportError("Google Earth Engine (ee) is required. Install with: pip install earthengine-api")
        
        self.bbox = bbox
        self.start_date = start_date
        self.end_date = end_date
        self.region = ee.Geometry.Rectangle(bbox)
        
    def initialize_gee(self, project_id: Optional[str] = None):
        """
        Initialize Google Earth Engine authentication.
        
        Args:
            project_id: GEE project ID (optional)
        """
        try:
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            print("Google Earth Engine initialized successfully.")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            print("Please authenticate using: earthengine authenticate")
    
    def calculate_ndvi(self, image: "ee.Image") -> "ee.Image":
        """
        Calculate NDVI (Normalized Difference Vegetation Index).
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            Image with NDVI band
        """
        nir = image.select('B8')
        red = image.select('B4')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return image.addBands(ndvi)
    
    def calculate_ndbi(self, image: "ee.Image") -> "ee.Image":
        """
        Calculate NDBI (Normalized Difference Built-up Index).
        
        NDBI = (SWIR - NIR) / (SWIR + NIR)
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            Image with NDBI band
        """
        swir = image.select('B11')
        nir = image.select('B8')
        ndbi = swir.subtract(nir).divide(swir.add(nir)).rename('NDBI')
        return image.addBands(ndbi)
    
    def calculate_ndwi(self, image: "ee.Image") -> "ee.Image":
        """
        Calculate NDWI (Normalized Difference Water Index).
        
        NDWI = (Green - NIR) / (Green + NIR)
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            Image with NDWI band
        """
        green = image.select('B3')
        nir = image.select('B8')
        ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
        return image.addBands(ndwi)
    
    def mask_clouds(self, image: "ee.Image", cloud_threshold: int = 20) -> "ee.Image":
        """
        Mask clouds using the SCL (Scene Classification) band.
        
        Args:
            image: Sentinel-2 image
            cloud_threshold: Maximum cloud cover percentage
            
        Returns:
            Cloud-masked image
        """
        scl = image.select('SCL')
        # Mask clouds (3), cloud shadows (8), and cirrus (9)
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
        return image.updateMask(mask)
    
    def get_image_collection(self, cloud_cover: int = 20) -> "ee.ImageCollection":
        """
        Get Sentinel-2 image collection for the specified region and time period.
        
        Args:
            cloud_cover: Maximum cloud cover percentage
            
        Returns:
            Filtered and processed image collection
        """
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterBounds(self.region)
                     .filterDate(self.start_date, self.end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)))
        
        # Calculate indices
        collection = collection.map(self.mask_clouds)
        collection = collection.map(self.calculate_ndvi)
        collection = collection.map(self.calculate_ndbi)
        collection = collection.map(self.calculate_ndwi)
        
        return collection
    
    def aggregate_temporal(self, collection: "ee.ImageCollection", 
                          period: str = 'month') -> "ee.ImageCollection":
        """
        Aggregate image collection by time period (monthly, quarterly, yearly).
        
        Args:
            collection: Image collection to aggregate
            period: Aggregation period ('month', 'quarter', 'year')
            
        Returns:
            Aggregated image collection
        """
        # Create time series list
        start = ee.Date(self.start_date)
        end = ee.Date(self.end_date)
        
        if period == 'month':
            n_periods = end.difference(start, 'month').round()
        elif period == 'quarter':
            n_periods = end.difference(start, 'month').divide(3).round()
        else:  # year
            n_periods = end.difference(start, 'year').round()
        
        # Create sequence of time periods
        sequence = ee.List.sequence(0, n_periods.subtract(1))
        
        def aggregate_period(n):
            n = ee.Number(n)
            if period == 'month':
                period_start = start.advance(n, 'month')
                period_end = period_start.advance(1, 'month')
            elif period == 'quarter':
                period_start = start.advance(n.multiply(3), 'month')
                period_end = period_start.advance(3, 'month')
            else:  # year
                period_start = start.advance(n, 'year')
                period_end = period_start.advance(1, 'year')
            
            # Filter and composite
            filtered = collection.filterDate(period_start, period_end)
            composite = filtered.median()
            
            return composite.set('system:time_start', period_start.millis())
        
        return ee.ImageCollection(sequence.map(aggregate_period))
    
    def export_to_geotiff(self, image: "ee.Image", filename: str, 
                         scale: int = 10, bands: Optional[List[str]] = None):
        """
        Export image to GeoTIFF format.
        
        Args:
            image: Image to export
            filename: Output filename
            scale: Resolution in meters
            bands: List of bands to export (optional)
        """
        if bands:
            image = image.select(bands)
        
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename,
            scale=scale,
            region=self.region,
            fileFormat='GeoTIFF',
            maxPixels=1e13
        )
        task.start()
        print(f"Export task started: {filename}")
        return task
    
    def get_time_series(self, point: Tuple[float, float], 
                       bands: List[str]) -> Dict[str, List]:
        """
        Extract time series for a specific point.
        
        Args:
            point: (longitude, latitude) tuple
            bands: List of bands to extract
            
        Returns:
            Dictionary with time series data
        """
        collection = self.get_image_collection()
        point_geom = ee.Geometry.Point(point)
        
        def extract_values(image):
            values = image.select(bands).reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point_geom,
                scale=10
            )
            return ee.Feature(None, values).set('time', image.date().millis())
        
        time_series = collection.map(extract_values)
        return time_series.getInfo()
