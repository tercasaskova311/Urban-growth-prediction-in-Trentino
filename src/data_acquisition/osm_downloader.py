"""
OpenStreetMap data acquisition module.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import pandas as pd
from shapely.geometry import box, Polygon

if TYPE_CHECKING:
    import osmnx as ox
    import geopandas as gpd

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    ox = None  # type: ignore
    print("Warning: OSMnx not installed. OSM features will be limited.")

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None  # type: ignore
    print("Warning: GeoPandas not installed. Spatial features will be limited.")


class OSMDataDownloader:
    """
    Class for downloading and processing OpenStreetMap data.
    """
    
    def __init__(self, bbox: List[float]):
        """
        Initialize the OSM data downloader.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        """
        if not HAS_OSMNX:
            raise ImportError("OSMnx is required. Install with: pip install osmnx")
        if not HAS_GEOPANDAS:
            raise ImportError("GeoPandas is required. Install with: pip install geopandas")
        
        self.bbox = bbox
        self.polygon = box(*bbox)
        
    def download_buildings(self) -> "gpd.GeoDataFrame":
        """
        Download building footprints from OSM.
        
        Returns:
            GeoDataFrame containing building geometries and attributes
        """
        try:
            tags = {'building': True}
            buildings = ox.features_from_bbox(
                bbox=(self.bbox[3], self.bbox[1], self.bbox[2], self.bbox[0]),
                tags=tags
            )
            
            # Filter only polygon geometries
            buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            # Calculate building area
            buildings['area'] = buildings.geometry.area
            
            print(f"Downloaded {len(buildings)} buildings")
            return buildings
            
        except Exception as e:
            print(f"Error downloading buildings: {e}")
            return gpd.GeoDataFrame()
    
    def download_roads(self, network_type: str = 'all') -> "gpd.GeoDataFrame":
        """
        Download road network from OSM.
        
        Args:
            network_type: Type of road network ('all', 'drive', 'walk', 'bike')
            
        Returns:
            GeoDataFrame containing road geometries and attributes
        """
        try:
            # Create graph from bounding box
            G = ox.graph_from_bbox(
                bbox=(self.bbox[3], self.bbox[1], self.bbox[2], self.bbox[0]),
                network_type=network_type,
                simplify=True
            )
            
            # Convert to GeoDataFrame
            nodes, edges = ox.graph_to_gdfs(G)
            
            # Calculate road length
            edges['length_m'] = edges.geometry.length
            
            print(f"Downloaded {len(edges)} road segments")
            return edges
            
        except Exception as e:
            print(f"Error downloading roads: {e}")
            return gpd.GeoDataFrame()
    
    def download_landuse(self) -> "gpd.GeoDataFrame":
        """
        Download land use data from OSM.
        
        Returns:
            GeoDataFrame containing land use geometries and attributes
        """
        try:
            tags = {'landuse': True}
            landuse = ox.features_from_bbox(
                bbox=(self.bbox[3], self.bbox[1], self.bbox[2], self.bbox[0]),
                tags=tags
            )
            
            # Filter only polygon geometries
            landuse = landuse[landuse.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            # Calculate area
            landuse['area'] = landuse.geometry.area
            
            print(f"Downloaded {len(landuse)} land use polygons")
            return landuse
            
        except Exception as e:
            print(f"Error downloading land use: {e}")
            return gpd.GeoDataFrame()
    
    def calculate_building_density(self, grid_size: float = 1000) -> "gpd.GeoDataFrame":
        """
        Calculate building density on a grid.
        
        Args:
            grid_size: Size of grid cells in meters
            
        Returns:
            GeoDataFrame with building density per grid cell
        """
        buildings = self.download_buildings()
        
        if buildings.empty:
            return gpd.GeoDataFrame()
        
        # Create grid
        grid = self._create_grid(grid_size)
        
        # Calculate building count and area per grid cell
        grid['building_count'] = 0
        grid['building_area'] = 0.0
        
        for idx, cell in grid.iterrows():
            intersecting = buildings[buildings.intersects(cell.geometry)]
            grid.at[idx, 'building_count'] = len(intersecting)
            grid.at[idx, 'building_area'] = intersecting['area'].sum()
        
        # Calculate density (buildings per km²)
        grid['building_density'] = grid['building_count'] / (grid_size * grid_size / 1e6)
        
        return grid
    
    def calculate_road_density(self, grid_size: float = 1000) -> "gpd.GeoDataFrame":
        """
        Calculate road density on a grid.
        
        Args:
            grid_size: Size of grid cells in meters
            
        Returns:
            GeoDataFrame with road density per grid cell
        """
        roads = self.download_roads()
        
        if roads.empty:
            return gpd.GeoDataFrame()
        
        # Create grid
        grid = self._create_grid(grid_size)
        
        # Calculate road length per grid cell
        grid['road_length'] = 0.0
        
        for idx, cell in grid.iterrows():
            intersecting = roads[roads.intersects(cell.geometry)]
            grid.at[idx, 'road_length'] = intersecting['length_m'].sum()
        
        # Calculate density (km of roads per km²)
        grid['road_density'] = grid['road_length'] / (grid_size * grid_size / 1e6) / 1000
        
        return grid
    
    def _create_grid(self, grid_size: float) -> "gpd.GeoDataFrame":
        """
        Create a grid covering the bounding box.
        
        Args:
            grid_size: Size of grid cells in meters
            
        Returns:
            GeoDataFrame with grid cells
        """
        from shapely.geometry import box
        import numpy as np
        
        # Convert bbox to meters (approximate)
        min_lon, min_lat, max_lon, max_lat = self.bbox
        
        # Calculate grid dimensions
        x_cells = int((max_lon - min_lon) * 111320 / grid_size)  # Approximate conversion
        y_cells = int((max_lat - min_lat) * 111320 / grid_size)
        
        # Create grid cells
        cells = []
        for i in range(x_cells):
            for j in range(y_cells):
                x_min = min_lon + i * (max_lon - min_lon) / x_cells
                x_max = min_lon + (i + 1) * (max_lon - min_lon) / x_cells
                y_min = min_lat + j * (max_lat - min_lat) / y_cells
                y_max = min_lat + (j + 1) * (max_lat - min_lat) / y_cells
                
                cells.append({
                    'geometry': box(x_min, y_min, x_max, y_max),
                    'cell_id': f"{i}_{j}"
                })
        
        return gpd.GeoDataFrame(cells, crs="EPSG:4326")
    
    def save_to_file(self, gdf: "gpd.GeoDataFrame", filename: str):
        """
        Save GeoDataFrame to file.
        
        Args:
            gdf: GeoDataFrame to save
            filename: Output filename (supports .shp, .geojson, .gpkg)
        """
        try:
            gdf.to_file(filename)
            print(f"Saved data to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
