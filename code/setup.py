# System
import copy
import csv
import sys
import os
import watermark
import dill as pickle
import itertools
import random
import zipfile
from collections import defaultdict
import pprint
pp = pprint.PrettyPrinter(indent=4)
from tqdm.notebook import tqdm
import warnings
import shutil
from pathlib import Path
from collections import defaultdict
import requests



# Math/Data
import math
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.neighbors import NearestNeighbors

# Network
import igraph as ig
import networkx as nx
from networkx.utils import pairwise

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.animation as animation


# Geo
import osmnx as ox
ox.settings.log_file = True
ox.settings.requests_timeout = 300
ox.settings.logs_folder = PATH["logs"]
import fiona
import shapely
from osgeo import gdal, osr
from haversine import haversine, haversine_vector
import pyproj
from shapely.geometry import Point, MultiPoint, LineString, Polygon, MultiLineString, MultiPolygon, shape, GeometryCollection
import shapely.ops as ops
from shapely.ops import unary_union
from shapely.ops import nearest_points
from shapely.plotting import plot_line
import geopandas as gpd
import geojson
import json
from owslib.wms import WebMapService
from rasterio.mask import mask as rio_mask  
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from rasterio.io import MemoryFile
from tesspy import Tessellation
import momepy
from pyproj import Transformer
import geopy
from geopy.distance import geodesic
#import ukcensusapi.Nomisweb as Api # not needed at the moment
import esda
from shapely.ops import polygonize




     
# dict of placeid:placeinfo
# If a city has a proper shapefile through nominatim
# In case no (False), manual download of shapefile is necessary, see below
cities = {}
with open(PATH["parameters"] + 'cities.csv') as f:
    csvreader = csv.DictReader(f, delimiter=';')
    for row in csvreader:
        cities[row['placeid']] = {}
        for field in csvreader.fieldnames[1:]:
            cities[row['placeid']][field] = row[field]     
if debug:
    print("\n\n=== Cities ===")
    pp.pprint(cities)
    print("==============\n\n")

# Create city subfolders  
for placeid, placeinfo in cities.items():
    for subfolder in ["data", "plots", "plots_networks", "results", "exports", "exports_json", "videos"]:
        placepath = PATH[subfolder] + placeid + "/"
        if not os.path.exists(placepath):
            os.makedirs(placepath)
            print("Successfully created folder " + placepath)

from IPython.display import Audio
sound_file = '../dingding.mp3'

print("Setup finished.\n")
