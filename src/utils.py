# import...

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
import glob
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from copy import deepcopy


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

##############################################################
# TODO: move this

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
scenario_folders = ["no_ltn_scenario", "more_ltn_scenario", "current_ltn_scenario"]
main_folders = ["data", "plots", "plots_networks", "results", "exports", "exports_json", "videos"]
for placeid in cities:
    for subfolder in main_folders:
        base_path = os.path.join(PATH[subfolder], placeid)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            print(f"Created folder: {base_path}")
        for scenario in scenario_folders:
            scenario_path = os.path.join(base_path, scenario)
            if not os.path.exists(scenario_path):
                os.makedirs(scenario_path)
                print(f"  └─ Created scenario folder: {scenario_path}")
##############################################################


# GRAPH PLOTTING

def holepatchlist_from_cov(cov, map_center):
    """Get a patchlist of holes (= shapely interiors) from a coverage Polygon or MultiPolygon
    """
    holeseq_per_poly = get_holes(cov)
    holepatchlist = []
    for hole_per_poly in holeseq_per_poly:
        for hole in hole_per_poly:
            holepatchlist.append(hole_to_patch(hole, map_center))
    return holepatchlist

def fill_holes(cov):
    """Fill holes (= shapely interiors) from a coverage Polygon or MultiPolygon
    """
    holeseq_per_poly = get_holes(cov)
    holes = []
    for hole_per_poly in holeseq_per_poly:
        for hole in hole_per_poly:
            holes.append(hole)
    eps = 0.00000001
    if isinstance(cov, shapely.geometry.multipolygon.MultiPolygon):
        cov_filled = ops.unary_union([poly for poly in cov] + [Polygon(hole).buffer(eps) for hole in holes])
    elif isinstance(cov, shapely.geometry.polygon.Polygon) and not cov.is_empty:
        cov_filled = ops.unary_union([cov] + [Polygon(hole).buffer(eps) for hole in holes])
    return cov_filled

def extract_relevant_polygon(placeid, mp):
    """Return the most relevant polygon of a multipolygon mp, for being considered the city limit.
    Depends on location.
    """
    if isinstance(mp, shapely.geometry.polygon.Polygon):
        return mp
    if placeid == "tokyo": # If Tokyo, take poly with most northern bound, otherwise largest
        p = max(mp, key=lambda a: a.bounds[-1])
    else:
        p = max(mp, key=lambda a: a.area)
    return p

def get_holes(cov):
    """Get holes (= shapely interiors) from a coverage Polygon or MultiPolygon
    """
    holes = []
    if isinstance(cov, shapely.geometry.multipolygon.MultiPolygon):
        for pol in cov.geoms: # cov is generally a MultiPolygon, so we iterate through its Polygons
            holes.append(pol.interiors)
    elif isinstance(cov, shapely.geometry.polygon.Polygon) and not cov.is_empty:
        holes.append(cov.interiors)
    return holes

def cov_to_patchlist(cov, map_center, return_holes = True):
    """Turns a coverage Polygon or MultiPolygon into a matplotlib patch list, for plotting
    """
    p = []
    if isinstance(cov, shapely.geometry.multipolygon.MultiPolygon):
        for pol in cov.geoms: # cov is generally a MultiPolygon, so we iterate through its Polygons
            p.append(pol_to_patch(pol, map_center))
    elif isinstance(cov, shapely.geometry.polygon.Polygon) and not cov.is_empty:
        p.append(pol_to_patch(cov, map_center))
    if not return_holes:
        return p
    else:
        holepatchlist = holepatchlist_from_cov(cov, map_center)
        return p, holepatchlist

def pol_to_patch(pol, map_center):
    """Turns a coverage Polygon into a matplotlib patch, for plotting
    """
    y, x = pol.exterior.coords.xy
    pos_transformed, _ = project_pos(y, x, map_center)
    return matplotlib.patches.Polygon(pos_transformed)

def hole_to_patch(hole, map_center):
    """Turns a LinearRing (hole) into a matplotlib patch, for plotting
    """
    y, x = hole.coords.xy
    pos_transformed, _ = project_pos(y, x, map_center)
    return matplotlib.patches.Polygon(pos_transformed)


def set_analysissubplot(key):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if key in ["length", "length_lcc", "coverage", "poi_coverage", "components", "efficiency_local", "efficiency_global"]:
        ax.set_ylim(bottom = 0)
    if key in ["directness_lcc", "directness_lcc_linkwise", "directness", "directness_all_linkwise"]:
        ax.set_ylim(bottom = 0.2)
    if key in ["directness_lcc", "directness_lcc_linkwise", "directness", "directness_all_linkwise", "efficiency_global", "efficiency_local"]:
        ax.set_ylim(top = 1)


def initplot():
    fig = plt.figure(figsize=(plotparam["bbox"][0]/plotparam["dpi"], plotparam["bbox"][1]/plotparam["dpi"]), dpi=plotparam["dpi"])
    plt.axes().set_aspect('equal')
    plt.axes().set_xmargin(0.01)
    plt.axes().set_ymargin(0.01)
    plt.axes().set_axis_off() # Does not work anymore - unnown why not.
    return fig

def nodesize_from_pois(nnids):
    """Determine POI node size based on number of POIs.
    The more POIs the smaller (linearly) to avoid overlaps.
    """
    minnodesize = 30
    maxnodesize = 220
    return max(minnodesize, maxnodesize-len(nnids))


def simplify_ig(G):
    """Simplify an igraph with ox.simplify_graph
    """
    G_temp = copy.deepcopy(G)
    G_temp.es["length"] = G_temp.es["weight"]
    output = ig.Graph.from_networkx(ox.simplify_graph(nx.MultiDiGraph(G_temp.to_networkx())).to_undirected())
    output.es["weight"] = output.es["length"]
    return output


def nxdraw(G, networktype, map_center = False, nnids = False, drawfunc = "nx.draw", nodesize = 0, weighted = False, maxwidthsquared = 0, simplified = False):
    """Take an igraph graph G and draw it with a networkx drawfunc.
    """
    if simplified:
        G.es["length"] = G.es["weight"]
        G_nx = ox.simplify_graph(nx.MultiDiGraph(G.to_networkx())).to_undirected()
    else:
        G_nx = G.to_networkx()
    if nnids is not False: # Restrict to nnids node ids
        nnids_nx = [k for k,v in dict(G_nx.nodes(data=True)).items() if v['id'] in nnids]
        G_nx = G_nx.subgraph(nnids_nx)
        
    pos_transformed, map_center = project_nxpos(G_nx, map_center)
    if weighted is True:
        # The max width should be the node diameter (=sqrt(nodesize))
        widths = list(nx.get_edge_attributes(G_nx, "weight").values())
        widthfactor = 1.1 * math.sqrt(maxwidthsquared) / max(widths)
        widths = [max(0.33, w * widthfactor) for w in widths]
        eval(drawfunc)(G_nx, pos_transformed, **plotparam[networktype], node_size = nodesize, width = widths)
    elif type(weighted) is float or type(weighted) is int and weighted > 0:
        eval(drawfunc)(G_nx, pos_transformed, **plotparam[networktype], node_size = nodesize, width = weighted)
    else:
        eval(drawfunc)(G_nx, pos_transformed, **plotparam[networktype], node_size = nodesize)
    return map_center



# OTHER FUNCTIONS

def common_entries(*dcts):
    """Like zip() but for dicts.
    See: https://stackoverflow.com/questions/16458340/python-equivalent-of-zip-for-dictionaries
    """
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

def project_nxpos(G, map_center = False):
    """Take a spatial nx network G and projects its GPS coordinates to local azimuthal.
    Returns transformed positions, as used by nx.draw()
    """
    lats = nx.get_node_attributes(G, 'x')
    lons = nx.get_node_attributes(G, 'y')
    pos = {nid:(lat,-lon) for (nid,lat,lon) in common_entries(lats,lons)}
    if map_center:
        loncenter = map_center[0]
        latcenter = map_center[1]
    else:
        loncenter = np.mean(list(lats.values()))
        latcenter = -1* np.mean(list(lons.values()))
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    pos_transformed = {nid:list(ops.transform(wgs84_to_aeqd.transform, Point(latlon)).coords)[0] for nid, latlon in pos.items()}
    return pos_transformed, (loncenter,latcenter)


def project_pos(lats, lons, map_center = False):
    """Project GPS coordinates to local azimuthal.
    """
    pos = [(lat,-lon) for lat,lon in zip(lats,lons)]
    if map_center:
        loncenter = map_center[0]
        latcenter = map_center[1]
    else:
        loncenter = np.mean(list(lats.values()))
        latcenter = -1* np.mean(list(lons.values()))
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    pos_transformed = [(ops.transform(wgs84_to_aeqd.transform, Point(latlon)).coords)[0] for latlon in pos]
    return pos_transformed, (loncenter,latcenter)


def round_coordinates(G, r = 7):
    for v in G.vs:
        G.vs[v.index]["x"] = round(G.vs[v.index]["x"], r)
        G.vs[v.index]["y"] = round(G.vs[v.index]["y"], r)

def mirror_y(G):
    for v in G.vs:
        y = G.vs[v.index]["y"]
        G.vs[v.index]["y"] = -y
    
def dist(v1, v2):
    dist = haversine((v1['y'],v1['x']),(v2['y'],v2['x']), unit="m") # x is lon, y is lat
    return dist

def dist_vector(v1_list, v2_list):
    dist_list = haversine_vector(v1_list, v2_list, unit="m") # [(lat,lon)], [(lat,lon)]
    return dist_list


def osm_to_ig(node, edge, weighting):
    """ Turns a node and edge dataframe into an igraph Graph.
    """
    G = ig.Graph(directed=False)
    x_coords = node['x'].tolist() 
    y_coords = node['y'].tolist()
    ids = node['osmid'].tolist()
    coords = []

    for i in range(len(x_coords)):
        G.add_vertex(x=x_coords[i], y=y_coords[i], id=ids[i])
        coords.append((x_coords[i], y_coords[i]))

    id_dict = dict(zip(G.vs['id'], np.arange(0, G.vcount()).tolist()))
    coords_dict = dict(zip(np.arange(0, G.vcount()).tolist(), coords))

    edge_list = []
    edge_info = {
        "weight": [],
        "osmid": [],
        # Only include ori_length if weighting is True
        "ori_length": [] if weighting else None  
    }
    
    for i in range(len(edge)):
        edge_list.append([id_dict.get(edge['u'][i]), id_dict.get(edge['v'][i])])
        edge_info["weight"].append(round(edge['length'][i], 10))
        edge_info["osmid"].append(edge['osmid'][i])
        
        if weighting:  # Only add ori_length if weighting is True
            edge_info["ori_length"].append(edge['ori_length'][i])  # Store the original length

    G.add_edges(edge_list)  # Add edges without attributes
    for i in range(len(edge)):
        G.es[i]["weight"] = edge_info["weight"][i]
        G.es[i]["osmid"] = edge_info["osmid"][i]
        
        if weighting:  # Set the original length only if weighting is True
            G.es[i]["ori_length"] = edge_info["ori_length"][i]

    G.simplify(combine_edges=max)
    return G


## Old 
# def osm_to_ig(node, edge, weighting=None):
#     """ Turns a node and edge dataframe into an igraph Graph. """
    
#     G = ig.Graph(directed=False)

#     # Print first few rows of edge dataframe
#     print("First 5 edges with lengths and maxspeeds:")
#     print(edge[['u', 'v', 'length', 'maxspeed']].head())

#     x_coords = node['x'].tolist() 
#     y_coords = node['y'].tolist()
#     ids = node['osmid'].tolist()
#     coords = []

#     for i in range(len(x_coords)):
#         G.add_vertex(x=x_coords[i], y=y_coords[i], id=ids[i])
#         coords.append((x_coords[i], y_coords[i]))

#     id_dict = dict(zip(G.vs['id'], np.arange(0, G.vcount()).tolist()))
#     coords_dict = dict(zip(np.arange(0, G.vcount()).tolist(), coords))

#     edge_list = []
#     edge_info = {"weight": [], "osmid": []}

#     if weighting:
#         print("Applying weighted calculation to edges.")
#         for i in range(len(edge)):
#             u, v = edge['u'][i], edge['v'][i]
#             edge_list.append([id_dict.get(u), id_dict.get(v)])
#             length = edge['length'][i]

#             try:
#                 speed_limit = int(str(edge['maxspeed'][i]).split()[0]) if pd.notnull(edge['maxspeed'][i]) else 30
#             except (ValueError, IndexError):
#                 speed_limit = 30

#             weight = (length * (speed_limit / 10)) * 10000
#             edge_info["weight"].append(round(weight, 10))
#             edge_info["osmid"].append(edge['osmid'][i])
#     else:
#         print("Applying unweighted calculation to edges.")
#         for i in range(len(edge)):
#             edge_list.append([id_dict.get(edge['u'][i]), id_dict.get(edge['v'][i])])
#             edge_info["weight"].append(round(edge['length'][i], 10))
#             edge_info["osmid"].append(edge['osmid'][i])

#     # Debug: Print edge list
#     #print("Edge list:", edge_list)

#     G.add_edges(edge_list)
    
#     # Check that the edge count matches
#     print(f"Number of edges in edge_list: {len(edge_list)}, edges in graph: {G.ecount()}")

#     for i in range(len(edge_list)):
#         G.es[i]["weight"] = edge_info["weight"][i]
#         G.es[i]["osmid"] = edge_info["osmid"][i]

#     # Debug: Print final edge weights
#     print("Final edge weights after assignment:")
#     print(G.es["weight"][:5])  # Check first few for validation

#     G.simplify(combine_edges=max)

#     # Assuming edges is a DataFrame or a list of your edges
#     for index, edge in enumerate(edges.itertuples()):
#         length = edge.length
#         speed_limit = edge.maxspeed
#         weight = length * (3600 / speed_limit)  # Calculate the weight based on length and speed

#         # Print only the first 15 edges
#         if index < 15:
#             print(f"Edge ID: {index}, Length: {length}, Speed limit: {speed_limit}, Calculated weight: {weight}")
#         # Add the weight to the graph here

#     return G



def compress_file(p, f, filetype = ".csv", delete_uncompressed = True):
    with zipfile.ZipFile(p + f + ".zip", 'w', zipfile.ZIP_DEFLATED) as zfile:
        zfile.write(p + f + filetype, f + filetype)
    if delete_uncompressed: os.remove(p + f + filetype)

def ox_to_csv(G, p, placeid, parameterid, postfix = "", compress = True, verbose = True):
    if "crs" not in G.graph:
        G.graph["crs"] = 'epsg:4326' # needed for OSMNX's graph_to_gdfs in utils_graph.py
    try:
        node, edge = ox.graph_to_gdfs(G)
    except ValueError:
        node, edge = gpd.GeoDataFrame(), gpd.GeoDataFrame()
    prefix = placeid + '_' + parameterid + postfix

    node.to_csv(p + prefix + '_nodes.csv', index = True)
    if compress: compress_file(p, prefix + '_nodes')
 
    edge.to_csv(p + prefix + '_edges.csv', index = True)
    if compress: compress_file(p, prefix + '_edges')

    if verbose: print(placeid + ": Successfully wrote graph " + parameterid + postfix)

def check_extract_zip(p, prefix):
    """ Check if a zip file prefix+'_nodes.zip' and + prefix+'_edges.zip'
    is available at path p. If so extract it and return True, otherwise False.
    If you call this function, remember to clean up (i.e. delete the unzipped files)
    after you are done like this:

    if compress:
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')
    """

    try: # Use zip files if available
        with zipfile.ZipFile(p + prefix + '_nodes.zip', 'r') as zfile:
            zfile.extract(prefix + '_nodes.csv', p)
        with zipfile.ZipFile(p + prefix + '_edges.zip', 'r') as zfile:
            zfile.extract(prefix + '_edges.csv', p)
        return True
    except:
        return False


def csv_to_ox(p, placeid, parameterid):
    """ Load a networkx graph from _edges.csv and _nodes.csv
    The edge file must have attributes u,v,osmid,length
    The node file must have attributes y,x,osmid
    Only these attributes are loaded.
    """
    prefix = placeid + '_' + parameterid
    compress = check_extract_zip(p, prefix)
    
    with open(p + prefix + '_edges.csv', 'r') as f:
        header = f.readline().strip().split(",")

        lines = []
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            line_list = [c for c in line]
            osmid = str(eval(line_list[header.index("osmid")])[0]) if isinstance(eval(line_list[header.index("osmid")]), list) else line_list[header.index("osmid")] # If this is a list due to multiedges, just load the first osmid
            length = str(eval(line_list[header.index("length")])[0]) if isinstance(eval(line_list[header.index("length")]), list) else line_list[header.index("length")] # If this is a list due to multiedges, just load the first osmid
            line_string = "" + line_list[header.index("u")] + " "+ line_list[header.index("v")] + " " + osmid + " " + length
            lines.append(line_string)
        G = nx.parse_edgelist(lines, nodetype = int, data = (("osmid", int),("length", float)), create_using = nx.MultiDiGraph) # MultiDiGraph is necessary for OSMNX, for example for get_undirected(G) in utils_graph.py
    with open(p + prefix + '_nodes.csv', 'r') as f:
        header = f.readline().strip().split(",")
        values_x = {}
        values_y = {}
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            line_list = [c for c in line]
            osmid = int(line_list[header.index("osmid")])
            values_x[osmid] = float(line_list[header.index("x")])
            values_y[osmid] = float(line_list[header.index("y")])

        nx.set_node_attributes(G, values_x, "x")
        nx.set_node_attributes(G, values_y, "y")

    if compress:
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')

    return G



def csv_to_ox_highway(p, placeid, parameterid): #  this is a modification of the orignal csv_to_ox function to include maxspeed
    """ Load a networkx graph from _edges.csv and _nodes.csv
    The edge file must have attributes u,v,osmid,maxspeed,highway,length
    The node file must have attributes y,x,osmid
    Only these attributes are loaded.
    """
    prefix = placeid + '_' + parameterid
    compress = check_extract_zip(p, prefix)
    
    with open(p + prefix + '_edges.csv', 'r') as f:
        header = f.readline().strip().split(",")

        lines = []
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            line_list = [c for c in line]
            osmid = str(eval(line_list[header.index("osmid")])[0]) if isinstance(eval(line_list[header.index("osmid")]), list) else line_list[header.index("osmid")] # If this is a list due to multiedges, just load the first osmid
            highway = line_list[header.index("highway")]
            length = str(eval(line_list[header.index("length")])[0]) if isinstance(eval(line_list[header.index("length")]), list) else line_list[header.index("length")] # If this is a list due to multiedges, just load the first osmid

            # Clean `maxspeed` to remove "mph" or other non-numeric characters
            maxspeed_raw = line_list[header.index("maxspeed")].strip()
            if not maxspeed_raw or not any(c.isdigit() for c in maxspeed_raw):  # Check for missing/invalid data
                maxspeed = "0"  # Default maxspeed
            else:
                maxspeed = ''.join(filter(str.isdigit, maxspeed_raw)) 

            line_string = "" + line_list[header.index("u")] + " "+ line_list[header.index("v")] + " " + osmid + " " + maxspeed + " " + highway + " " + length
            lines.append(line_string)
        G = nx.parse_edgelist(lines, nodetype = int, data = (("osmid", int),("maxspeed", int),("highway", str),("length", float)), create_using = nx.MultiDiGraph) # MultiDiGraph is necessary for OSMNX, for example for get_undirected(G) in utils_graph.py
    with open(p + prefix + '_nodes.csv', 'r') as f:
        header = f.readline().strip().split(",")
        values_x = {}
        values_y = {}
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            line_list = [c for c in line]
            osmid = int(line_list[header.index("osmid")])
            values_x[osmid] = float(line_list[header.index("x")])
            values_y[osmid] = float(line_list[header.index("y")])

        nx.set_node_attributes(G, values_x, "x")
        nx.set_node_attributes(G, values_y, "y")

    if compress:
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')
        
    return G




def calculate_weight(row):
    """
    Calculate new weight based on length and speed limit.
    """
    # Default speed limit is 30 mph if 'maxspeed' is missing or NaN
    if pd.isna(row['maxspeed']):
        speed_factor = 3  # Corresponding to 30 mph
    else:
        speed_factor = int(str(row['maxspeed']).split()[0][0])  # Extract first digit from the speed. 
        # This presumes no speed limit over 99, which is reasonable for most roads.
        # however this could produce issues in some countries with speed limits over 100 km/h?
    
    # Multiply the speed factor by the length to get the new weight
    return row['length'] * speed_factor







def csv_to_ig(p, placeid, parameterid, cleanup=True, weighting=None):
    """ Load an ig graph from _edges.csv and _nodes.csv
    The edge file must have attributes u,v,osmid,length
    The node file must have attributes y,x,osmid
    Only these attributes are loaded.
    """
    prefix = placeid + '_' + parameterid
    compress = check_extract_zip(p, prefix)
    empty = False
    try:
        n = pd.read_csv(p + prefix + '_nodes.csv')
        e = pd.read_csv(p + prefix + '_edges.csv')
    except:
        empty = True

    if compress and cleanup and not SERVER:  # Do not clean up on the server as csv is needed in parallel jobs
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')

    if empty:
        return ig.Graph(directed=False)

    if weighting:
        # Process the edges to modify length based on speed limits
        e['maxspeed'] = e['maxspeed'].str.replace(' mph', '', regex=False).astype(float)
        e['maxspeed'].fillna(20, inplace=True)  # Assign default speed of 20 where NaN
        e['ori_length'] = e['length']  # Store original length only if weighting is True
        e['length'] = e['length'] * e['maxspeed']  # Modify the length based on speed

    G = osm_to_ig(n, e, weighting)  # Pass weighting to osm_to_ig
    round_coordinates(G)
    mirror_y(G)
    return G



def ig_to_geojson(G):
    linestring_list = []
    for e in G.es():
        linestring_list.append(geojson.LineString([(e.source_vertex["x"], -e.source_vertex["y"]), (e.target_vertex["x"], -e.target_vertex["y"])]))
    G_geojson = geojson.GeometryCollection(linestring_list)
    return G_geojson




# NETWORK GENERATION

def highest_closeness_node(G):
    closeness_values = G.closeness(weights = 'weight')
    sorted_closeness = sorted(closeness_values, reverse = True)
    index = closeness_values.index(sorted_closeness[0])
    return G.vs(index)['id']

def clusterindices_by_length(clusterinfo, rev = True):
    return [k for k, v in sorted(clusterinfo.items(), key=lambda item: item[1]["length"], reverse = rev)]

class MyPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

def segments_intersect(A,B,C,D):
    """Check if two line segments intersect (except for colinearity)
    Returns true if line segments AB and CD intersect properly.
    Adapted from: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    """
    if (A.x == C.x and A.y == C.y) or (A.x == D.x and A.y == D.y) or (B.x == C.x and B.y == C.y) or (B.x == D.x and B.y == D.y): return False # If the segments share an endpoint they do not intersect properly
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def new_edge_intersects(G, enew):
    """Given a graph G and a potential new edge enew,
    check if enew will intersect any old edge.
    """
    E1 = MyPoint(enew[0], enew[1])
    E2 = MyPoint(enew[2], enew[3])
    for e in G.es():
        O1 = MyPoint(e.source_vertex["x"], e.source_vertex["y"])
        O2 = MyPoint(e.target_vertex["x"], e.target_vertex["y"])
        if segments_intersect(E1, E2, O1, O2):
            return True
    return False
    

def delete_overlaps(G_res, G_orig, verbose = False):
    """Deletes inplace all overlaps of G_res with G_orig (from G_res)
    based on node ids. In other words: G_res -= G_orig
    """
    del_edges = []
    for e in list(G_res.es):
        try:
            n1_id = e.source_vertex["id"]
            n2_id = e.target_vertex["id"]
            # If there is already an edge in the original network, delete it
            n1_index = G_orig.vs.find(id = n1_id).index
            n2_index = G_orig.vs.find(id = n2_id).index
            if G_orig.are_connected(n1_index, n2_index):
                del_edges.append(e.index)
        except:
            pass
    G_res.delete_edges(del_edges)
    # Remove isolated nodes
    isolated_nodes = G_res.vs.select(_degree_eq=0)
    G_res.delete_vertices(isolated_nodes)
    if verbose: print("Removed " + str(len(del_edges)) + " overlapping edges and " + str(len(isolated_nodes)) + " nodes.")

def constrict_overlaps(G_res, G_orig, factor = 5):
    """Increases length by factor of all overlaps of G_res with G_orig (in G_res) based on edge ids.
    """
    for e in list(G_res.es):
        try:
            n1_id = e.source_vertex["id"]
            n2_id = e.target_vertex["id"]
            n1_index = G_orig.vs.find(id = n1_id).index
            n2_index = G_orig.vs.find(id = n2_id).index
            if G_orig.are_connected(n1_index, n2_index):
                G_res.es[e.index]["weight"] = factor * G_res.es[e.index]["weight"]
        except:
            pass



    

def greedy_triangulation_routing_clusters(G, G_total, clusters, clusterinfo, prune_quantiles = [1], prune_measure = "betweenness", verbose = False, full_run = False):
    """Greedy Triangulation (GT) of a bike network G's clusters,
    then routing on the graph G_total that includes car infra to connect the GT.
    G and G_total are ipgraph graphs
    
    The GT connects pairs of clusters in ascending order of their distance provided
    that no edge crossing is introduced. It leads to a maximal connected planar
    graph, while minimizing the total length of edges considered. 
    See: cardillo2006spp
    
    Distance here is routing distance, while edge crossing is checked on an abstract 
    level.
    """
    
    if len(clusters) < 2: return ([], []) # We can't do anything with less than 2 clusters

    centroid_indices = [v["centroid_index"] for k, v in sorted(clusterinfo.items(), key=lambda item: item[1]["size"], reverse = True)]
    G_temp = copy.deepcopy(G_total)
    for e in G_temp.es: # delete all edges
        G_temp.es.delete(e)
    
    clusterpairs = clusterpairs_by_distance(G, G_total, clusters, clusterinfo, True, verbose, full_run)
    if len(clusterpairs) == 0: return ([], [])
    
    centroidpairs = [((clusterinfo[c[0][0]]['centroid_id'], clusterinfo[c[0][1]]['centroid_id']), c[2]) for c in clusterpairs]
    
    GT_abstracts = []
    GTs = []
    for prune_quantile in prune_quantiles:
        GT_abstract = copy.deepcopy(G_temp.subgraph(centroid_indices))
        GT_abstract = greedy_triangulation(GT_abstract, centroidpairs, prune_quantile, prune_measure)
        GT_abstracts.append(GT_abstract)

        centroidids_closestnodeids = {} # dict for retrieveing quickly closest node ids pairs from centroidid pairs
        for x in clusterpairs:
            centroidids_closestnodeids[(clusterinfo[x[0][0]]["centroid_id"], clusterinfo[x[0][1]]["centroid_id"])] = (x[1][0], x[1][1])
            centroidids_closestnodeids[(clusterinfo[x[0][1]]["centroid_id"], clusterinfo[x[0][0]]["centroid_id"])] = (x[1][1], x[1][0]) # also add switched version as we do not care about order

        # Get node pairs we need to route, sorted by distance
        routenodepairs = []
        for e in GT_abstract.es:
            # get the centroid-ids from closestnode-ids
            routenodepairs.append([centroidids_closestnodeids[(e.source_vertex["id"], e.target_vertex["id"])], e["weight"]])

        routenodepairs.sort(key=lambda x: x[1])

        # Do the routing, on G_total
        GT_indices = set()
        for poipair, poipair_distance in routenodepairs:
            poipair_ind = (G_total.vs.find(id = poipair[0]).index, G_total.vs.find(id = poipair[1]).index)
            sp = set(G_total.get_shortest_paths(poipair_ind[0], poipair_ind[1], weights = "weight", output = "vpath")[0])
            GT_indices = GT_indices.union(sp)

        GT = G_total.induced_subgraph(GT_indices)
        GTs.append(GT)
    
    return(GTs, GT_abstracts)


def clusterpairs_by_distance(G, G_total, clusters, clusterinfo, return_distances = False, verbose = False, full_run = False):
    """Calculates the (weighted) graph distances on G for a number of clusters.
    Returns all pairs of cluster ids and closest nodes in ascending order of their distance. 
    If return_distances, then distances are also returned.

    Returns a list containing these elements, sorted by distance:
    [(clusterid1, clusterid2), (closestnodeid1, closestnodeid2), distance]
    """
    
    cluster_indices = clusterindices_by_length(clusterinfo, False) # Start with the smallest so the for loop is as short as possible
    clusterpairs = []
    clustercopies = {}
    
    # Create copies of all clusters
    for i in range(len(cluster_indices)):
        clustercopies[i] = clusters[i].copy()
        
    # Take one cluster
    for i, c1 in enumerate(cluster_indices[:-1]):
        c1_indices = G_total.vs.select(lambda x: x["id"] in clustercopies[c1].vs()["id"]).indices
        print("Working on cluster " + str(i+1) + " of " + str(len(cluster_indices)) + "...")
        for j, c2 in enumerate(cluster_indices[i+1:]):
            closest_pair = {'i': -1, 'j': -1}
            min_dist = np.inf
            c2_indices = G_total.vs.select(lambda x: x["id"] in clustercopies[c2].vs()["id"]).indices
            if verbose: print("... routing " + str(len(c1_indices)) + " nodes to " + str(len(c2_indices)) + " nodes in other cluster " + str(j+1) + " of " + str(len(cluster_indices[i+1:])) + ".")
            
            if full_run:
                # Compare all pairs of nodes in both clusters (takes long)
                for a in list(c1_indices):
                    sp = G_total.get_shortest_paths(a, c2_indices, weights = "weight", output = "epath")

                    if all([not elem for elem in sp]):
                        # If there is no path from one node, there is no path from any node
                        break
                    else:
                        for path, c2_index in zip(sp, c2_indices):
                            if len(path) >= 1:
                                dist_nodes = sum([G_total.es[e]['weight'] for e in path])
                                if dist_nodes < min_dist:
                                    closest_pair['i'] = G_total.vs[a]["id"]
                                    closest_pair['j'] = G_total.vs[c2_index]["id"]
                                    min_dist = dist_nodes
            else:
                # Do a heuristic that should be close enough.
                # From cluster 1, look at all shortest paths only from its centroid
                a = clusterinfo[c1]["centroid_index"]
                sp = G_total.get_shortest_paths(a, c2_indices, weights = "weight", output = "epath")
                if all([not elem for elem in sp]):
                    # If there is no path from one node, there is no path from any node
                    break
                else:
                    for path, c2_index in zip(sp, c2_indices):
                        if len(path) >= 1:
                            dist_nodes = sum([G_total.es[e]['weight'] for e in path])
                            if dist_nodes < min_dist:
                                closest_pair['j'] = G_total.vs[c2_index]["id"]
                                min_dist = dist_nodes
                # Closest c2 node to centroid1 found. Now find all c1 nodes to that closest c2 node.
                b = G_total.vs.find(id = closest_pair['j']).index
                sp = G_total.get_shortest_paths(b, c1_indices, weights = "weight", output = "epath")
                if all([not elem for elem in sp]):
                    # If there is no path from one node, there is no path from any node
                    break
                else:
                    for path, c1_index in zip(sp, c1_indices):
                        if len(path) >= 1:
                            dist_nodes = sum([G_total.es[e]['weight'] for e in path])
                            if dist_nodes <= min_dist: # <=, not <!
                                closest_pair['i'] = G_total.vs[c1_index]["id"]
                                min_dist = dist_nodes
            
            if closest_pair['i'] != -1 and closest_pair['j'] != -1:
                clusterpairs.append([(c1, c2), (closest_pair['i'], closest_pair['j']), min_dist])
                                    
    clusterpairs.sort(key = lambda x: x[-1])
    if return_distances:
        return clusterpairs
    else:
        return [[o[0], o[1]] for o in clusterpairs]


def mst_routing(G, pois, weighting=None):
    """Minimum Spanning Tree (MST) of a graph G's node subset pois,
    then routing to connect the MST.
    G is an ipgraph graph, pois is a list of node ids.
    
    The MST is the planar graph with the minimum number of (weighted) 
    links in order to assure connectedness.

    Distance here is routing distance, while edge crossing is checked on an abstract 
    level.
    """

    if len(pois) < 2: return (ig.Graph(), ig.Graph()) # We can't do anything with less than 2 POIs

    # MST_abstract is the MST with same nodes but euclidian links
    pois_indices = set()
    for poi in pois:
        pois_indices.add(G.vs.find(id = poi).index)
    G_temp = copy.deepcopy(G)
    for e in G_temp.es: # delete all edges
        G_temp.es.delete(e)
        
    poipairs = poipairs_by_distance(G, pois, weighting, True)
    if len(poipairs) == 0: return (ig.Graph(), ig.Graph())

    MST_abstract = copy.deepcopy(G_temp.subgraph(pois_indices))
    for poipair, poipair_distance in poipairs:
        poipair_ind = (MST_abstract.vs.find(id = poipair[0]).index, MST_abstract.vs.find(id = poipair[1]).index)
        MST_abstract.add_edge(poipair_ind[0], poipair_ind[1] , weight = poipair_distance)
    MST_abstract = MST_abstract.spanning_tree(weights = "weight")

    # Get node pairs we need to route, sorted by distance
    routenodepairs = {}
    for e in MST_abstract.es:
        routenodepairs[(e.source_vertex["id"], e.target_vertex["id"])] = e["weight"]
    routenodepairs = sorted(routenodepairs.items(), key = lambda x: x[1])

    # Do the routing
    MST_indices = set()
    for poipair, poipair_distance in routenodepairs:
        poipair_ind = (G.vs.find(id = poipair[0]).index, G.vs.find(id = poipair[1]).index)
        sp = set(G.get_shortest_paths(poipair_ind[0], poipair_ind[1], weights = "weight", output = "vpath")[0])
        MST_indices = MST_indices.union(sp)

    MST = G.induced_subgraph(MST_indices)
    
    return (MST, MST_abstract)



def greedy_triangulation(GT, poipairs, prune_quantile = 1, prune_measure = "betweenness", edgeorder = False):
    """Greedy Triangulation (GT) of a graph GT with an empty edge set.
    Distances between pairs of nodes are given by poipairs.
    
    The GT connects pairs of nodes in ascending order of their distance provided
    that no edge crossing is introduced. It leads to a maximal connected planar
    graph, while minimizing the total length of edges considered. 
    See: cardillo2006spp
    """
    
    for poipair, poipair_distance in poipairs:
        poipair_ind = (GT.vs.find(id = poipair[0]).index, GT.vs.find(id = poipair[1]).index)
        if not new_edge_intersects(GT, (GT.vs[poipair_ind[0]]["x"], GT.vs[poipair_ind[0]]["y"], GT.vs[poipair_ind[1]]["x"], GT.vs[poipair_ind[1]]["y"])):
            GT.add_edge(poipair_ind[0], poipair_ind[1], weight = poipair_distance)
            
    # Get the measure for pruning
    if prune_measure == "betweenness":
        BW = GT.edge_betweenness(directed = False, weights = "weight")
        qt = np.quantile(BW, 1-prune_quantile)
        sub_edges = []
        for c, e in enumerate(GT.es):
            if BW[c] >= qt: 
                sub_edges.append(c)
            GT.es[c]["bw"] = BW[c]
            GT.es[c]["width"] = math.sqrt(BW[c]+1)*0.5
        # Prune
        GT = GT.subgraph_edges(sub_edges)
    elif prune_measure == "closeness":
        CC = GT.closeness(vertices = None, weights = "weight")
        qt = np.quantile(CC, 1-prune_quantile)
        sub_nodes = []
        for c, v in enumerate(GT.vs):
            if CC[c] >= qt: 
                sub_nodes.append(c)
            GT.vs[c]["cc"] = CC[c]
        GT = GT.induced_subgraph(sub_nodes)
    elif prune_measure == "random":
        ind = np.quantile(np.arange(len(edgeorder)), prune_quantile, interpolation = "lower") + 1 # "lower" and + 1 so smallest quantile has at least one edge
        GT = GT.subgraph_edges(edgeorder[:ind])
    
    return GT


def restore_original_lengths(G):
    """Restore original lengths from the 'ori_length' attribute."""
    for e in G.es:
        e["weight"] = e["ori_length"]
 


def greedy_triangulation_routing(G, pois, weighting=None, prune_quantiles = [1], prune_measure = "betweenness"):
    """Greedy Triangulation (GT) of a graph G's node subset pois,
    then routing to connect the GT (up to a quantile of betweenness
    betweenness_quantile).
    G is an ipgraph graph, pois is a list of node ids.
    
    The GT connects pairs of nodes in ascending order of their distance provided
    that no edge crossing is introduced. It leads to a maximal connected planar
    graph, while minimizing the total length of edges considered. 
    See: cardillo2006spp
    
    Distance here is routing distance, while edge crossing is checked on an abstract 
    level.
    """
    
    if len(pois) < 2: return ([], []) # We can't do anything with less than 2 POIs

    # GT_abstract is the GT with same nodes but euclidian links to keep track of edge crossings
    pois_indices = set()
    for poi in pois:
        pois_indices.add(G.vs.find(id = poi).index)
    G_temp = copy.deepcopy(G)
    for e in G_temp.es: # delete all edges
        G_temp.es.delete(e)
        
    poipairs = poipairs_by_distance(G, pois, weighting, True)
    if len(poipairs) == 0: return ([], [])

    if prune_measure == "random":
        # run the whole GT first
        GT = copy.deepcopy(G_temp.subgraph(pois_indices))
        for poipair, poipair_distance in poipairs:
            poipair_ind = (GT.vs.find(id = poipair[0]).index, GT.vs.find(id = poipair[1]).index)
            if not new_edge_intersects(GT, (GT.vs[poipair_ind[0]]["x"], GT.vs[poipair_ind[0]]["y"], GT.vs[poipair_ind[1]]["x"], GT.vs[poipair_ind[1]]["y"])):
                GT.add_edge(poipair_ind[0], poipair_ind[1], weight = poipair_distance)
        # create a random order for the edges
        random.seed(0) # const seed for reproducibility
        edgeorder = random.sample(range(GT.ecount()), k = GT.ecount())
    else: 
        edgeorder = False
    
    GT_abstracts = []
    GTs = []
    for prune_quantile in tqdm(prune_quantiles, desc = "Greedy triangulation", leave = False):
        GT_abstract = copy.deepcopy(G_temp.subgraph(pois_indices))
        GT_abstract = greedy_triangulation(GT_abstract, poipairs, prune_quantile, prune_measure, edgeorder)
        GT_abstracts.append(GT_abstract)
        
        # Get node pairs we need to route, sorted by distance
        routenodepairs = {}
        for e in GT_abstract.es:
            routenodepairs[(e.source_vertex["id"], e.target_vertex["id"])] = e["weight"]
        routenodepairs = sorted(routenodepairs.items(), key = lambda x: x[1])

        # Do the routing
        GT_indices = set()
        for poipair, poipair_distance in routenodepairs:
            poipair_ind = (G.vs.find(id = poipair[0]).index, G.vs.find(id = poipair[1]).index)
            # debug
            #print(f"Edge weights before routing: {G.es['weight'][:10]}")  # Prints first 10 weights
            #print(f"Routing between: {poipair[0]} and {poipair[1]} with distance: {poipair_distance}")
            sp = set(G.get_shortest_paths(poipair_ind[0], poipair_ind[1], weights = "weight", output = "vpath")[0])
            #print(f"Shortest path between {poipair[0]} and {poipair[1]}: {sp}")

            GT_indices = GT_indices.union(sp)

        GT = G.induced_subgraph(GT_indices)
        GTs.append(GT)
    
    return (GTs, GT_abstracts)
    
    
def poipairs_by_distance(G, pois, weighting=None, return_distances = False):
    """Calculates the (weighted) graph distances on G for a subset of nodes pois.
    Returns all pairs of poi ids in ascending order of their distance. 
    If return_distances, then distances are also returned.
    If we are using a weighted graph, we need to calculate the distances using orignal
    edge lengths rather than adjusted weighted lengths.
    """
    
    # Get poi indices
    indices = []
    for poi in pois:
        indices.append(G_carall.vs.find(id = poi).index)
    
    # Get sequences of nodes and edges in shortest paths between all pairs of pois
    poi_nodes = []
    poi_edges = []
    for c, v in enumerate(indices):
        poi_nodes.append(G.get_shortest_paths(v, indices[c:], weights = "weight", output = "vpath"))
        poi_edges.append(G.get_shortest_paths(v, indices[c:], weights = "weight", output = "epath"))

    # Sum up weights (distances) of all paths
    poi_dist = {}
    for paths_n, paths_e in zip(poi_nodes, poi_edges):
        for path_n, path_e in zip(paths_n, paths_e):
            # Sum up distances of path segments from first to last node
            if weighting:
                # Use the 'weight' for finding the shortest path
                path_dist = sum([G.es[e]['ori_length'] for e in path_e])  # Use 'ori_length' for distance
            else:
                path_dist = sum([G.es[e]['weight'] for e in path_e])  # Fallback to 'weight' if weighting is False
            
            if path_dist > 0:
                poi_dist[(path_n[0], path_n[-1])] = path_dist
            
    temp = sorted(poi_dist.items(), key = lambda x: x[1])
    # Back to ids
    output = []
    for p in temp:
        output.append([(G.vs[p[0][0]]["id"], G.vs[p[0][1]]["id"]), p[1]])
    
    if return_distances:
        return output
    else:
        return [o[0] for o in output]





# ANALYSIS

def rotate_grid(p, origin = (0, 0), degrees = 0):
        """Rotate a list of points around an origin (in 2D). 
        
        Parameters:
            p (tuple or list of tuples): (x,y) coordinates of points to rotate
            origin (tuple): (x,y) coordinates of rotation origin
            degrees (int or float): degree (clockwise)

        Returns:
            ndarray: the rotated points, as an ndarray of 1x2 ndarrays
        """
        # https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
        angle = np.deg2rad(-degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)


# Two functions from: https://github.com/gboeing/osmnx-examples/blob/v0.11/notebooks/17-street-network-orientations.ipynb
def reverse_bearing(x):
    return x + 180 if x < 180 else x - 180

def count_and_merge(n, bearings):
    # make twice as many bins as desired, then merge them in pairs
    # prevents bin-edge effects around common values like 0° and 90°
    n = n * 2
    bins = np.arange(n + 1) * 360 / n
    count, _ = np.histogram(bearings, bins=bins)
    
    # move the last bin to the front, so eg 0.01° and 359.99° will be binned together
    count = np.roll(count, 1)
    return count[::2] + count[1::2]


def calculate_directness(G, numnodepairs = 500):
    """Calculate directness on G over all connected node pairs in indices. This calculation method divides the total sum of euclidian distances by total sum of network distances.
    """
    
    indices = random.sample(list(G.vs), min(numnodepairs, len(G.vs)))

    poi_edges = []
    total_distance_direct = 0
    for c, v in enumerate(indices):
        poi_edges.append(G.get_shortest_paths(v, indices[c:], weights = "weight", output = "epath"))
        temp = G.get_shortest_paths(v, indices[c:], weights = "weight", output = "vpath")
        try:
            total_distance_direct += sum(dist_vector([(G.vs[t[0]]["y"], G.vs[t[0]]["x"]) for t in temp], [(G.vs[t[-1]]["y"], G.vs[t[-1]]["x"]) for t in temp])) # must be in format lat,lon = y, x
        except: # Rarely, routing does not work. Unclear why.
            pass
    total_distance_network = 0
    for paths_e in poi_edges:
        for path_e in paths_e:
            # Sum up distances of path segments from first to last node
            total_distance_network += sum([G.es[e]['weight'] for e in path_e])
    
    return total_distance_direct / total_distance_network

def calculate_directness_linkwise(G, numnodepairs = 500):
    """Calculate directness on G over all connected node pairs in indices. This is maybe the common calculation method: It takes the average of linkwise euclidian distances divided by network distances.

        If G has multiple components, node pairs in different components are discarded.
    """

    indices = random.sample(list(G.vs), min(numnodepairs, len(G.vs)))

    directness_links = np.zeros(int((len(indices)*(len(indices)-1))/2))
    ind = 0
    for c, v in enumerate(indices):
        poi_edges = G.get_shortest_paths(v, indices[c:], weights = "weight", output = "epath")
        for c_delta, path_e in enumerate(poi_edges[1:]): # Discard first empty list because it is the node to itself
            if path_e: # if path is non-empty, meaning the node pair is in the same component
                distance_network = sum([G.es[e]['weight'] for e in path_e]) # sum over all edges of path
                distance_direct = dist(v, indices[c+c_delta+1]) # dist first to last node, must be in format lat,lon = y, x

                directness_links[ind] = distance_direct / distance_network
                ind += 1
    directness_links = directness_links[:ind] # discard disconnected node pairs

    return np.mean(directness_links)


def listmean(lst): 
    try: return sum(lst) / len(lst)
    except: return 0

def calculate_coverage_edges(G, buffer_m = 500, return_cov = False, G_prev = ig.Graph(), cov_prev = Polygon()):
    """Calculates the area and shape covered by the graph's edges.
    If G_prev and cov_prev are given, only the difference between G and G_prev are calculated, then added to cov_prev.
    """

    G_added = copy.deepcopy(G)
    delete_overlaps(G_added, G_prev)

    # https://gis.stackexchange.com/questions/121256/creating-a-circle-with-radius-in-metres
    loncenter = listmean([v["x"] for v in G.vs])
    latcenter = listmean([v["y"] for v in G.vs])
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    aeqd_to_wgs84 = pyproj.Transformer.from_proj(
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"))
    edgetuples = [((e.source_vertex["x"], e.source_vertex["y"]), (e.target_vertex["x"], e.target_vertex["y"])) for e in G_added.es]
    # Shapely buffer seems slow for complex objects: https://stackoverflow.com/questions/57753813/speed-up-shapely-buffer
    # Therefore we buffer piecewise.
    cov_added = Polygon()
    for c, t in enumerate(edgetuples):
        # if cov.geom_type == 'MultiPolygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), sum([len(pol.exterior.coords) for pol in cov]))
        # elif cov.geom_type == 'Polygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), len(pol.exterior.coords))
        buf = ops.transform(aeqd_to_wgs84.transform, ops.transform(wgs84_to_aeqd.transform, LineString(t)).buffer(buffer_m))
        cov_added = ops.unary_union([cov_added, Polygon(buf)])

    # Merge with cov_prev
    if not cov_added.is_empty: # We need this check because apparently an empty Polygon adds an area.
        cov = ops.unary_union([cov_added, cov_prev])
    else:
        cov = cov_prev

    cov_transformed = ops.transform(wgs84_to_aeqd.transform, cov)
    covered_area = cov_transformed.area / 1000000 # turn from m2 to km2

    if return_cov:
        return (covered_area, cov)
    else:
        return covered_area


def calculate_poiscovered(G, cov, nnids):
    """Calculates how many nodes, given by nnids, are covered by the shapely (multi)polygon cov
    """
    
    pois_indices = set()
    for poi in nnids:
        pois_indices.add(G.vs.find(id = poi).index)

    poiscovered = 0
    for poi in pois_indices:
        v = G.vs[poi]
        if Point(v["x"], v["y"]).within(cov):
            poiscovered += 1
    
    return poiscovered


def calculate_efficiency_global(G, numnodepairs = 500, normalized = True, debug=False):
    """Calculates global network efficiency.
    If there are more than numnodepairs nodes, measure over pairings of a 
    random sample of numnodepairs nodes.
    """
    if "x" not in G.vs.attributes() or "y" not in G.vs.attributes():
        raise KeyError("Graph vertices are missing 'x' or 'y' attributes.")


    if G is None: return 0
    if G.vcount() > numnodepairs:
        nodeindices = random.sample(list(G.vs.indices), numnodepairs)
    else:
        nodeindices = list(G.vs.indices)
    d_ij = G.shortest_paths(source = nodeindices, target = nodeindices, weights = "weight")
    d_ij = [item for sublist in d_ij for item in sublist] # flatten

    ### Check if d_ij contains valid distances
    if not d_ij: return 0  # No distances available
    ###

    

    EG = sum([1/d for d in d_ij if d != 0])

    if debug:
        print("EG: ", EG)
    if not normalized: return EG
    pairs = list(itertools.permutations(nodeindices, 2))
    if len(pairs) < 1: return 0
    l_ij = dist_vector([(G.vs[p[0]]["y"], G.vs[p[0]]["x"]) for p in pairs],
                            [(G.vs[p[1]]["y"], G.vs[p[1]]["x"]) for p in pairs]) # must be in format lat,lon = y,x
    EG_id = sum([1/l for l in l_ij if l != 0])

    if debug:
        print("EG_id", EG_id)
    # re comment this block later
    #if (EG / EG_id) > 1: # This should not be allowed to happen!
    #    pp.pprint(d_ij)
    #    pp.pprint(l_ij)
    #    pp.pprint([e for e in G.es])
    #    print(pairs)
    #    print([(G.vs[p[0]]["y"], G.vs[p[0]]["x"]) for p in pairs],
    #                         [(G.vs[p[1]]["y"], G.vs[p[1]]["x"]) for p in pairs]) # must be in format lat,lon = y,x
    #    print(EG, EG_id)
    #   sys.exit()
    # assert EG / EG_id <= 1, "Normalized EG > 1. This should not be possible."


    if EG_id == 0:
        print("EG_id is zero. Returning default efficiency value.")
        return 0  # Or another appropriate default value
    return EG / EG_id



def calculate_efficiency_local(G, numnodepairs = 500, normalized = True):
    """Calculates local network efficiency.
    If there are more than numnodepairs nodes, measure over pairings of a 
    random sample of numnodepairs nodes.
    """

    if G is None: return 0
    if G.vcount() > numnodepairs:
        nodeindices = random.sample(list(G.vs.indices), numnodepairs)
    else:
        nodeindices = list(G.vs.indices)
    EGi = []
    vcounts = []
    ecounts = []
    for i in nodeindices:
        if len(G.neighbors(i)) > 1: # If we have a nontrivial neighborhood
            G_induced = G.induced_subgraph(G.neighbors(i))
            EGi.append(calculate_efficiency_global(G_induced, numnodepairs, normalized))
    return listmean(EGi)

def calculate_metrics(
    G, GT_abstract, G_big, nnids, calcmetrics={"length": 0, "length_lcc": 0, "coverage": 0, "directness": 0,
                                               "directness_lcc": 0, "poi_coverage": 0, "components": 0,
                                               "overlap_biketrack": 0, "overlap_bikeable": 0, "efficiency_global": 0,
                                               "efficiency_local": 0, "directness_lcc_linkwise": 0,
                                               "directness_all_linkwise": 0, "overlap_neighbourhood": 0},
    buffer_walk=500, numnodepairs=500, verbose=False, return_cov=True, G_prev=ig.Graph(),
    cov_prev=Polygon(), ignore_GT_abstract=False, Gexisting={}, Gneighbourhoods=None):
    """Calculates all metrics (using the keys from calcmetrics)."""

    output = {key: 0 for key in calcmetrics}
    cov = Polygon()

    # Check that the graph has links (sometimes we have an isolated node)
    if G.ecount() > 0 and GT_abstract.ecount() > 0:
        # Get LCC
        cl = G.clusters()
        LCC = cl.giant()

        # EFFICIENCY 
        if not ignore_GT_abstract:
            if verbose and ("efficiency_global" in calcmetrics or "efficiency_local" in calcmetrics): print("Calculating efficiency...")
            if "efficiency_global" in calcmetrics:
                output["efficiency_global"] = calculate_efficiency_global(GT_abstract, numnodepairs)
            if "efficiency_local" in calcmetrics:
                output["efficiency_local"] = calculate_efficiency_local(GT_abstract, numnodepairs) 
        
        # EFFICIENCY ROUTED
        if verbose and ("efficiency_global_routed" in calcmetrics or "efficiency_local_routed" in calcmetrics): print("Calculating efficiency (routed)...")
        if "efficiency_global_routed" in calcmetrics:
            try:
                output["efficiency_global_routed"] = calculate_efficiency_global(simplify_ig(G), numnodepairs)
            except:
                print("Problem with efficiency_global_routed.") 
        if "efficiency_local_routed" in calcmetrics:
            try:
                output["efficiency_local_routed"] = calculate_efficiency_local(simplify_ig(G), numnodepairs)
            except:
                print("Problem with efficiency_local_routed.")

        # LENGTH
        if verbose and ("length" in calcmetrics or "length_lcc" in calcmetrics): print("Calculating length...")
        if "length" in calcmetrics:
            output["length"] = sum([e['weight'] for e in G.es])
        if "length_lcc" in calcmetrics:
            if len(cl) > 1:
                output["length_lcc"] = sum([e['weight'] for e in LCC.es])
            else:
                output["length_lcc"] = output["length"]
        
        # COVERAGE
        if "coverage" in calcmetrics:
            if verbose: print("Calculating coverage...")
            covered_area, cov = calculate_coverage_edges(G, buffer_walk, return_cov, G_prev, cov_prev)
            output["coverage"] = covered_area

            # OVERLAP WITH EXISTING NETS
            if Gexisting:
                if "overlap_biketrack" in calcmetrics:
                    try:
                        output["overlap_biketrack"] = edge_lengths(intersect_igraphs(Gexisting["biketrack"], G))
                    except:  # If there is not bike infrastructure, set to zero
                        output["overlap_biketrack"] = 0
                if "overlap_bikeable" in calcmetrics:
                    try:
                        output["overlap_bikeable"] = edge_lengths(intersect_igraphs(Gexisting["bikeable"], G))
                    except:  # If there is not bikeable infrastructure, set to zero
                        output["overlap_bikeable"] = 0

        # OVERLAP WITH NEIGHBOURHOOD NETWORK
        if Gneighbourhoods and "overlap_neighbourhood" in calcmetrics:
            if verbose: print("Calculating overlap_neighbourhood...")
            try:
                output["overlap_neighbourhood"] = edge_lengths(intersect_igraphs(Gneighbourhoods, G))
            except:  # If there are issues with intersecting graphs, set to zero
                output["overlap_neighbourhood"] = 0

        # POI COVERAGE
        if "poi_coverage" in calcmetrics:
            if verbose: print("Calculating POI coverage...")
            output["poi_coverage"] = calculate_poiscovered(G_big, cov, nnids)

        # COMPONENTS
        if "components" in calcmetrics:
            if verbose: print("Calculating components...")
            output["components"] = len(list(G.components()))
        
        # DIRECTNESS
        if verbose and ("directness" in calcmetrics or "directness_lcc" in calcmetrics): print("Calculating directness...")
        if "directness" in calcmetrics:
            output["directness"] = calculate_directness(G, numnodepairs)
        if "directness_lcc" in calcmetrics:
            if len(cl) > 1:
                output["directness_lcc"] = calculate_directness(LCC, numnodepairs)
            else:
                output["directness_lcc"] = output["directness"]

        # DIRECTNESS LINKWISE
        if verbose and ("directness_lcc_linkwise" in calcmetrics): print("Calculating directness linkwise...")
        if "directness_lcc_linkwise" in calcmetrics:
            if len(cl) > 1:
                output["directness_lcc_linkwise"] = calculate_directness_linkwise(LCC, numnodepairs)
            else:
                output["directness_lcc_linkwise"] = calculate_directness_linkwise(G, numnodepairs)
        if verbose and ("directness_all_linkwise" in calcmetrics): print("Calculating directness linkwise (all components)...")
        if "directness_all_linkwise" in calcmetrics:
            if "directness_lcc_linkwise" in calcmetrics and len(cl) <= 1:
                output["directness_all_linkwise"] = output["directness_lcc_linkwise"]
            else:  # we have >1 components
                output["directness_all_linkwise"] = calculate_directness_linkwise(G, numnodepairs)

    if return_cov: 
        return output, cov
    else:
        return output



def overlap_linepoly(l, p):
    """Calculates the length of shapely LineString l falling inside the shapely Polygon p
    """
    return p.intersection(l).length if l.length else 0


def edge_lengths(G):
    """Returns the total length of edges in an igraph graph.
    """
    return sum([e['weight'] for e in G.es])


def intersect_igraphs(G1, G2):
    """Generates the graph intersection of igraph graphs G1 and G2, copying also link and node attributes.
    """
    # Ginter = G1.__and__(G2) # This does not work with attributes.
    if G1.ecount() > G2.ecount(): # Iterate through edges of the smaller graph
        G1, G2 = G2, G1
    inter_nodes = set()
    inter_edges = []
    inter_edge_attributes = {}
    inter_node_attributes = {}
    edge_attribute_name_list = G2.edge_attributes()
    node_attribute_name_list = G2.vertex_attributes()
    for edge_attribute_name in edge_attribute_name_list:
        inter_edge_attributes[edge_attribute_name] = []
    for node_attribute_name in node_attribute_name_list:
        inter_node_attributes[node_attribute_name] = []
    for e in list(G1.es):
        n1_id = e.source_vertex["id"]
        n2_id = e.target_vertex["id"]
        try:
            n1_index = G2.vs.find(id = n1_id).index
            n2_index = G2.vs.find(id = n2_id).index
        except ValueError:
            continue
        if G2.are_connected(n1_index, n2_index):
            inter_edges.append((n1_index, n2_index))
            inter_nodes.add(n1_index)
            inter_nodes.add(n2_index)
            edge_attributes = e.attributes()
            for edge_attribute_name in edge_attribute_name_list:
                inter_edge_attributes[edge_attribute_name].append(edge_attributes[edge_attribute_name])

    # map nodeids to first len(inter_nodes) integers
    idmap = {n_index:i for n_index,i in zip(inter_nodes, range(len(inter_nodes)))}

    G_inter = ig.Graph()
    G_inter.add_vertices(len(inter_nodes))
    G_inter.add_edges([(idmap[e[0]], idmap[e[1]]) for e in inter_edges])
    for edge_attribute_name in edge_attribute_name_list:
        G_inter.es[edge_attribute_name] = inter_edge_attributes[edge_attribute_name]

    for n_index in idmap.keys():
        v = G2.vs[n_index]
        node_attributes = v.attributes()
        for node_attribute_name in node_attribute_name_list:
            inter_node_attributes[node_attribute_name].append(node_attributes[node_attribute_name])
    for node_attribute_name in node_attribute_name_list:
        G_inter.vs[node_attribute_name] = inter_node_attributes[node_attribute_name]

    return G_inter


def calculate_metrics_additively(
    Gs, GT_abstracts, prune_quantiles, G_big, nnids, buffer_walk=500, numnodepairs=500, verbose=False, 
    return_cov=True, Gexisting={}, Gneighbourhoods=None,
    output={
        "length": [], "length_lcc": [], "coverage": [], "directness": [], "directness_lcc": [],
        "poi_coverage": [], "components": [], "overlap_biketrack": [], "overlap_bikeable": [],
        "efficiency_global": [], "efficiency_local": [], "efficiency_global_routed": [], 
        "efficiency_local_routed": [], "directness_lcc_linkwise": [], "directness_all_linkwise": [],
        "overlap_neighbourhood": []  # Add the new metric here
    }
):
    """Calculates all metrics, additively. 
    Coverage differences are calculated in every step instead of the whole coverage.
    """

    # BICYCLE NETWORKS
    covs = {}  # Covers using buffer_walk
    cov_prev = Polygon()
    GT_prev = ig.Graph()

    for GT, GT_abstract, prune_quantile in zip(Gs, GT_abstracts, tqdm(prune_quantiles, desc="Bicycle networks", leave=False)):
        if verbose: print("Calculating bike network metrics for quantile " + str(prune_quantile))
        metrics, cov = calculate_metrics(
            GT, GT_abstract, G_big, nnids, output, buffer_walk, numnodepairs, verbose, 
            return_cov, GT_prev, cov_prev, False, Gexisting, Gneighbourhoods
        )

        for key in output.keys():
            output[key].append(metrics[key])
        covs[prune_quantile] = cov
        cov_prev = copy.deepcopy(cov)
        GT_prev = copy.deepcopy(GT)

    return output, covs


def generate_video(placeid, imgname, vformat = "webm", duplicatelastframe = 5, verbose = True):
    """Generate a video from a set of images using OpenCV
    """
    # Code adapted from: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python#44948030
    
    images = [img for img in os.listdir(PATH["plots_networks"] + placeid + "/") if img.startswith(placeid + imgname)]
    images.sort()
    frame = cv2.imread(os.path.join(PATH["plots_networks"] + placeid + "/", images[0]))
    height, width, layers = frame.shape

    if vformat == "webm":
        # https://stackoverflow.com/questions/49530857/python-opencv-video-format-play-in-browser
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        video = cv2.VideoWriter(PATH["videos"] + placeid + "/" + placeid + imgname + '.webm', fourcc, 10, (width, height))
    elif vformat == "mp4":
        # https://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/#comment-390650
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(PATH["videos"] + placeid + "/" + placeid + imgname + '.mp4', fourcc, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(PATH["plots_networks"] + placeid + "/", image)))
    # Add the last frame duplicatelastframe more times:
    for i in range(0, duplicatelastframe):
        video.write(cv2.imread(os.path.join(PATH["plots_networks"] + placeid + "/", images[-1])))

    cv2.destroyAllWindows()
    video.release()
    if verbose:
        print("Video " + placeid + imgname + '.' + vformat + ' generated from ' + str(len(images)) + " frames.")




def write_result(res, mode, placeid, poi_source, prune_measure, suffix, dictnested={}, weighting=None, scenario=None):
    """Write results (pickle or dict to csv), now supports scenario subfolders and scenario-tagged filenames"""
    if mode == "pickle":
        openmode = "wb"
    else:
        openmode = "w"

    # Modify filename based on weighting flag
    weighting_str = "_weighted" if weighting else ""
    scenario_str = f"_{scenario}" if scenario else ""

    # Construct the filename
    if poi_source:
        if prune_measure:
            filename = f"{placeid}_poi_{poi_source}_{prune_measure}{weighting_str}{scenario_str}{suffix}"
        else:
            filename = f"{placeid}_poi_{poi_source}{weighting_str}{scenario_str}{suffix}"
    else:
        if prune_measure:
            filename = f"{placeid}_{prune_measure}{weighting_str}{scenario_str}{suffix}"
        else:
            filename = f"{placeid}{weighting_str}{scenario_str}{suffix}"

    # Create scenario folder path
    folder_path = os.path.join(PATH["results"], placeid, scenario if scenario else "")
    os.makedirs(folder_path, exist_ok=True)

    # set file path to absolute, normalised path to ensure it works across different OS!
    file_path = os.path.abspath(os.path.normpath(os.path.join(folder_path, filename)))

    with open(file_path, openmode) as f:
        if mode == "pickle":
            pickle.dump(res, f)
        elif mode == "dict":
            w = csv.writer(f)
            w.writerow(res.keys())
            try:  # dict with list values
                w.writerows(zip(*res.values()))
            except:  # dict with single values
                w.writerow(res.values())
        elif mode == "dictnested":
            fields = ['network'] + list(dictnested.keys())
            w = csv.DictWriter(f, fields)
            w.writeheader()
            for key, val in sorted(res.items()):
                row = {'network': key}
                row.update(val)
                w.writerow(row)


                

def write_result_covers(res, mode, placeid, suffix, dictnested={}, weighting=None):
    # makes results format place_existing_covers_weighted.csv etc. 
    """Write results (pickle or dict to csv), with _weighted before the file extension if needed
    """
    if mode == "pickle":
        openmode = "wb"
        file_extension = ".pickle"
    else:
        openmode = "w"
        file_extension = ".csv"

    # Modify filename to append '_weighted' before the file extension if weighting is True
    if weighting:
        suffix = suffix.replace(file_extension, "") + "_weighted" + file_extension
    else:
        suffix += file_extension

    # Construct the filename
    filename = placeid + "_" + suffix

    # Write the file
    with open(PATH["results"] + placeid + "/" + filename, openmode) as f:
        if mode == "pickle":
            pickle.dump(res, f)
        elif mode == "dict":
            w = csv.writer(f)
            w.writerow(res.keys())
            try:  # dict with list values
                w.writerows(zip(*res.values()))
            except:  # dict with single values
                w.writerow(res.values())
        elif mode == "dictnested":
            # Writing nested dictionary to CSV
            fields = ['network'] + list(dictnested.keys())
            w = csv.DictWriter(f, fields)
            w.writeheader()
            for key, val in sorted(res.items()):
                row = {'network': key}
                row.update(val)
                w.writerow(row)




def gdf_to_geojson(gdf, properties):
    """Turn a gdf file into a GeoJSON.
    The gdf must consist only of geometries of type Point.
    Adapted from: https://geoffboeing.com/2015/10/exporting-python-data-geojson/
    """
    geojson = {'type':'FeatureCollection', 'features':[]}
    for _, row in gdf.iterrows():
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Point',
                               'coordinates':[]}}
        feature['geometry']['coordinates'] = [row.geometry.x, row.geometry.y]
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson



def ig_to_shapely(G):
    """Turn an igraph graph G to a shapely LineString
    """
    edgetuples = [((e.source_vertex["x"], e.source_vertex["y"]), (e.target_vertex["x"], e.target_vertex["y"])) for e in G.es]
    G_shapely = LineString()
    for t in edgetuples:
        G_shapely = ops.unary_union([G_shapely, LineString(t)])
    return G_shapely


# Neighbourhoods

def load_neighbourhoods(path, debug=False):
    """
    Load all neighbourhoods geopackages with 'scored_neighbourhoods_' in the filename. 

    Parameters:
        path (str): The base path where the GeoPackage files are located.
    Returns:
        dict: A dictionary with cleaned filenames as keys and GeoDataFrames as values.
    """
    # Construct the path to the GeoPackage directory
    gpkg_dir = os.path.join(path)
    geopackages = {}
    # Define the prefix to remove
    prefix = "scored_neighbourhoods_"

    # Check if the directory exists
    if os.path.exists(gpkg_dir):
        # Iterate over all files in the directory
        for filename in os.listdir(gpkg_dir):
            if filename.endswith('.gpkg') and "scored_neighbourhoods_" in filename:  # Check for GeoPackage files with the desired prefix
                # Construct the full path to the GeoPackage file
                gpkg_path = os.path.join(gpkg_dir, filename)
                try:
                    # Load the GeoPackage into a GeoDataFrame
                    gdf = gpd.read_file(gpkg_path)
                    
                    # Remove the .gpkg extension from the filename
                    city_name = os.path.splitext(filename)[0]
                    
                    # Remove the "scored_neighbourhoods_" prefix if it exists
                    if city_name.startswith(prefix):
                        city_name = city_name[len(prefix):]
                    
                    if debug:
                        print(filename)
                        print(gpkg_dir)
                    # Add the cleaned filename (city_name) and GeoDataFrame to the dictionary
                    geopackages[city_name] = gdf
                except Exception as e:
                    print(f"Error loading GeoPackage {filename}: {e}")
    else:
        print(f"Directory does not exist: {gpkg_dir}")

    print(f"{len(geopackages)} Cities loaded")
    
    return geopackages

print("Loaded functions.\n")


def ox_gpkg_to_graph(gpkg_path, nodes_layer='nodes', edges_layer='edges', u_col='u', v_col='v', key_col='key'):
    """
    Reads a GeoPackage created from ox.graph_to_geopackage, splits it into nodes and edges, resets the MultiIndex, and creates an OSMnx graph.

    Parameters:
        gpkg_path (str): Path to the GeoPackage file.
        nodes_layer (str): Name of the nodes layer in the GeoPackage. Default is 'nodes'.
        edges_layer (str): Name of the edges layer in the GeoPackage. Default is 'edges'.
        u_col (str): Column name for the start node ID in the edges layer. Default is 'u'.
        v_col (str): Column name for the end node ID in the edges layer. Default is 'v'.
        key_col (str): Column name for the edge key in the edges layer. Default is 'key'.

    Returns:
        G (networkx.MultiDiGraph): The OSMnx graph created from the nodes and edges.
    """
    # Read the nodes and edges layers from the GeoPackage
    nodes_gdf = gpd.read_file(gpkg_path, layer=nodes_layer)
    edges_gdf = gpd.read_file(gpkg_path, layer=edges_layer)

    # set the index for nodes_gdf
    nodes_gdf.set_index('osmid', inplace=True)

    # Ensure the required columns exist in the edges_gdf
    if not all(col in edges_gdf.columns for col in [u_col, v_col, key_col]):
        raise ValueError(f"edges_gdf must contain columns '{u_col}', '{v_col}', and '{key_col}'.")

    # Set the MultiIndex for edges_gdf
    edges_gdf.set_index([u_col, v_col, key_col], inplace=True)

    # Create the OSMnx graph
    G = ox.graph_from_gdfs(nodes_gdf, edges_gdf)

    return G

def nearest_edge_between_polygons(G, poly1, poly2):
    """Find the shortest path between the edges of two polygons based on routing distance."""
    min_dist = float('inf')
    best_pair = None

    # Get edges of both polygons as lists of coordinate pairs
    poly1_edges = list(zip(poly1.exterior.coords[:-1], poly1.exterior.coords[1:]))
    poly2_edges = list(zip(poly2.exterior.coords[:-1], poly2.exterior.coords[1:]))

    # Iterate over all edges of both polygons
    for edge1 in poly1_edges:
        for edge2 in poly2_edges:
            # Use existing graph's shortest path function between edge points
            sp = G.get_shortest_paths(edge1[0], edge2[0], weights='weight', output='vpath')
            dist = sum([G.es[e]["weight"] for e in sp[0]]) # Add the weights of the shortest path

            if dist < min_dist:
                min_dist = dist
                best_pair = (edge1[0], edge2[0])

    return best_pair, min_dist





def greedy_triangulation_polygon_routing(G, pois, weighting=None, prune_quantiles = [1], prune_measure = "betweenness"):
    """Greedy Triangulation (GT) of a graph G's node subset pois,
    then routing to connect the GT (up to a quantile of betweenness
    betweenness_quantile).
    G is an ipgraph graph, pois is a list of node ids.
    
    The GT connects pairs of nodes in ascending order of their distance provided
    that no edge crossing is introduced. It leads to a maximal connected planar
    graph, while minimizing the total length of edges considered. 
    See: cardillo2006spp
    
    Distance here is routing distance, while edge crossing is checked on an abstract 
    level.
    """
    
    if len(pois) < 2: return ([], []) # We can't do anything with less than 2 POIs

    # GT_abstract is the GT with same nodes but euclidian links to keep track of edge crossings
    pois_indices = set()
    for poi in pois:
        pois_indices.add(G.vs.find(id = poi).index)
    G_temp = copy.deepcopy(G)
    for e in G_temp.es: # delete all edges
        G_temp.es.delete(e)
        
    poipairs = poipairs_by_distance(G, pois, weighting, True)
    if len(poipairs) == 0: return ([], [])

    if prune_measure == "random":
        # run the whole GT first
        GT = copy.deepcopy(G_temp.subgraph(pois_indices))
        for poipair, poipair_distance in poipairs:
            poipair_ind = (GT.vs.find(id = poipair[0]).index, GT.vs.find(id = poipair[1]).index)
            if not new_edge_intersects(GT, (GT.vs[poipair_ind[0]]["x"], GT.vs[poipair_ind[0]]["y"], GT.vs[poipair_ind[1]]["x"], GT.vs[poipair_ind[1]]["y"])):
                GT.add_edge(poipair_ind[0], poipair_ind[1], weight = poipair_distance)
        # create a random order for the edges
        random.seed(0) # const seed for reproducibility
        edgeorder = random.sample(range(GT.ecount()), k = GT.ecount())
    else: 
        edgeorder = False
    
    GT_abstracts = []
    GTs = []
    for prune_quantile in tqdm(prune_quantiles, desc = "Greedy triangulation", leave = False):
        GT_abstract = copy.deepcopy(G_temp.subgraph(pois_indices))
        GT_abstract = greedy_triangulation(GT_abstract, poipairs, prune_quantile, prune_measure, edgeorder)
        GT_abstracts.append(GT_abstract)
        
        # Get node pairs we need to route, sorted by distance
        routenodepairs = {}
        for e in GT_abstract.es:
            routenodepairs[(e.source_vertex["id"], e.target_vertex["id"])] = e["weight"]
        routenodepairs = sorted(routenodepairs.items(), key = lambda x: x[1])

        # Do the routing
        GT_indices = set()
        for poipair, poipair_distance in routenodepairs:
            poipair_ind = (G.vs.find(id = poipair[0]).index, G.vs.find(id = poipair[1]).index)
            # debug
            #print(f"Edge weights before routing: {G.es['weight'][:10]}")  # Prints first 10 weights
            #print(f"Routing between: {poipair[0]} and {poipair[1]} with distance: {poipair_distance}")
            sp = set(G.get_shortest_paths(poipair_ind[0], poipair_ind[1], weights = "weight", output = "vpath")[0])
            #print(f"Shortest path between {poipair[0]} and {poipair[1]}: {sp}")

            GT_indices = GT_indices.union(sp)

        GT = G.induced_subgraph(GT_indices)
        GTs.append(GT)
    
    return (GTs, GT_abstracts)
    

def get_neighbourhood_centroids(gdf):
    """
    Find the centroid of each neighbourhood

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame containing the city's polygons (neighbourhoods).

    Returns:
        GeoDataFrame: A GeoDataFrame containing the centroids moved to the nearest edge.
    """
    centroids = gdf.geometry.centroid  # Calculate centroids for each polygon
    centroids_gdf = gpd.GeoDataFrame({'neighbourhood_id': gdf['neighbourhood_id'], 'geometry': centroids}, crs=gdf.crs) 
    
    return centroids_gdf 

def prepare_neighbourhoods(cities):
    """
    Convert columns from strings to numbers. Quirk of exporting geopackages with numbers...

    Parameters:
        cities (dict): A dictionary with filenames as keys and GeoDataFrames as values.
    """
    for city_name, gdf in cities.items():
        for column in gdf.columns:
            if column != 'geometry':
                try:
                    gdf[column] = pd.to_numeric(gdf[column], errors='coerce')
                except Exception as e:
                    print(f"Error converting column '{column}' in '{city_name}': {e}")

    return cities

    
def get_neighbourhood_streets(gdf, debug=False):
    """"
    Get all the streets within each neighbourhood.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame containing neighbourhoods.
    Returns:            
        gdf of nodes and edges within neighbourhoods
    """ 
    # Add a unique ID column to the GeoDataFrame
    print(f"GeoDataFrame shape before adding ID: {gdf.shape}")

    gdf['ID'] = range(1, len(gdf) + 1)  # Adding ID column starting from 1

    # create bounding box slightly larger than neighbourhoods
    gdf_mercator = gdf.to_crs(epsg=3857)
    gdf_mercator = gdf_mercator.buffer(1000)
    gdf_buffered = gpd.GeoDataFrame(geometry=gdf_mercator, crs="EPSG:3857").to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf_buffered.total_bounds
    
    # get driving network (we're only interested in streets cars could be on)
    network = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(network)
    edges = gpd.sjoin(edges, gdf[['ID', 'overall_score', 'geometry']], how="left", op='intersects')

    if debug == True:
        unique_ids = edges['ID'].dropna().unique()
        np.random.seed(42) 
        random_colors = {ID: mcolors.to_hex(np.random.rand(3)) for ID in unique_ids}
        edges['color'] = edges['ID'].map(random_colors)
        edges['color'] = edges['color'].fillna('#808080')  # Gray for NaN values
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        edges.plot(ax=ax, color=edges['color'], legend=False) 
        ax.set_title('Edges colored randomly by Neighbourhood ID')
        plt.show()

    return nodes, edges


def get_neighbourhood_street_graph(gdf, debug=False):
    """"
    Get all the streets within each neighbourhood.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame containing neighbourhoods.
    Returns:            
        gdf of nodes and edges within neighbourhoods
    """ 
    # Add a unique ID column to the GeoDataFrame
    print(f"GeoDataFrame shape before adding ID: {gdf.shape}")
    gdf['ID'] = range(1, len(gdf) + 1)  # Adding ID column starting from 1

    # create bounding box slightly larger than neighbourhoods
    gdf_mercator = gdf.to_crs(epsg=3857)
    gdf_mercator = gdf_mercator.buffer(10)
    gdf_buffered = gpd.GeoDataFrame(geometry=gdf_mercator, crs="EPSG:3857").to_crs(epsg=4326)
    
    # get driving network (we're only interested in streets cars could be on)
    network = nx.MultiDiGraph()
    for i, polygon in enumerate(gdf_buffered.geometry):
        try:
            net = ox.graph_from_polygon(polygon, network_type='drive')
            if len(net) == 0:
                print(f"Polygon {i}: Empty graph returned. Skipping.")
                continue
            network = nx.compose(network, net)
        except ValueError as e:
            print(f"Polygon {i}: Skipping due to ValueError: {e}")
            continue
        except Exception as e:
            print(f"Polygon {i}: Skipping due to other error: {e}")
            continue

    if len(network.nodes) == 0:
        print("No valid network data was found in any neighbourhood.")
        # Return empty GeoDataFrames and empty graph
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), nx.MultiDiGraph() 
    nodes, edges = ox.graph_to_gdfs(network)
    edges = gpd.sjoin(edges, gdf[['ID', 'overall_score', 'geometry']], how="left", op='intersects')
    exclude_conditions = (
        (edges['highway'].isin(['trunk', 'trunk_link', 'motorway', 'motorway_link',
                                'primary', 'primary_link', 'secondary', 'secondary_link',
                                'tertiary', 'tertiary_link'])) |
        (edges['maxspeed'].isin(['60 mph', '70 mph', '40 mph', 
                                 ('30 mph', '60 mph'), ('30 mph', '50 mph'), 
                                 ('70 mph', '50 mph'), ('40 mph', '60 mph'), 
                                 ('70 mph', '60 mph'), ('60 mph', '40 mph'), 
                                 ('50 mph', '40 mph'), ('30 mph', '40 mph'), 
                                 ('20 mph', '60 mph'), ('70 mph', '40 mph'), 
                                 ('30 mph', '70 mph')]))
    ) 
    edges = edges[~exclude_conditions] # remove any high stress roads if we pick them up

    if debug == True:
        unique_ids = edges['ID'].dropna().unique()
        np.random.seed(42) 
        random_colors = {ID: mcolors.to_hex(np.random.rand(3)) for ID in unique_ids}
        edges['color'] = edges['ID'].map(random_colors)
        edges['color'] = edges['color'].fillna('#808080')  # Gray for NaN values
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        edges.plot(ax=ax, color=edges['color'], legend=False) 
        ax.set_title('Edges colored randomly by Neighbourhood ID')
        plt.show()


    edges = edges.dropna(subset=['ID'])
    G = ox.graph_from_gdfs(nodes, edges)
    
    return nodes, edges, G








def get_neighbourhood_streets_split(gdf, debug):
    """"
    Get all the streets within each neighbourhood.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame containing neighbourhoods.
    Returns:            
        gdf of nodes and edges within neighbourhoods
    """ 
    # add ID for export
    gdf['ID'] = range(1, len(gdf) + 1)  # Adding ID column starting from 1

    # create bounding box slightly larger than neighbourhoods
    gdf_mercator = gdf.to_crs(epsg=3857)
    gdf_mercator = gdf_mercator.buffer(1000)
    gdf_buffered = gpd.GeoDataFrame(geometry=gdf_mercator, crs="EPSG:3857").to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf_buffered.total_bounds
    
    # get street network (we want to consider all exit/entry points)
    network = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='all')
    nodes, edges = ox.graph_to_gdfs(network)
    def get_boundary_graph(network):
        """
        Create a graph from bounding roads.

        Args:
            network (networkx.Graph): The original graph.

        Returns:
            networkx.Graph: The boundary graph.
        """
        boundary_g = network.copy()
        # Define the conditions for keeping edges
        conditions = [
        (
            data.get('highway') in ['trunk', 'trunk_link', 'motorway', 'motorway_link', 'primary', 'primary_link',
                                    'secondary', 'secondary_link', 'tertiary', 'tertiary_link'] 
        ) or (
            data.get('maxspeed') in ['60 mph', '70 mph', '40 mph', 
                                    ('30 mph', '60 mph'), ('30 mph', '50 mph'), 
                                    ('70 mph', '50 mph'), ('40 mph', '60 mph'), 
                                    ('70 mph', '60 mph'), ('60 mph', '40 mph'),
                                    ('50 mph', '40 mph'), ('30 mph', '40 mph'),
                                    ('20 mph', '60 mph'), ('70 mph', '40 mph'), 
                                    ('30 mph', '70 mph')]
            ) 
            for u, v, k, data in boundary_g.edges(keys=True, data=True)
        ]
        # Keep only the edges that satisfy the conditions
        edges_to_remove = [
            (u, v, k) for (u, v, k), condition in zip(boundary_g.edges(keys=True), conditions) if not condition
        ]
        boundary_g.remove_edges_from(edges_to_remove)
        # Clean nodes by removing isolated nodes from the graph
        isolated_nodes = list(nx.isolates(boundary_g))
        boundary_g.remove_nodes_from(isolated_nodes)
        return boundary_g
    boundary_g = get_boundary_graph(network)
    filtered_g = network.copy()
    filtered_g.remove_edges_from(boundary_g.edges())
    filtered_g.remove_nodes_from(boundary_g.nodes())
    # Clip the graph to the boundary (but make the boundary slightly larger)
    filtered_g_edges = ox.graph_to_gdfs(filtered_g, nodes=False)
    filtered_g_nodes = ox.graph_to_gdfs(filtered_g, edges=False)
    filtered_g_edges = filtered_g_edges.clip(gdf)
    filtered_g_nodes = filtered_g_nodes.clip(gdf)

    # Add new nodes at the end of any edge which has been truncated
    # this is needed to ensure that the graph is correctly disconnected
    # at boundary roads
    def add_end_nodes(filtered_g_nodes, filtered_g_edges):
        # Create a GeoSeries of existing node geometries for spatial operations
        existing_node_geometries = gpd.GeoSeries(filtered_g_nodes.geometry.tolist())
        new_nodes = []
        new_edges = []
        # Iterate through each edge to check its endpoints
        for idx, edge in filtered_g_edges.iterrows():
            geometries = [edge.geometry] if isinstance(edge.geometry, LineString) else edge.geometry.geoms
            # Loop through each geometry in the edge
            for geom in geometries:
                u = geom.coords[0]  # Start point (first coordinate)
                v = geom.coords[-1]  # End point (last coordinate)
                # Check if the start point exists
                if not existing_node_geometries.contains(Point(u)).any():
                    # Create a new node at the start point
                    new_node = gpd.GeoDataFrame(geometry=[Point(u)], crs=filtered_g_nodes.crs)
                    new_node['id'] = f'new_{len(filtered_g_nodes) + len(new_nodes)}'  # Generate a unique id
                    new_node['x'] = u[0]
                    new_node['y'] = u[1]
                    new_nodes.append(new_node)
                # Check if the end point exists
                if not existing_node_geometries.contains(Point(v)).any():
                    # Create a new node at the end point
                    new_node = gpd.GeoDataFrame(geometry=[Point(v)], crs=filtered_g_nodes.crs)
                    new_node['id'] = f'new_{len(filtered_g_nodes) + len(new_nodes)}'  # Generate a unique id
                    new_node['x'] = v[0]
                    new_node['y'] = v[1]
                    new_nodes.append(new_node)
                # Add new edges to new_edges list if endpoints are new nodes
                new_edges.append((u, v, geom))  # Keep the geometry of the edge
        # Combine new nodes into a GeoDataFrame
        if new_nodes:
            new_nodes_gdf = pd.concat(new_nodes, ignore_index=True)
            filtered_g_nodes = gpd.GeoDataFrame(pd.concat([filtered_g_nodes, new_nodes_gdf], ignore_index=True))
        return filtered_g_nodes, filtered_g_edges
    filtered_g_nodes, filtered_g_edges = add_end_nodes(filtered_g_nodes, filtered_g_edges)
    # Rebuild graph with new end nodes
    filtered_g.clear()
    filtered_g = nx.MultiDiGraph()

    # add nodes and edges back in
    unique_nodes = set()
    for _, row in filtered_g_edges.iterrows():
        if row.geometry.type == 'LineString':
            coords = list(row.geometry.coords)
        elif row.geometry.type == 'MultiLineString':
            coords = [coord for line in row.geometry.geoms for coord in line.coords]
        unique_nodes.update(coords)
    # Add nodes with attributes
    for node in unique_nodes:
        if isinstance(node, tuple):
            x, y = node
            filtered_g.add_node(node, x=x, y=y, geometry=Point(x, y))
    # Add nodes from filtered_g_nodes
    for idx, row in filtered_g_nodes.iterrows():
        node_id = idx
        if node_id not in filtered_g.nodes:
            filtered_g.add_node(node_id, osmid=node_id, x=row.geometry.x, y=row.geometry.y, geometry=row.geometry)
    # Add edges
    for _, row in filtered_g_edges.iterrows():
        if row.geometry.type == 'LineString':
            coords = list(row.geometry.coords)
            for i in range(len(coords) - 1):
                filtered_g.add_edge(coords[i], coords[i + 1], geometry=row.geometry, osmid=row['osmid'])
        elif row.geometry.type == 'MultiLineString':
            for line in row.geometry.geoms:
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    filtered_g.add_edge(coords[i], coords[i + 1], geometry=line, osmid=row['osmid'])
    # Assign osmids to nodes with coordinate IDs
    # this ensure omsnx compatibility
    for node, data in filtered_g.nodes(data=True):
        if isinstance(node, tuple):
            # Find an edge that contains this node (dirty method of getting osmid)
            for u, v, key, edge_data in filtered_g.edges(keys=True, data=True):
                if node in [u, v]:
                    data['osmid'] = edge_data['osmid']
                    break
    filtered_g.graph['crs'] = 'EPSG:4326' # give the graph a crs
    neighbourhood_graphs = filtered_g
    nodes, edges = ox.graph_to_gdfs(filtered_g)
    # Set 'osmid' as the index, replacing the old index
    nodes = nodes.set_index('osmid', drop=True)

    edges = gpd.sjoin(edges, gdf[['ID', 'overall_score', 'geometry']], how="left", op='intersects')

    if debug == True:
        # plot out the network into its "neighbourhoods"
        network = filtered_g
        undirected_network = network.to_undirected()
        connected_components = list(nx.connected_components(undirected_network))
        colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(connected_components))]
        edge_color_map = {}
        for color, component in zip(colors, connected_components):
            component_edges = []
            for node in component:
                for neighbor in undirected_network.neighbors(node):
                    edge = (node, neighbor)
                    reverse_edge = (neighbor, node)
                    if edge in network.edges or reverse_edge in network.edges:
                        component_edges.append(edge)

            # Assign the same color to all edges in the component
            for edge in set(component_edges): 
                edge_color_map[edge] = color
        edge_colors = []
        for edge in network.edges:
            # Ensure we look for both directions in the edge color map
            edge_colors.append(edge_color_map.get(edge, edge_color_map.get((edge[1], edge[0]), 'black')))  # Default to black if not found

        # Draw the network without nodes, increase figsize for larger plot
        fig, ax = plt.subplots(figsize=(20, 15))  # Set the desired figure size
        ox.plot_graph(network, ax=ax, node_color='none', edge_color=edge_colors,
                    edge_linewidth=2, show=False)
        plt.show()

    return nodes, edges, neighbourhood_graphs




def get_exit_nodes(neighbourhoods, G_biketrack, buffer_distance=5):
    """
    Find bike track (and LTNs if present) nodes within a buffer of neighbourhood boundaries.
    
    Args:
        neighbourhoods (dict): Dictionary of neighbourhood GeoDataFrames
        G_biketrack (networkx.Graph): Bike track network graph
        buffer_distance (int): Buffer distance in meters
        
    Returns:
        GeoDataFrame: Nodes within the buffer
    """
    # Get bike track nodes
    nodes = ox.graph_to_gdfs(G_biketrack, nodes=True, edges=False)
    # Create buffer around neighbourhood boundaries
    all_buffers = []
    for place_name, gdf in neighbourhoods.items():
        # Explode multi-part geometries and create boundary buffer
        exploded = gdf.explode().reset_index(drop=True)
        buffered = exploded.boundary.to_crs(epsg=3857).buffer(buffer_distance).to_crs(exploded.crs)
        # Store buffers with neighbourhood IDs
        buffer_gdf = gpd.GeoDataFrame({
            'neighbourhood_id': exploded.index,
            'geometry': buffered
        }, crs=exploded.crs)
        
        all_buffers.append(buffer_gdf)
    
    # Combine all buffers into a single GeoDataFrame
    combined_buffers = gpd.GeoDataFrame(pd.concat(all_buffers, ignore_index=True), crs=all_buffers[0].crs)
    
    # Find nodes intersecting with any buffer
    nodes_in_buffer = gpd.sjoin(nodes, combined_buffers, how='inner', op='intersects')
    
    # Clean up columns
    nodes_in_buffer = nodes_in_buffer[['geometry', 'neighbourhood_id']].reset_index()
    
    return nodes_in_buffer


def greedy_triangulation_routing_GT_abstracts(G, pois, weighting=None, prune_quantiles=[1], prune_measure="betweenness"):
    """Generates Greedy Triangulation (GT_abstracts) of a graph G's node subset pois.
    This version focuses only on GT_abstracts without generating GTs.
    """

    if len(pois) < 2:
        return []  # We can't do anything with less than 2 POIs

    # Initialize the POI indices and an empty copy of G
    pois_indices = {G.vs.find(id=poi).index for poi in pois}
    G_temp = copy.deepcopy(G)
    for e in G_temp.es:
        G_temp.es.delete(e)  # Delete all edges in G_temp

    poipairs = poipairs_by_distance(G, pois, weighting, True)
    if not poipairs:
        return []

    # If prune_measure is "random", define edge order
    edgeorder = False
    if prune_measure == "random":
        GT = copy.deepcopy(G_temp.subgraph(pois_indices))
        for poipair, poipair_distance in poipairs:
            poipair_ind = (
                GT.vs.find(id=poipair[0]).index, 
                GT.vs.find(id=poipair[1]).index
            )
            if not new_edge_intersects(
                GT, (
                    GT.vs[poipair_ind[0]]["x"], 
                    GT.vs[poipair_ind[0]]["y"], 
                    GT.vs[poipair_ind[1]]["x"], 
                    GT.vs[poipair_ind[1]]["y"]
                )
            ):
                GT.add_edge(poipair_ind[0], poipair_ind[1], weight=poipair_distance)
        
        # Define a random edge order
        random.seed(0)  # Constant seed for reproducibility
        edgeorder = random.sample(range(GT.ecount()), k=GT.ecount())
    
    # Generate GT_abstracts for each prune_quantile
    GT_abstracts = []
    for prune_quantile in tqdm(prune_quantiles, desc="Greedy triangulation", leave=False):
        GT_abstract = copy.deepcopy(G_temp.subgraph(pois_indices))
        GT_abstract = greedy_triangulation(GT_abstract, poipairs, prune_quantile, prune_measure, edgeorder)
        GT_abstracts.append(GT_abstract)
    
    return GT_abstracts



def get_urban_areas(place):
    def set_location_boundary(place):
        """
        Sets up the location boundary by geocoding the given place and buffering it.

        Parameters:
        place (str): The name or address of the place to geocode.

        Returns:
        geopandas.GeoDataFrame: The buffered boundary of the location.
        """
        # Set location and get boundary
        boundary = ox.geocode_to_gdf(place)
        boundary = boundary.to_crs('3857') # we convert to EPSG 3857 to buffer in meters
        boundary_buffered = boundary.buffer(100) # Buffer boundary to prevent strange edge cases...

        return boundary_buffered, boundary

    ## get urban footprints from GUF

    def get_guf(place):
        """
        Retrieves a clipped GeoDataFrame of GUF urban areas within a specified place boundary.

        Parameters:
        - place (str): The name or address of the place to retrieve urban areas for.

        Returns:
        - gdf_clipped (GeoDataFrame): A GeoDataFrame containing the clipped urban areas within the specified place boundary.
        """

        # Step 1: Access the WMS Service
        wms_url = 'https://geoservice.dlr.de/eoc/land/wms?GUF04_DLR_v1_Mosaic'
        wms = WebMapService(wms_url, version='1.1.1')

        # Step 2: Identify the Layer with ID 102. This is the Global Urban Footprint layer GUF
        for layer_name, layer in wms.contents.items():
            if '102' in layer_name:
                print(f"Layer ID 102 found: {layer_name}")

        # Assuming 'GUF04_DLR_v1_Mosaic' is the layer with ID 102
        layer = 'GUF04_DLR_v1_Mosaic'  # Replace with the actual layer name if different

        # Step 3: Get the polygon boundary using osmnx

        boundary_gdf = ox.geocode_to_gdf(place)

        boundary = boundary_gdf.to_crs('EPSG:3857')
        # buffer boundary to ensure clips include riverlines which may act as borders between geographies
        boundary_buffered = boundary.buffer(100)
        boundary_buffered = boundary_buffered.to_crs('EPSG:4326')
        boundary_polygon = boundary_gdf.geometry[0]
        wms_boundary = boundary_buffered.geometry[0]

        # Convert the polygon to a bounding box
        minx, miny, maxx, maxy = wms_boundary.bounds

        # Step 4: Request the data from WMS using the bounding box
        width = 1024
        height = 1024
        response = wms.getmap(
            layers=[layer],
            srs='EPSG:4326',
            bbox=(minx, miny, maxx, maxy),
            size=(width, height),
            format='image/geotiff'
        )

        # Step 5: Load the Raster Data into Rasterio
        with MemoryFile(response.read()) as memfile:
            with memfile.open() as src:
                image = src.read(1)  # Read the first band
                transform = src.transform
                crs = src.crs

                # Clip the raster data to the polygon
                out_image, out_transform = rio_mask(src, [mapping(wms_boundary)], crop=True)  # Use renamed mask function
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff",
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": out_transform,
                                "crs": crs})

        # Step 6: Convert Raster to Vector
        mask_arr = (out_image[0] != 0).astype(np.uint8)  # Assuming non-zero values are urban areas

        shapes_gen = shapes(mask_arr, mask=mask_arr, transform=out_transform)

        polygons = []
        for geom, value in shapes_gen:
            polygons.append(shape(geom))

        # Create a GeoDataFrame from the polygons
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)

        # Step 7: Create Buffers Around Urban Areas
        buffer_distance = 100  # Buffer distance in meters (adjust as needed)
        gdf_buffered = gdf.copy()
        gdf_buffered['geometry'] = gdf['geometry'].buffer(buffer_distance)

        # Step 8: Clip the GeoDataFrame to the boundary of the place
        gdf_clipped = gpd.clip(gdf, boundary_gdf)

        return gdf_clipped

    ## get residential areas
    def get_residential_areas(polygon):
        polygon = polygon.to_crs('EPSG:4326')
        # Retrieve features from OpenStreetMap
        features = ox.features_from_polygon(polygon.iloc[0], tags={'landuse': 'residential'})
        
        # Convert features to a GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(features)
        gdf = gdf.set_crs('EPSG:4326')
        
        return gdf



    ## join urban foot prints and residential areas
    # this is to create a single polygon of where neighbourhoods can be found within

    def join_geodataframes(gdf1, gdf2):
        # Ensure both GeoDataFrames have the exact same CRS
        target_crs = 'EPSG:4326'  # WGS 84
        gdf1 = gdf1.to_crs(target_crs)
        gdf2 = gdf2.to_crs(target_crs)
        
        # Concatenate GeoDataFrames
        joined_gdf = pd.concat([gdf1, gdf2], ignore_index=True)
        
        return gpd.GeoDataFrame(joined_gdf, crs=target_crs)





    ## create a small buffer to ensure all areas a captured correctly

    def buffer_geometries_in_meters(gdf, distance):
        # Define the World Mercator projected CRS
        projected_crs = 'EPSG:3857'  # World Mercator

        # Project to the new CRS
        gdf_projected = gdf.to_crs(projected_crs)
        
        # Buffer the geometries
        gdf_projected['geometry'] = gdf_projected['geometry'].buffer(distance)
        
        # Reproject back to the original CRS
        gdf_buffered = gdf_projected.to_crs(gdf.crs)
        
        return gdf_buffered



    ## union into one gdf

    def unary_union_polygons(gdf):
        # Combine all geometries into a single geometry
        unified_geometry = unary_union(gdf['geometry'])
        
        # Create a new GeoDataFrame with a single row containing the unified geometry
        combined_gdf = gpd.GeoDataFrame({'geometry': [unified_geometry]}, crs=gdf.crs)
        
        return combined_gdf
    
    boundary_buffered, boundary = set_location_boundary(place)
    guf = get_guf(place)
    residential_areas = get_residential_areas(boundary_buffered)

    guf_residential_gdf = join_geodataframes(guf, residential_areas)
    guf_residential_gdf = buffer_geometries_in_meters(guf_residential_gdf, 100)  # Buffer by 100 meters
    guf_residential_gdf = unary_union_polygons(guf_residential_gdf)

    return(guf_residential_gdf)



def process_maxspeeds(graph):
    """
    Process the 'maxspeed' attributes in the edges of the given graph.
    If 'maxspeed' is a list, only keep the first item. Otherwise, leave it as is.

    Parameters:
        graph (networkx.Graph): The input graph with edges containing 'maxspeed' attributes.

    Returns:
        networkx.Graph: The graph with processed 'maxspeed' attributes.
    """
    # Function to extract the first speed if maxspeed is a list
    def get_first_speed(maxspeed):
        if isinstance(maxspeed, list):  # Check if maxspeed is a list
            return maxspeed[0]  # Return the first item
        return maxspeed  # Otherwise, return as is

    # Iterate through all edges and process 'maxspeed'
    for u, v, data in graph.edges(data=True):
        if 'maxspeed' in data:
            data['maxspeed'] = get_first_speed(data['maxspeed'])  # Process and update 'maxspeed'

    return graph

def process_lists(graph):
    """
    Process the attributes in the edges of the given graph.
    If any attribute value is a list, only keep the first item. Otherwise, leave it as is.

    Parameters:
        graph (networkx.Graph): The input graph with edges containing attributes.

    Returns:
        networkx.Graph: The graph with processed attributes, where list attributes are reduced to their first item.
    """
    # Function to extract the first item from a list if the attribute is a list
    def get_first_item(attribute_value):
        if isinstance(attribute_value, list):  # Check if the attribute value is a list
            return attribute_value[0]  # Return the first item of the list
        return attribute_value  # Otherwise, return the value as is

    # Iterate through all edges and process their attributes
    for u, v, data in graph.edges(data=True):
        for attr, value in data.items():
            data[attr] = get_first_item(value)  # Process each attribute

    return graph



def get_nearest_nodes_to_gdf(G, nodes_gdf):
     """
    Get the nearest node IDs from a graph G for the points in a GeoDataFrame.
    
    Parameters:
        G (nx.MultiDiGraph): The graph to search for nearest nodes.
        nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries.

    Returns:
        nearest_node_ids (pd.Series): A Pandas Series of nearest node IDs.
    """
     x_coords = nodes_gdf.geometry.x
     y_coords = nodes_gdf.geometry.y
     nearest_node_ids = ox.distance.nearest_nodes(G, x_coords, y_coords, return_dist=False)
     
     return nearest_node_ids





def greedy_triangulation_ltns(ltn_points_gdf):
    """
    Function to create a greedy triangulation for LTN nodes only
    Used to find the ID of ltn nodes' neighbours

    parameters:
        ltn_points_gdf (gpd.GeoDataFrame): GeoDataFrame containing LTN points

    returns:
    """
    ltn_points_gdf = ltn_points_gdf.copy()  
    ltn_points_gdf = ltn_points_gdf.to_crs('EPSG:3857')  # Convert to a metric CRS
    
    # Extract coordinates of points
    coords = np.array([(point.x, point.y) for point in ltn_points_gdf.geometry])
    
    # Compute all possible edges with their distances
    edges = []
    for i, j in itertools.combinations(range(len(coords)), 2):
        distance = np.linalg.norm(coords[i] - coords[j])
        edges.append((i, j, distance))
    
    # Sort edges by distance
    edges = sorted(edges, key=lambda x: x[2])
    
    selected_edges = []
    existing_lines = GeometryCollection()
    
    # Iterate over edges
    for i, j, distance in edges:
        new_edge = LineString([coords[i], coords[j]])
        if not existing_lines.crosses(new_edge):
            selected_edges.append((i, j, distance))
            existing_lines = unary_union([existing_lines, new_edge])
    
    # Create a GeoDataFrame for triangulated edges
    lines = []
    start_points = []
    end_points = []
    distances = []
    for i, j, distance in selected_edges:
        lines.append(LineString([coords[i], coords[j]]))
        start_points.append(ltn_points_gdf.iloc[i]['osmid'])
        end_points.append(ltn_points_gdf.iloc[j]['osmid'])
        distances.append(distance)
    
    greedy_triangulation_gdf = gpd.GeoDataFrame({
        'geometry': lines,
        'start_osmid': start_points,
        'end_osmid': end_points,
        'distance': distances
    }, crs=ltn_points_gdf.crs)
    
    return greedy_triangulation_gdf


# get pairs of neighbourhoods to later route between
def get_ltn_node_pairs(ltn_nodes, greedy_triangulation_ltns_gdf):
    G_ltn = nx.Graph()
    
    # make networkx graph
    for _, row in ltn_nodes.iterrows():
        G_ltn.add_node(row['osmid'], geometry=row['geometry'], neighbourhood_id=row['neighbourhood_id'])
    for _, row in greedy_triangulation_ltns_gdf.iterrows():
        G_ltn.add_edge(row['start_osmid'], row['end_osmid'], distance=row['distance'], geometry=row['geometry'])
    
    # Now, for each node in the graph, find its neighbors and create node pairs
    ltn_node_pairs = []
    for node in G_ltn.nodes():
        neighbors = list(G_ltn.neighbors(node)) 
        for neighbor in neighbors:
            if node < neighbor:  # To avoid duplicates, only add pairs once (node, neighbour) where node < neighbour
                ltn_node_pairs.append((node, neighbor))
    
    return ltn_node_pairs



# get pairs of neighbourhoods to later route between
def get_node_pairs(nodes, greedy_triangulation_ltns_gdf):
    G = nx.Graph()
    
    # make networkx graph
    for _, row in nodes.iterrows():
        G.add_node(row['osmid'], geometry=row['geometry'], neighbourhood_id=row['neighbourhood_id'])
    for _, row in greedy_triangulation_ltns_gdf.iterrows():
        G.add_edge(row['start_osmid'], row['end_osmid'], distance=row['distance'], geometry=row['geometry'])
    
    # Now, for each node in the graph, find its neighbors and create node pairs
    node_pairs = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node)) 
        for neighbor in neighbors:
            if node < neighbor:  # To avoid duplicates, only add pairs once (node, neighbour) where node < neighbour
                node_pairs.append((node, neighbor))
    
    return node_pairs

# get pairs of neighbourhoods to later route between
def get_node_pairs_no_ltn(nodes, greedy_triangulation_ltns_gdf):
    G = nx.Graph()
    
    # make networkx graph
    for _, row in nodes.iterrows():
        G.add_node(row['osmid'], geometry=row['geometry'])
    for _, row in greedy_triangulation_ltns_gdf.iterrows():
        G.add_edge(row['start_osmid'], row['end_osmid'], distance=row['distance'], geometry=row['geometry'])
    
    # Now, for each node in the graph, find its neighbors and create node pairs
    node_pairs = []
    for node in G.nodes():
        neighbours = list(G.neighbors(node)) 
        for neighbour in neighbours:
            if node < neighbour:  # To avoid duplicates, only add pairs once (node, neighbour) where node < neighbour
                node_pairs.append((node, neighbour))
    
    return node_pairs



def greedy_triangulation_all(ltn_points_gdf, tess_points_gdf):
    
     # give gdf points unqiue ids
    ltn_points_gdf['id'] = range(1, len(ltn_points_gdf) + 1)
    ltn_points_gdf['source'] = 'ltn'
    tess_points_gdf['id'] = range(len(ltn_points_gdf) + 1, len(ltn_points_gdf) + 1 + len(tess_points_gdf))
    tess_points_gdf['source'] = 'tess'

    # ensure we are calculate distances in meters
    ltn_points_gdf = ltn_points_gdf.to_crs('EPSG:3857')
    tess_points_gdf = tess_points_gdf.to_crs('EPSG:3857')


    all_points_gdf = gpd.GeoDataFrame(pd.concat([ltn_points_gdf, tess_points_gdf], ignore_index=True), crs=ltn_points_gdf.crs)

    # Extract coordinates of points
    points = list(all_points_gdf.geometry)
    coords = np.array([(point.x, point.y) for point in points])
    
    # Compute all possible edges with their distances
    edges = []
    for i, j in itertools.combinations(range(len(coords)), 2):
        distance = np.linalg.norm(coords[i] - coords[j])
        edges.append((i, j, distance))
    
    # Sort edges by distance
    edges = sorted(edges, key=lambda x: x[2])  # Sort by the distance (x[2])
    
    # Initialize triangulation
    selected_edges = []
    existing_lines = GeometryCollection()  # Use an empty GeometryCollection
    
    # Iterate over edges
    for i, j, distance in edges:
        new_edge = LineString([coords[i], coords[j]])
        
        # Check if this edge intersects existing edges
        if not existing_lines.crosses(new_edge):
            selected_edges.append((i, j, distance))
            # Update existing lines by adding the new edge
            existing_lines = unary_union([existing_lines, new_edge])

    # Create a GeoDataFrame for triangulated edges with additional attributes
    lines = []
    start_osmids = []
    end_osmids = []
    distances = []
    
    for i, j, distance in selected_edges:
        lines.append(LineString([coords[i], coords[j]]))
        start_osmids.append(all_points_gdf.iloc[i]['osmid'])  # Store the index of the starting point
        end_osmids.append(all_points_gdf.iloc[j]['osmid'])    # Store the index of the ending point
        distances.append(distance)  # Store the distance for this edge
    
    # Create GeoDataFrame with attributes
    greedy_triangulation_gdf = gpd.GeoDataFrame({
        'geometry': lines,
        'start_osmid': start_osmids,
        'end_osmid': end_osmids,
        'distance': distances
    }, crs=ltn_points_gdf.crs)

    
    return greedy_triangulation_gdf, ltn_points_gdf, tess_points_gdf



def compute_total_betweenness(shortest_paths, ebc_list):
    """Calculate total betweenness centrality per shortest paths."""
    total_betweenness = {}
    for pair, edges in shortest_paths.items():
        if edges is not None:
            # Sum betweenness centrality for edges in the path
            total_betweenness[pair] = sum(
                centrality for edge, centrality in ebc_list if edge in edges
            )
        else:
            total_betweenness[pair] = None

    return total_betweenness


def compute_path_lengths(shortest_paths, graph):
    """Calculate total path lengths per shortest paths."""
    path_lengths = {}
    for (node1, node2), edges in shortest_paths.items():
        if edges is not None:  # If a path exists
            total_length = sum(graph[u][v]['distance'] for u, v in edges)
            path_lengths[(node1, node2)] = total_length
        else:  # No path
            path_lengths[(node1, node2)] = None

    return path_lengths


def get_ebc_of_shortest_paths(greedy_triangulation_all_gdf, ltn_nodes, tess_nodes, ltn_node_pairs):
    """
    Given the outputs of greedy_triangulation_all (GeoDataFrame of edges and node GeoDataFrames),
    this function returns the edge betweenness centrality (ebc) for LTN nodes and other nodes.

    Args:
    - greedy_triangulation_all_gdf: GeoDataFrame containing the edges of the graph
    - ltn_nodes: GeoDataFrame containing the LTN nodes
    - tess_nodes: GeoDataFrame containing the tessellation nodes
    - ltn_node_pairs: List of tuples of LTN node pairs to compute shortest paths between

    Returns:
    - ebc_ltn: Dictionary of edge betweenness centrality for LTN node pairs
    - ebc_other: Dictionary of edge betweenness centrality for all other node pairs
    """
    # Create the graph from the triangulation GeoDataFrame
    GT_abstract = nx.Graph()
    for _, row in greedy_triangulation_all_gdf.iterrows():
        start = row['start_osmid']
        end = row['end_osmid']
        distance = row['sp_lts_distance']
        GT_abstract.add_edge(start, end, geometry=row['geometry'], distance=distance)

    # Add node attributes
    combined_gdf = pd.concat([ltn_nodes, tess_nodes], ignore_index=True)
    attributes = combined_gdf.set_index('id').to_dict('index')
    nx.set_node_attributes(GT_abstract, attributes)

    # Calculate edge betweenness centrality (ebc)
    ebc = nx.edge_betweenness_centrality(GT_abstract, weight= 'sp_lts_distance', normalized=True)
    ebc_list = [(edge, centrality) for edge, centrality in ebc.items()]

    # Separate LTN and non-LTN nodes
    ltn_node_ids = set(ltn_nodes['id'])
    all_node_ids = set(GT_abstract.nodes)
    non_ltn_node_ids = all_node_ids - ltn_node_ids

    # Shortest paths between LTN nodes
    shortest_paths_ltn = {}
    for node1, node2 in ltn_node_pairs:
        path = nx.shortest_path(GT_abstract, source=node1, target=node2)
        edges = list(nx.utils.pairwise(path))
        shortest_paths_ltn[(node1, node2)] = edges
    

    # Shortest paths between all other node combinations
    shortest_paths_other = {}
    for node1, node2 in itertools.combinations(all_node_ids, 2):
        if (node1, node2) not in shortest_paths_ltn:  # Avoid recomputing LTN-LTN pairs
            path = nx.shortest_path(GT_abstract, source=node1, target=node2)
            edges = list(nx.utils.pairwise(path))
            shortest_paths_other[(node1, node2)] = edges

    # Compute path lengths
    path_lengths_ltn = compute_path_lengths(shortest_paths_ltn, GT_abstract)
    path_lengths_other = compute_path_lengths(shortest_paths_other, GT_abstract)

    # Compute total betweenness centrality for each group of shortest paths
    ebc_ltn = compute_total_betweenness(shortest_paths_ltn, ebc_list)
    ebc_other = compute_total_betweenness(shortest_paths_other, ebc_list)

    # Sort by betweenness centrality
    ebc_ltn = dict(sorted(ebc_ltn.items(), key=lambda item: item[1]))
    ebc_other = dict(sorted(ebc_other.items(), key=lambda item: item[1]))

    return ebc_ltn, ebc_other, shortest_paths_ltn, shortest_paths_other




def make_sp_dict(paths_list):
    """
    Function to convert a list of paths into a dictionary of shortest paths.
    """
    sp = {}
    for p in paths_list:
        if len(p) < 2:
            continue
        u, v = p[0], p[-1]
        sp[(u, v)] = p
        sp[(v, u)] = list(reversed(p))
    return sp

    # Helper to convert node-path list into edge-path list dict
def make_sp_edge_dict(paths_list):
    """
    Function to convert a list of paths into a dictionary of shortest paths,
    """
    sp = {}
    for p in paths_list:
        if len(p) < 2:
            continue
        # build list of edge tuples
        edge_list = [(p[i], p[i+1]) for i in range(len(p)-1)]
        u, v = p[0], p[-1]
        sp[(u, v)] = edge_list
        sp[(v, u)] = [(v2, v1) for (v1, v2) in reversed(edge_list)]
    return sp




def adjust_triangulation_to_budget_ltn_priority(triangulation_gdf, D, shortest_paths_ltn, ebc_ltn, shortest_paths_other, ebc_other, previous_selected_edges=None, ltn_node_pairs=None):
    """
    Adjust a given triangulation to fit within the specified budget D,
    ensuring that previously selected edges are always included.
    Only after all ltns are connected do we move to include the growth of other areas.
    """
    # make a graph
    G = nx.Graph()
    for _, row in triangulation_gdf.iterrows():
        G.add_edge(
            row['start_osmid'],
            row['end_osmid'],
            geometry=row['geometry'],
            distance=row['distance'],
            sp_true_distance=row['sp_true_distance'],
            sp_lts_distance=row['sp_lts_distance'],
            eucl_dist = row['eucl_dist']
        )

    total_length = 0
    selected_edges = set(tuple(sorted(edge)) for edge in (previous_selected_edges or [])) # use tuple to ensure we don't double count edges

    # Include previously selected edges so that we aren't starting from stratch each loop through
    for u, v in selected_edges:
        if G.has_edge(u, v):
            total_length += G[u][v]['distance']

    # Track the ltns which are connected
    connected_ltn_pairs = set()

    # Track all other connected pairs
    connected_other_pairs = set()

    # Prune for ltn node pairs first
    for (node1, node2), centrality in ebc_ltn.items():
        if node1 in G.nodes and node2 in G.nodes:
            edges = shortest_paths_ltn.get((node1, node2), [])
            if edges:  # If a valid path exists
                # Calculate new edges and their length
                new_edges = [tuple(sorted((u, v))) for u, v in edges if tuple(sorted((u, v))) not in selected_edges]
                new_length = sum(G[min(u, v)][max(u, v)]['distance'] for u, v in new_edges)
                # Check if adding this path exceeds the budget D
                if total_length + new_length > D:
                    continue
                # Add the edges to selected_edges
                selected_edges.update(new_edges)
                total_length += new_length
                connected_ltn_pairs.add((node1, node2))

    
    # Check if all ltn node pairs are connected
    if set(ltn_node_pairs).issubset(connected_ltn_pairs):
        # Now move to all other connections (ltn to tess, tess to tess, tess to ltn etc)
        for (node1, node2), centrality in ebc_other.items():
            if node1 in G.nodes and node2 in G.nodes:
                edges = shortest_paths_other.get((node1, node2), [])
                if edges:  # If a valid path exists
                    new_edges = [tuple(sorted((u, v))) for u, v in edges if tuple(sorted((u, v))) not in selected_edges]
                    new_length = sum(G[min(u, v)][max(u, v)]['distance'] for u, v in new_edges)
                    # Check if adding this path exceeds the budget D
                    if total_length + new_length > D:
                        continue
                    # Add the edges to selected_edges
                    selected_edges.update(new_edges)
                    total_length += new_length
                    connected_other_pairs.add((node1, node2))
    # missing_pairs = [pair for pair in ltn_node_pairs if pair not in connected_ltn_pairs]
        
    # edges which aren't in a shortest path won't have been selected
        # we will add these last, as they are the least important
        unused_edges = []
        for _, row in triangulation_gdf.iterrows():
            e = tuple(sorted((row['start_osmid'], row['end_osmid'])))
            dist = row['distance']
            if e not in selected_edges:
                unused_edges.append((dist, e))
        unused_edges.sort(key=lambda x: x[0])
        for dist, edge in unused_edges:
            if total_length + dist <= D:
                selected_edges.add(edge)
                total_length += dist
            else:
                break
    


    # Build the adjusted GeoDataFrame
    lines = []
    distances = []
    start_osmids = []
    end_osmids = []
    sp_true_distance=[]
    sp_lts_distance=[]
    eucl_dist = []


    for u, v in selected_edges:
        lines.append(G[u][v]['geometry'])
        distances.append(G[u][v]['distance'])
        start_osmids.append(u)
        end_osmids.append(v)
        sp_true_distance.append(G[u][v]['sp_true_distance'])
        sp_lts_distance.append(G[u][v]['sp_lts_distance'])
        eucl_dist.append(G[u][v]['eucl_dist'])

    adjusted_gdf = gpd.GeoDataFrame({
        'geometry': lines,
        'start_osmid': start_osmids,
        'end_osmid': end_osmids,
        'distance': distances,
        'sp_true_distance': sp_true_distance,
        'sp_lts_distance': sp_lts_distance,
        'eucl_dist': eucl_dist
    }, crs=triangulation_gdf.crs)

    return adjusted_gdf, selected_edges, connected_ltn_pairs, connected_other_pairs






def adjust_triangulation_to_budget(triangulation_gdf, D, shortest_paths_all, ebc_all, previous_selected_edges=None):
    """
    Adjust a given triangulation to fit within the specified budget D,
    ensuring that previously selected edges are always included.
    """

    # Build the graph from triangulation
    G = nx.Graph()
    for _, row in triangulation_gdf.iterrows():
        G.add_edge(
            row['start_osmid'],
            row['end_osmid'],
            geometry=row['geometry'],
            distance=row['distance'],
            sp_true_distance=row['sp_true_distance'],
            sp_lts_distance=row['sp_lts_distance'],
            eucl_dist=row['eucl_dist']
        )

    total_length = 0
    selected_edges = set(tuple(sorted(edge)) for edge in (previous_selected_edges or []))  # ensure consistent ordering

    # Include previously selected edges
    for u, v in selected_edges:
        if G.has_edge(u, v):
            total_length += G[u][v]['distance']

    # Track connected node pairs
    connected_pairs = set()

    # Add node pairs based on centrality
    for (node1, node2), centrality in sorted(ebc_all.items(), key=lambda x: x[1], reverse=True):
        if node1 in G.nodes and node2 in G.nodes:
            edges = shortest_paths_all.get((node1, node2), [])
            if edges:
                new_edges = [tuple(sorted((u, v))) for u, v in edges if tuple(sorted((u, v))) not in selected_edges]
                new_length = sum(G[u][v]['distance'] for u, v in new_edges)
                if total_length + new_length > D:
                    continue
                selected_edges.update(new_edges)
                total_length += new_length
                connected_pairs.add((node1, node2))

    # Add unused edges (shouldn't be needed, but you never know...)
    unused_edges = []
    for _, row in triangulation_gdf.iterrows():
        e = tuple(sorted((row['start_osmid'], row['end_osmid'])))
        if e not in selected_edges:
            unused_edges.append((row['distance'], e))
    unused_edges.sort(key=lambda x: x[0])

    for dist, edge in unused_edges:
        if total_length + dist <= D:
            selected_edges.add(edge)
            total_length += dist
        else:
            break

    
    lines = []
    distances = []
    start_osmids = []
    end_osmids = []
    sp_true_distance = []
    sp_lts_distance = []
    eucl_dist = []

    for u, v in selected_edges:
        lines.append(G[u][v]['geometry'])
        distances.append(G[u][v]['distance'])
        start_osmids.append(u)
        end_osmids.append(v)
        sp_true_distance.append(G[u][v]['sp_true_distance'])
        sp_lts_distance.append(G[u][v]['sp_lts_distance'])
        eucl_dist.append(G[u][v]['eucl_dist'])

    adjusted_gdf = gpd.GeoDataFrame({
        'geometry': lines,
        'start_osmid': start_osmids,
        'end_osmid': end_osmids,
        'distance': distances,
        'sp_true_distance': sp_true_distance,
        'sp_lts_distance': sp_lts_distance,
        'eucl_dist': eucl_dist
    }, crs=triangulation_gdf.crs)

    return adjusted_gdf, selected_edges, connected_pairs




def build_greedy_triangulation_no_ltns(tess_points_gdf):
    """ 
    Perform a greedy triangulation using only tessellation points
    """
    # Assign unique IDs and source label to tess points
    tess_points_gdf = tess_points_gdf.copy()
    tess_points_gdf['id'] = range(1, len(tess_points_gdf) + 1)
    tess_points_gdf['source'] = 'tess'

    # Ensure distances are in meters
    tess_points_gdf = tess_points_gdf.to_crs('EPSG:3857')

    # Extract coordinates
    points = list(tess_points_gdf.geometry)
    coords = np.array([(point.x, point.y) for point in points])
    
    # Compute all possible edges
    edges = []
    for i, j in itertools.combinations(range(len(coords)), 2):
        distance = np.linalg.norm(coords[i] - coords[j])
        edges.append((i, j, distance))
    
    # Sort edges by distance
    edges = sorted(edges, key=lambda x: x[2])
    
    # Initialize triangulation
    selected_edges = []
    existing_lines = GeometryCollection()
    
    # Iterate over edges and add those that don't intersect existing edges
    for i, j, distance in edges:
        new_edge = LineString([coords[i], coords[j]])
        if not existing_lines.crosses(new_edge):
            selected_edges.append((i, j, distance))
            existing_lines = unary_union([existing_lines, new_edge])

    # Convert back to lat/lon for accurate geodesic distance calculation
    tess_points_gdf = tess_points_gdf.to_crs('EPSG:4326')
    node_distances = {}
    for (idx1, row1), (idx2, row2) in itertools.combinations(tess_points_gdf.iterrows(), 2):
        coord1 = (row1.geometry.y, row1.geometry.x)
        coord2 = (row2.geometry.y, row2.geometry.x)
        distance = geodesic(coord1, coord2).meters
        node_distances[(idx1, idx2)] = distance

    # Build GeoDataFrame for edges
    lines = []
    start_osmids = []
    end_osmids = []
    distances = []

    for i, j, _ in selected_edges:
        lines.append(LineString([coords[i], coords[j]]))
        start_osmids.append(tess_points_gdf.iloc[i]['osmid'])
        end_osmids.append(tess_points_gdf.iloc[j]['osmid'])
        distance = node_distances[(i, j)]
        distances.append(distance)

    greedy_triangulation_gdf = gpd.GeoDataFrame({
        'geometry': lines,
        'start_osmid': start_osmids,
        'end_osmid': end_osmids,
        'distance': distances
    }, crs=3857)

    return greedy_triangulation_gdf, tess_points_gdf


def build_greedy_triangulation(ltn_points_gdf, tess_points_gdf):
    """
    Perform a greedy triangulation using LTNs and another set of points
    """
    # Assign unique IDs to the points
    ltn_points_gdf['id'] = range(1, len(ltn_points_gdf) + 1)
    ltn_points_gdf['source'] = 'ltn'
    tess_points_gdf['id'] = range(len(ltn_points_gdf) + 1, len(ltn_points_gdf) + 1 + len(tess_points_gdf))
    tess_points_gdf['source'] = 'tess'

    # Ensure we are calculating distances in meters
    ltn_points_gdf = ltn_points_gdf.to_crs('EPSG:3857')
    tess_points_gdf = tess_points_gdf.to_crs('EPSG:3857')

    all_points_gdf = gpd.GeoDataFrame(pd.concat([ltn_points_gdf, tess_points_gdf], ignore_index=True), crs=ltn_points_gdf.crs)

    # Extract coordinates of points
    points = list(all_points_gdf.geometry)
    coords = np.array([(point.x, point.y) for point in points])
    
    # Compute all possible edges 
    edges = []
    for i, j in itertools.combinations(range(len(coords)), 2):
        distance = np.linalg.norm(coords[i] - coords[j]) # this distance is not accurate, see later on 
        edges.append((i, j, distance))
    
    # Sort edges by distance
    edges = sorted(edges, key=lambda x: x[2])  # Sort by the distance (x[2])
    
    # Initialize triangulation
    selected_edges = []
    existing_lines = GeometryCollection()  # Use an empty GeometryCollection

    # Iterate over edges
    for i, j, distance in edges:
        new_edge = LineString([coords[i], coords[j]])
        
        # Check if this edge intersects existing edges
        if not existing_lines.crosses(new_edge):
            selected_edges.append((i, j, distance))
            # Update existing lines by adding the new edge
            existing_lines = unary_union([existing_lines, new_edge])

    ## get better distance measurements
    all_points_gdf = all_points_gdf.to_crs('EPSG:4326')
    node_distances = {}
    for (idx1, row1), (idx2, row2) in itertools.combinations(all_points_gdf.iterrows(), 2):
        coord1 = (row1.geometry.y, row1.geometry.x)  # (lat, lon)
        coord2 = (row2.geometry.y, row2.geometry.x)  # (lat, lon)
        # Compute geodesic distance
        distance = geodesic(coord1, coord2).meters  # or .kilometers
        node_distances[(idx1, idx2)] = distance


    # Create a GeoDataFrame for triangulated edges with additional attributes
    lines = []
    start_osmids = []
    end_osmids = []
    distances = []
    
    for i, j, distance in selected_edges:
        lines.append(LineString([coords[i], coords[j]]))
        start_osmids.append(all_points_gdf.iloc[i]['osmid'])  # Store the index of the starting point
        end_osmids.append(all_points_gdf.iloc[j]['osmid'])    # Store the index of the ending point
        distance = node_distances[(i,j)]
        distances.append(distance)  # Store the distance for this edge
    
    # Create GeoDataFrame with attributes
    greedy_triangulation_gdf = gpd.GeoDataFrame({
        'geometry': lines,
        'start_osmid': start_osmids,
        'end_osmid': end_osmids,
        'distance': distances
    }, crs=ltn_points_gdf.crs)

    return greedy_triangulation_gdf, ltn_points_gdf, tess_points_gdf



def get_sp_demand_weights(node_pairs, shortest_paths, graph, edge_length_key='sp_lts_distance'):
    """
    Calculates the weighted demand (based on edge total flow and length) for each path from the dutch PCT scenario.
    Similar structure to get_sp_ebc_weights.
    
    :param node_pairs: List of tuples (start_node, end_node).
    :param shortest_paths: List of paths (each a list of nodes along the path).
    :param graph: NetworkX graph with 'total_flow' and length attributes.
    :param edge_length_key: Key for length in edge attributes (default: 'sp_lts_distance').
    :return: Dict mapping (start, end) node pairs to demand-based weight.
    """
    result_dict = {}

    for i, (start_node, end_node) in enumerate(node_pairs):
        path = shortest_paths[i]

        # Turn node path into edge pairs
        edges_in_path = [(path[j], path[j + 1]) for j in range(len(path) - 1)]

        edges_info = []
        for u, v in edges_in_path:
            edge_data = graph.get_edge_data(u, v) or graph.get_edge_data(v, u)
            if edge_data is None:
                raise ValueError(f"Edge ({u}, {v}) not found in graph for path {path}")

            length = edge_data.get(edge_length_key, 0)
            flow = edge_data.get('total_flow', 0)
            edges_info.append((length, flow))

        total_length = sum(length for length, _ in edges_info)
        num_edges = len(edges_info)

        if num_edges == 0:
            result_dict[(start_node, end_node)] = 0.0
            continue

        # Weighted demand: (flow * relative length) averaged across edges
        weighted_sum = sum(flow * (length / total_length) for length, flow in edges_info) if total_length > 0 else 0.0
        result_dict[(start_node, end_node)] = weighted_sum / num_edges

    return result_dict


def get_sp_ebc_weights(node_pairs, shortest_paths, graph, edge_betweenness, edge_length_key='sp_lts_distance'):
    """
    
    :param node_pairs: List of tuples (start_node, end_node) representing node pairs.
    :param shortest_paths: List of lists, where each inner list contains nodes in the shortest path for the corresponding node pair.
    :param graph: NetworkX graph object or similar structure with an edge data retrieval method.
    :param edge_betweenness: Dictionary with edge tuples as keys and betweenness values as values.
    :param edge_length_key: Key used to access edge length from the graph's edge attributes (default: 'sp_lts_distance').
    :return: Dictionary mapping node pairs to their computed metric values.

    here we are attempting to calculate the importance of each connection in the network along the path between two points
    for this we use sum of { (ebc of edge * (length of edge/sum of length of edges) ) } / number-of-edges, per path
    """
    result_dict = {}
    
    for i, (start_node, end_node) in enumerate(node_pairs):
        path = shortest_paths[i]
        
        # Extract edges in the path (consecutive node pairs)
        edges_in_path = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
        
        # Collect edge lengths and betweenness values
        edges_info = []
        for u, v in edges_in_path:
            edge_data = graph.get_edge_data(u, v)
            if not edge_data:
                raise ValueError(f"Edge ({u}, {v}) not found in graph for path {path}")
            
            length = edge_data.get(edge_length_key, 0)
            ebc = edge_betweenness.get((u, v), edge_betweenness.get((v, u)))
            
            if ebc is None:
                raise KeyError(f"Edge betweenness not found for edge ({u}, {v}) or ({v}, {u})")
            
            edges_info.append((length, ebc))
        
        # Calculate total length of edges in the path
        total_length = sum(length for length, _ in edges_info)
        num_edges = len(edges_info)
        
        if num_edges == 0:
            result_dict[(start_node, end_node)] = 0.0
            continue
        
        weighted_sum = sum(ebc * (length / total_length) for length, ebc in edges_info) if total_length > 0 else 0.0
        
        # Store the final metric value
        result_dict[(start_node, end_node)] = weighted_sum / num_edges
    
    return result_dict





def gdf_to_nx_graph(gdf, target_crs="EPSG:4326"):
    """
    Converts a GeoDataFrame with start_osmid, end_osmid, and edge attributes 
    to a NetworkX MultiDiGraph and adds 'x' and 'y' coordinates as longitude and latitude.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing edges with 'start_osmid', 'end_osmid', 
                        and other edge attributes (e.g., 'geometry', 'weight', 'betweeness').
    geo_crs (str): The CRS to transform to for longitude and latitude (default is "EPSG:4326").

    Returns:
    NetworkX MultiDiGraph: A NetworkX graph with nodes having 'x' (longitude), 'y' (latitude), 
                           'x_original', and 'y_original' attributes, and edges with 
                           'geometry', 'weight', and 'betweeness' attributes.
    """
    GT_abstract_nx = nx.MultiDiGraph()

    # Detect the CRS of the input GeoDataFrame
    original_crs = gdf.crs
    if original_crs is None:
        raise ValueError("Input GeoDataFrame does not have a CRS defined.")

    # Create a transformer to convert from the original CRS to geographic CRS
    transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)

    # Add edges and ensure nodes have 'x', 'y', 'x_original', and 'y_original' attributes
    for _, row in gdf.iterrows():
        u = row['start_osmid']
        v = row['end_osmid']
        
        # Extract start and end coordinates from the geometry
        start_coords_original = row['geometry'].coords[0]
        end_coords_original = row['geometry'].coords[-1]
        
        # Transform coordinates to longitude and latitude
        start_coords_geo = transformer.transform(*start_coords_original)
        end_coords_geo = transformer.transform(*end_coords_original)
        
        # Add or update the nodes with both original and geographic coordinates
        if not GT_abstract_nx.has_node(u):
            GT_abstract_nx.add_node(u, 
                                    x=start_coords_geo[0], y=start_coords_geo[1],  # Longitude, Latitude
                                    x_original=start_coords_original[0], y_original=start_coords_original[1])  # Original CRS
        if not GT_abstract_nx.has_node(v):
            GT_abstract_nx.add_node(v, 
                                    x=end_coords_geo[0], y=end_coords_geo[1],  # Longitude, Latitude
                                    x_original=end_coords_original[0], y_original=end_coords_original[1])  # Original CRS
        
        # Add edge with attributes
        data = {
            'geometry': row['geometry'],
            'weight': row['distance'],
            'sp_true_distance': row['sp_true_distance'],   
            'sp_lts_distance': row['sp_lts_distance'],
            'eucl_dist' : row['eucl_dist'],
        }
        GT_abstract_nx.add_edge(u, v, **data)

    return GT_abstract_nx





def deweight_edges(GT, tag_lts):
    """
    Undo the edge weighting by dividing the 'length' attribute by the LTS value.
    Operates only on the GT subgraph (not the original G_weighted graph).
    The division is based on the 'highway' attribute using the LTS values in `tag_lts`.
    """
    for u, v, key, data in GT.edges(keys=True, data=True):
        highway = data.get('highway')  # Get the highway type
        if highway:
            lts = tag_lts.get(highway, 1)  # Get LTS value for the highway, defaulting to 1
            if lts > 0:  # Ensure valid LTS value (non-zero)
                data['length'] /= lts  # Divide length by the LTS value to undo the modification



def snap_to_largest_stroke(point, snapthreshold, stroke_gdf):
    if not isinstance(point, Point):  # Ensure it's a valid point
        return point  

    # set to meters
    stroke_gdf = stroke_gdf.to_crs(epsg=3857)
    
    # Find strokes within snapthreshold distance
    stroke_gdf["distance"] = stroke_gdf.geometry.distance(point)
    nearby_strokes = stroke_gdf[stroke_gdf["distance"] <= snapthreshold]

    if nearby_strokes.empty:
        print("No street found for point:", point)
        return point  # No match found, return original point

    # Select the stroke with the highest n_segments
    largest_stroke = nearby_strokes.loc[nearby_strokes["n_segments"].idxmax()]

    # Snap to the nearest point on this stroke
    nearest_point = nearest_points(point, largest_stroke.geometry)[1]
    
    return nearest_point





def compute_routed_distance_for_GT(row, G):
    """
    Compute the routed_distance for a given row of greedy_gdf using graph G.
    
    Depending on the ltn_origin and ltn_destination flags:
      - If both are True: try all combinations of exit points for origin and destination.
      - If one is True: try all exit points for that endpoint and the single non-LTN endpoint.
      - Otherwise: compute the direct shortest path between start_osmid and end_osmid.
      
    Returns the minimum route length found or np.nan if no route is found.
    """
    # Shortcut variables
    origin = row['start_osmid']
    dest = row['end_osmid']
    
    # Case 1: Both endpoints are in an LTN.
    if row['ltn_origin'] and row['ltn_destination']:
        try:
            origin_neigh = osmid_to_neigh[origin]
            dest_neigh = osmid_to_neigh[dest]
        except KeyError:
            # One of the osmids is missing in the centroids mapping
            return np.nan

        origin_exits = neigh_to_exits.get(origin_neigh, [])
        dest_exits = neigh_to_exits.get(dest_neigh, [])
        
        distances = []
        for o_exit in origin_exits:
            for d_exit in dest_exits:
                try:
                    d_val = nx.shortest_path_length(G, source=o_exit, target=d_exit, weight="length")
                    distances.append(d_val)
                except nx.NetworkXNoPath:
                    continue  # Skip if no path exists for this pair
        return min(distances) if distances else np.nan

    # Case 2: Only the origin is in an LTN.
    elif row['ltn_origin'] and not row['ltn_destination']:
        try:
            origin_neigh = osmid_to_neigh[origin]
        except KeyError:
            return np.nan

        origin_exits = neigh_to_exits.get(origin_neigh, [])
        distances = []
        for o_exit in origin_exits:
            try:
                d_val = nx.shortest_path_length(G, source=o_exit, target=dest, weight="length")
                distances.append(d_val)
            except nx.NetworkXNoPath:
                continue
        return min(distances) if distances else np.nan

    # Case 3: Only the destination is in an LTN.
    elif not row['ltn_origin'] and row['ltn_destination']:
        try:
            dest_neigh = osmid_to_neigh[dest]
        except KeyError:
            return np.nan

        dest_exits = neigh_to_exits.get(dest_neigh, [])
        distances = []
        for d_exit in dest_exits:
            try:
                d_val = nx.shortest_path_length(G, source=origin, target=d_exit, weight="length")
                distances.append(d_val)
            except nx.NetworkXNoPath:
                continue
        return min(distances) if distances else np.nan

    # Case 4: Neither endpoint is in an LTN.
    else:
        try:
            return nx.shortest_path_length(G, source=origin, target=dest, weight="length")
        except nx.NetworkXNoPath:
            return np.nan
        


def compute_routed_path_for_GT(row, G):
    """
    Compute the shortest path for a given row of greedy_gdf using graph G.
    
    Depending on the ltn_origin and ltn_destination flags:
      - If both are True: try all combinations of exit points for origin and destination.
      - If one is True: try all exit points for that endpoint and the single non-LTN endpoint.
      - Otherwise: compute the direct shortest path between start_osmid and end_osmid.
      
    Returns the shortest path (list of nodes) found or np.nan if no path exists.
    """
    origin = row['start_osmid']
    dest = row['end_osmid']
    
    min_path = None
    min_length = float('inf')
    
    # Case 1: Both endpoints are in an LTN
    if row['ltn_origin'] and row['ltn_destination']:
        try:
            origin_neigh = osmid_to_neigh[origin]
            dest_neigh = osmid_to_neigh[dest]
        except KeyError:
            return np.nan

        origin_exits = neigh_to_exits.get(origin_neigh, [])
        dest_exits = neigh_to_exits.get(dest_neigh, [])
        
        for o_exit in origin_exits:
            for d_exit in dest_exits:
                try:
                    # Calculate path length first
                    length = nx.shortest_path_length(G, o_exit, d_exit, weight='length')
                    if length < min_length:
                        # Update with the actual path
                        path = nx.shortest_path(G, o_exit, d_exit, weight='length')
                        min_length = length
                        min_path = path
                except nx.NetworkXNoPath:
                    continue
        return min_path if min_path is not None else np.nan

    # Case 2: Only origin is in an LTN
    elif row['ltn_origin'] and not row['ltn_destination']:
        try:
            origin_neigh = osmid_to_neigh[origin]
        except KeyError:
            return np.nan

        origin_exits = neigh_to_exits.get(origin_neigh, [])
        
        for o_exit in origin_exits:
            try:
                length = nx.shortest_path_length(G, o_exit, dest, weight='length')
                if length < min_length:
                    path = nx.shortest_path(G, o_exit, dest, weight='length')
                    min_length = length
                    min_path = path
            except nx.NetworkXNoPath:
                continue
        return min_path if min_path is not None else np.nan

    # Case 3: Only destination is in an LTN
    elif not row['ltn_origin'] and row['ltn_destination']:
        try:
            dest_neigh = osmid_to_neigh[dest]
        except KeyError:
            return np.nan

        dest_exits = neigh_to_exits.get(dest_neigh, [])
        
        for d_exit in dest_exits:
            try:
                length = nx.shortest_path_length(G, origin, d_exit, weight='length')
                if length < min_length:
                    path = nx.shortest_path(G, origin, d_exit, weight='length')
                    min_length = length
                    min_path = path
            except nx.NetworkXNoPath:
                continue
        return min_path if min_path is not None else np.nan

    # Case 4: Neither endpoint is in an LTN
    else:
        try:
            return nx.shortest_path(G, origin, dest, weight='length')
        except nx.NetworkXNoPath:
            return np.nan
        



def calculate_sp_route_distance(route, G):
    """
    Find the total distance of a shortest path route in a given graph.
    If there are multiple edges between two nodes, choose the one with the minimum 'length'.
    """
    # Check if route is valid (i.e. it's a list and not np.nan)
    if not isinstance(route, list):
        return np.nan

    total_length = 0
    # Iterate through consecutive pairs of nodes in the route.
    for u, v in zip(route[:-1], route[1:]):
        # Get the edge data between u and v.
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            print("No edge")
            # If no edge exists between these nodes, return np.nan
            return np.nan
        
        # In a MultiGraph, edge_data is a dict of dicts.
        # We choose the edge with the smallest 'length' attribute.
        # Each edge's attributes are stored in the inner dictionaries.
        try:
            edge_attrs = list(edge_data.values())
            min_edge_length = min(attr.get('length', 0) for attr in edge_attrs)
        except Exception as e:
            print(f"Error processing edge from {u} to {v}: {e}")
            return np.nan

        total_length += min_edge_length

    return total_length

def add_lsoa_population(lsoa_bound):
    """
    Add population data to LSOA GeoDataFrame using UKCensusAPI.
    
    Args:
        lsoa_bound (gpd.GeoDataFrame): Input GeoDataFrame with LSOA codes in 'geo_code'
        
    Returns:
        gpd.GeoDataFrame: Original GeoDataFrame with new 'pop' column added
    """
    api = Api.Nomisweb(PATH["data"] + placeid)
    
    # Extract LSOA codes and LAD names from input data
    lsoa_codes = lsoa_bound['geo_code'].tolist()
    lad_names = lsoa_bound['lad_name'].unique().tolist()
    
    # Configure census query
    query_params = {
        "CELL": "0",
        "date": "latest",
        "RURAL_URBAN": "0",
        "select": "GEOGRAPHY_CODE,OBS_VALUE",
        "MEASURES": "20100",
        "geography": api.get_geo_codes(api.get_lad_codes(lad_names), 
        Api.Nomisweb.GeoCodeLookup["LSOA11"]) #use 2011 boundaries to match with PCT data
    }
    
    # Get and filter population data
    population = api.get_data("KS101EW", query_params) # population table
    population = population[population.GEOGRAPHY_CODE.isin(lsoa_codes)]
    
    # Merge with our lsoa dataframe
    return lsoa_bound.merge(
        population[['GEOGRAPHY_CODE', 'OBS_VALUE']],
        left_on='geo_code',
        right_on='GEOGRAPHY_CODE',
        how='left'
    ).drop(columns=['GEOGRAPHY_CODE']).rename(columns={'OBS_VALUE': 'pop'})



def get_longest_connected_components(G):
    """Returns the longest connected components of a graph."""
    components = nx.weakly_connected_components(G)
    max_length = 0
    max_component = None
    for comp in components:
        subgraph = G.subgraph(comp)
        length = sum(data['length'] for _, _, data in subgraph.edges(data=True))
        if length > max_length:
            max_length = length
            max_component = comp

    return sum(data['length'] for _, _, data in G.subgraph(max_component).edges(data=True)) if max_component else 0



def get_building_populations(lsoa_bound, boundary):
    # get buildings for the study area and assign population to them to better obtain who lives within a short distance of the cycle network
    
    buildings = ox.features.features_from_polygon(
        boundary.unary_union,
        tags={'building': True}
    )
    buildings = buildings[['building', 'geometry']].reset_index(drop=True)

    # drop non-residential buildings
    building_categories = ['apartments', 'terrace', 'residential', 'hall_of_residence',
        'dormitory', 'tower', 'house', 'shelter', 'semidetached_house',
        'bungalow', 'detached', 'cabin', 'yes', 'barracks', 'annexe',
        'farm', 'houseboat', 'static_caravan'] # buliding types to keep

    buildings = buildings[
        buildings['building'].str.lower().isin([c.lower() for c in building_categories])
    ].copy().reset_index(drop=True)

    # Keep only rows where the geometry is a Polygon or MultiPolygon (some buildings are tagged as points)
    buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])].reset_index(drop=True)
    buildings = buildings.to_crs(lsoa_bound.crs)
    buildings = gpd.overlay(buildings, lsoa_bound, how='intersection')
    buildings = buildings[['building', 'geometry']]

    buildings = buildings.to_crs(epsg=3857) # to work in meters
    buildings['area'] = buildings['geometry'].area # find area
    if debug:
        print(buildings['building'].unique())
    buildings = buildings.to_crs(4326)
    buildings = buildings.to_crs(lsoa_bound.crs)
    buildings_joined = gpd.sjoin(buildings, lsoa_bound[['pop', 'geometry']], how='left', predicate='intersects')

    # Compute the total building area per LSOA.
    total_area = buildings_joined.groupby('index_right')['area'].sum().reset_index().rename(columns={'area': 'total_area'})
    buildings_joined = buildings_joined.merge(total_area, on='index_right')
    buildings_joined['pop_exact'] = (buildings_joined['area'] / buildings_joined['total_area']) * buildings_joined['pop']
    def assign_pop(group): # assign population to buildings, but avoid decimal numbers as you can't have 0.7 of a person!
        group['pop_floor'] = np.floor(group['pop_exact'])
        group['pop_frac'] = group['pop_exact'] - group['pop_floor']
        remainder = int(group['pop'].iloc[0] - group['pop_floor'].sum())
        group['pop_assigned'] = group['pop_floor']
        if remainder > 0:
            group.loc[group['pop_frac'].nlargest(remainder).index, 'pop_assigned'] += 1
        return group.astype({'pop_assigned': 'int'})
    buildings = buildings_joined.groupby('index_right', group_keys=False).apply(assign_pop)
    if debug:
        buildings.plot(column='pop_assigned')
    buildings = buildings[['building', 'geometry', 'pop_assigned']].reset_index(drop=True)

    return buildings


def find_bounding_lines(centroids, lines_gdf):
    """
    centroids: GeoSeries of points from the center of slivers
    lines_gdf: greedy triangulation
    
    Returns a dict mapping each centroid‐index to a list of three line‐indices.
    """
    # keep (idx, geom) tuples so we can recover original indices
    lines_idx_geom = list(lines_gdf.geometry.items())  
    raw_lines      = [geom for _, geom in lines_idx_geom]
    triangles = list(polygonize(raw_lines)) # build polygons from greedy triangulation
    result = {}
    for cent_idx, pt in centroids.items():
        # find the triangle containing this point
        tri = next((t for t in triangles if t.contains(pt)), None)
        if tri is None:
            result[cent_idx] = []
            continue
        # extract its three edges
        coords = list(tri.exterior.coords)  # last==first
        tri_edges = [LineString([coords[j], coords[j+1]]) for j in range(3)]
        # for each edge, find the matching original line‐index
        bound_idxs = []
        for edge in tri_edges:
            rev = LineString(edge.coords[::-1])
            match = next(
                (idx for idx, geom in lines_idx_geom 
                 if geom.equals(edge) or geom.equals(rev)),
                None)
            if match is not None:
                bound_idxs.append(match)
        result[cent_idx] = bound_idxs
    return result


def run_random_growth(placeid, poi_source, investment_levels, weighting, greedy_gdf, G_caralls, G_weighted, all_centroids, exit_points, sp_length, sp_path, ltn_gdf, tess_gdf, debug=False):
    '''Creates a bike network through random order, given a list of input edges and a budget. Used to create many random runs.'''
    shuffled_edges = greedy_gdf.sample(frac=1)  # no fixed seed for variability
    random_edges = pd.Series(False, index=greedy_gdf.index)
    distance = 0.0
    edge_pointer = 0

    global_processed_pairs_random = set()
    cumulative_GT_indices_random = set()
    Random_GT_abstracts = []
    Random_GT_abstracts_gdf = []
    Random_GTs = []
    Random_GTs_gdf = []
    
    for D in tqdm(investment_levels, desc="Pruning GT abstract randomly and routing on network for meters of investment"):
        # Calculate remaining budget for new edges
        remaining_budget = D - distance
        if remaining_budget > 0 and edge_pointer < len(shuffled_edges):
            cumulative_distances = shuffled_edges.iloc[edge_pointer:]['distance'].cumsum()
            within_budget = cumulative_distances <= remaining_budget
            num_edges_to_add = within_budget.sum()
            if num_edges_to_add > 0:
                new_idx = shuffled_edges.iloc[edge_pointer:edge_pointer + num_edges_to_add].index
                random_edges[new_idx] = True
                distance += cumulative_distances.iloc[num_edges_to_add - 1]
                edge_pointer += num_edges_to_add

        # Save abstract graph
        GT_abstract_gdf = greedy_gdf[random_edges].copy()
        if GT_abstract_gdf.empty:
            print(f"[run_random_growth] No edges selected at investment level {D}. Skipping.")
            Random_GT_abstracts_gdf.append(GT_abstract_gdf)
            Random_GT_abstracts.append(nx.Graph())
            Random_GTs.append(nx.Graph())
            Random_GTs_gdf.append(gpd.GeoDataFrame())
            continue
        else:
            Random_GT_abstracts_gdf.append(GT_abstract_gdf)
            GT_abstract_nx = gdf_to_nx_graph(GT_abstract_gdf)
            Random_GT_abstracts.append(GT_abstract_nx)
            routenodepairs = list(GT_abstract_nx.edges())

        if debug:
            ax = GT_abstract_gdf.plot()
            ltn_gdf.plot(ax=ax, color='red', markersize=10)
            tess_gdf.plot(ax=ax, color='green', markersize=5)
            for idx, row in ltn_gdf.iterrows():
                ax.annotate(
                    text=str(row['osmid']),  # Use index or another column for labeling
                    xy=(row.geometry.x, row.geometry.y),  # Get the coordinates of the point
                    xytext=(3, 3),  
                    textcoords="offset points",
                    fontsize=8,
                    color="red"
                )

        # Prepare for routing
        GT_indices = set()
        for u, v in routenodepairs:
            pair = (u, v)
            if pair in global_processed_pairs_random or tuple(reversed(pair)) in global_processed_pairs_random:
                continue

            # Determine node types
            is_u_nei = u in all_centroids['nearest_node'].values
            is_v_nei = v in all_centroids['nearest_node'].values

            if is_u_nei and is_v_nei:
                # Neigh-Neigh: exit->exit routing
                na = all_centroids.loc[all_centroids['nearest_node'] == u, 'neighbourhood_id'].iloc[0]
                nb = all_centroids.loc[all_centroids['nearest_node'] == v, 'neighbourhood_id'].iloc[0]
                exit_a = exit_points.loc[exit_points['neighbourhood_id'] == na, 'osmid']
                exit_b = exit_points.loc[exit_points['neighbourhood_id'] == nb, 'osmid']
                best_len = float('inf')
                best_path = None
                for ea in exit_a:
                    for eb in exit_b:
                        if (ea, eb) in sp_length:
                            L = sp_length[(ea, eb)]
                            if L < best_len:
                                best_len = L
                                best_path = sp_path[(ea, eb)]
                if best_path:
                    GT_indices.update(best_path)

            elif is_u_nei and not is_v_nei:
                # Neigh -> Tessellation
                na = all_centroids.loc[all_centroids['nearest_node'] == u, 'neighbourhood_id'].iloc[0]
                exit_a = exit_points.loc[exit_points['neighbourhood_id'] == na, 'osmid']
                best_len = float('inf')
                best_path = None
                for ea in exit_a:
                    if (ea, v) in sp_length:
                        L = sp_length[(ea, v)]
                        if L < best_len:
                            best_len = L
                            best_path = sp_path[(ea, v)]
                if best_path:
                    GT_indices.update(best_path)

            elif not is_u_nei and is_v_nei:
                # Tessellation -> Neigh
                nb = all_centroids.loc[all_centroids['nearest_node'] == v, 'neighbourhood_id'].iloc[0]
                exit_b = exit_points.loc[exit_points['neighbourhood_id'] == nb, 'osmid']
                best_len = float('inf')
                best_path = None
                for eb in exit_b:
                    if (u, eb) in sp_length:
                        L = sp_length[(u, eb)]
                        if L < best_len:
                            best_len = L
                            best_path = sp_path[(u, eb)]
                if best_path:
                    GT_indices.update(best_path)

            else:
                # Tess-Tess direct
                if (u, v) in sp_path:
                    GT_indices.update(sp_path[(u, v)])

            global_processed_pairs_random.add(pair)

        cumulative_GT_indices_random.update(GT_indices)

        if len(cumulative_GT_indices_random) == 0:
            print(f"[run_random_growth] No routes found at investment level {D}. Skipping.")
            Random_GTs.append(nx.Graph())
            Random_GTs_gdf.append(gpd.GeoDataFrame())
            continue
        # Build GT subgraph and store
        GT = G_caralls[placeid].subgraph(cumulative_GT_indices_random)
        for a, b, data in GT.edges(data=True):
            if 'length' in data:
                data['weight'] = data['length']

        if GT.number_of_edges() == 0:
            print(f"[run_random_growth] Skipping empty GT at investment level {D}")
            Random_GTs.append(nx.Graph())
            Random_GTs_gdf.append(gpd.GeoDataFrame())
            continue

        Random_GTs.append(GT)
        _, Random_GT_edges = ox.graph_to_gdfs(GT)
        Random_GTs_gdf.append(Random_GT_edges)

        if debug:
            GT_nodes, GT_edges = ox.graph_to_gdfs(GT)
            GT_edges = GT_edges.to_crs(epsg=3857)
            ax = GT_edges.plot()
            ltn_gdf.plot(ax=ax, color='red', markersize=10)
            tess_gdf.plot(ax=ax, color='green', markersize=5)
            ax.set_title(f"Investment level: {D}, Number of edges: {len(GT.edges)}")

    results = {
        "placeid": placeid,
        "prune_measure": "random",
        "poi_source": poi_source,
        "prune_quantiles": investment_levels,
        "GTs": Random_GTs,
        "GT_abstracts": Random_GT_abstracts
    }
    return results


def get_composite_lcc_length(G, G_biketrack):
    """
    Returns the total length of the longest weakly connected component
    in the merged graph of G and G_biketrack. The component length is the
    sum of edge lengths (using 'length' attribute).
    """
    merged = nx.compose(G, G_biketrack)
    components = nx.weakly_connected_components(merged)
    max_length = 0
    for comp in components:
        subgraph = merged.subgraph(comp)
        length = sum(data.get('length', 0) for _, _, data in subgraph.edges(data=True))
        if length > max_length:
            max_length = length

    return max_length


def compute_total_lengths(graphs):
    # used to find the length of each bicycle network at each stage of growth
    return [sum(nx.get_edge_attributes(G, 'length').values()) for G in graphs]

def compute_abs_deviation(series, baseline):
    # find the difference against the baseline (random growth)
    return [s - b for s, b in zip(series, baseline)]


def compute_total_investment_lengths(graphs, distance_cost):
    """
    Compute the investment-weighted length of each graph in a list.
    """
    results = []
    for G in graphs:
        if not isinstance(G, nx.Graph) or len(G.edges) == 0:
            results.append(0)
            continue # this is for if during random growth an empty graph is created - rare but happens

        for u, v, data in G.edges(data=True):
            highway_type = data.get('highway', 'unclassified')
            length = data.get('length', 0)
            data['investment_length'] = length * distance_cost.get(highway_type, 1)

        results.append(sum(nx.get_edge_attributes(G, 'investment_length').values()))

    return results


def compute_length_difference(graphs):
    """Compute total difference between raw length and investment-weighted length for a series of graphs."""
    return [
        sum(nx.get_edge_attributes(G, 'length').values()) -
        sum(nx.get_edge_attributes(G, 'investment_length').values())
        for G in graphs
    ]


def compute_graph_total_length(G):
        """
        Compute the total raw length of all edges in a single graph.
        """
        return sum(data.get('length', 0) for _, _, data in G.edges(data=True))


def compute_biketrack_connected_lengths(graphs, G_biketrack):
    """
    For a list of graphs, compute:
    - the total raw length of each graph
    - the length of the biketrack network connected to each graph
    - the combined total of the two
    """
    G_lengths = []
    biketrack_lengths = []
    combined_lengths = []
    
    for G in graphs:
        length_G = compute_graph_total_length(G)
        common_nodes = set(G.nodes) & set(G_biketrack.nodes)
        length_G_biketrack = compute_graph_total_length(G_biketrack.subgraph(common_nodes)) if common_nodes else 0
        
        G_lengths.append(length_G)
        biketrack_lengths.append(length_G_biketrack)
        combined_lengths.append(length_G + length_G_biketrack)
    
    return G_lengths, biketrack_lengths, combined_lengths




def compute_lcc_lengths(graph_list, G_biketrack):
    # find the longest connected component, including any existing cycle network
    # use this to find the lcc+all the extra we connect to
    total_lengths_lcc = []
    for G in graph_list:
        merged = nx.compose(G, G_biketrack)
        components = list(nx.weakly_connected_components(merged))
        max_length = 0.0
        for comp in components:
            subgraph = merged.subgraph(comp)
            total_length = sum(data.get('length', 0) for _, _, data in subgraph.edges(data=True))
            if total_length > max_length:
                max_length = total_length
        total_lengths_lcc.append(max_length)
    return total_lengths_lcc


def load_results(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_results(results, pickle_path, csv_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    df = pd.DataFrame({k: pd.Series(v) for k, v in results.items()})
    df.to_csv(csv_path, index=False)






def create_buffer(G, buffer_walk, simplify_tolerance=10, prev_edges=None, prev_union=None):
    """Incrementally create a buffer, reusing previous buffer for existing edges."""
    gdf_edges = ox.graph_to_gdfs(G, nodes=False).to_crs(epsg=3857)
    current_edges = set(gdf_edges.index)
    new_edges = current_edges if prev_edges is None else current_edges - prev_edges
    simplified = gdf_edges.loc[list(new_edges), "geometry"].simplify(simplify_tolerance)
    buffered = simplified.buffer(buffer_walk)
    new_union = buffered.unary_union if not buffered.empty else None
    if prev_union is None:
        combined = new_union
    elif new_union is None:
        combined = prev_union
    else:
        combined = prev_union.union(new_union)
    buffer_gdf = gpd.GeoDataFrame(geometry=[combined], crs="EPSG:3857").to_crs(epsg=4326)
    return buffer_gdf, current_edges, combined

def process_and_save_buffers_parallel(G_list, name, rerun, path_base, buffer_walk, simplify_tolerance=5, max_workers=4):
    """Process and save buffers in parallel (much faster!)."""
    filename = f"{path_base}_{name}.pickle"
    if rerun or not os.path.exists(filename):
        print(f"Generating {name} buffers with parallel processing...")
        buffers = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Map graphs to create_buffer in parallel
            futures = [executor.submit(create_buffer, G, buffer_walk, simplify_tolerance) for G in G_list]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {name}"):
                buffer_gdf, _, _ = f.result()
                buffers.append(buffer_gdf)
        with open(filename, "wb") as f:
            pickle.dump(buffers, f)
    else:
        print(f"Loading cached {name} buffers...")
        with open(filename, "rb") as f:
            buffers = pickle.load(f)
    return buffers


def get_edge_path(G, path_nodes):
    # find routes between points on a graph
    edges = []
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        data = G.get_edge_data(u, v)
        if data:
            key = list(data.keys())[0]
            edges.append((u, v, key))
        else:
            edges.append((u, v, None))
    return edges

def add_ped_edges_to_cycle_graph(cycle_graph, ped_graph, ped_path_edges):
    # find edges in the pedestrian path that are not in the cycle graph and add them
    for u, v, key in ped_path_edges:
        for node in (u, v):
            if not cycle_graph.has_node(node):
                # Copy node attributes from ped_graph if present
                if ped_graph.has_node(node):
                    cycle_graph.add_node(node, **ped_graph.nodes[node])
                else:
                    # Node missing in ped_graph too — add empty node with warning
                    cycle_graph.add_node(node)
                    print(f"Warning: Node {node} missing in ped_graph; added without attributes.")
        
        # Now add the edge 
        if not cycle_graph.has_edge(u, v, key):
            edge_data = ped_graph.get_edge_data(u, v, key)
            if edge_data:
                cycle_graph.add_edge(u, v, key=key, **edge_data)
            else:
                edge_data_rev = ped_graph.get_edge_data(v, u, key)
                if edge_data_rev:
                    cycle_graph.add_edge(u, v, key=key, **edge_data_rev)
    return cycle_graph


def clean_edge_attributes(G):
    ## this cleans the edge attributes which are lists, e.g. osmid, highway, name
    ## it is required as omsnx doesn't like lists :(
    ## takes in a graph of streets in a neighbourhood, returns the same graph but cleaned
    for u, v, key, data in G.edges(keys=True, data=True):
        for attr, value in list(data.items()):
            if isinstance(value, list):
                if attr == 'osmid':
                    # Take the first osmid only
                    if len(value) > 0:
                        data[attr] = value[0]
                    else:
                        data[attr] = None  # or handle empty list case
                elif attr in ['highway', 'name']:
                    # Take the first item or join if you prefer all
                    data[attr] = value[0] if len(value) > 0 else None
                else:
                    # For other attributes, just take the first element or join them if needed
                    data[attr] = value[0] if len(value) > 0 else None
    return G

def plot_and_save_network_stats(results, output_plot_path, output_csv_path, scenario):
    network_stats = []

    for neighbourhood_name, stats in results.items():
        cycle_graph_before = stats["cycle_graph_before"]
        cycle_graph_after = stats["cycle_graph_after"]

        nodes_before = len(cycle_graph_before.nodes)
        nodes_after = len(cycle_graph_after.nodes)

        edges_before = len(cycle_graph_before.edges)
        edges_after = len(cycle_graph_after.edges)

        length_before = sum(data['length'] for u, v, k, data in cycle_graph_before.edges(keys=True, data=True))
        length_after = sum(data['length'] for u, v, k, data in cycle_graph_after.edges(keys=True, data=True))

        network_stats.append({
            'neighbourhood': neighbourhood_name,
            'nodes_before': nodes_before,
            'nodes_after': nodes_after,
            'edges_before': edges_before,
            'edges_after': edges_after,
            'length_before': length_before,
            'length_after': length_after
        })

    network_df = pd.DataFrame(network_stats).set_index('neighbourhood')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    legend_labels = ['Cycle Network', 'Cycle Network with Key Pedestrian Links']

    # Plot nodes
    network_df[['nodes_before', 'nodes_after']].plot(kind='bar', ax=axes[0], legend=False)
    axes[0].set_title(f'Number of Nodes Before vs After ({scenario})')
    axes[0].set_ylabel('Count')

    # Plot edges
    network_df[['edges_before', 'edges_after']].plot(kind='bar', ax=axes[1], legend=False)
    axes[1].set_title(f'Number of Edges Before vs After ({scenario})')
    axes[1].set_ylabel('Count')

    # Plot length
    network_df[['length_before', 'length_after']].plot(kind='bar', ax=axes[2], legend=False)
    axes[2].set_title(f'Total Edge Length Before vs After ({scenario})')
    axes[2].set_ylabel('Length (meters)')

    for ax in axes:
        ax.set_xticks(range(len(network_df.index)))
        ax.set_xticklabels(network_df.index, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Place')

    handles, _ = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_plot_path, dpi=300)
    plt.close(fig)

    # Ensure output directory exists for CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    network_df.to_csv(output_csv_path)



def patch_cycle_graph_with_pedestrian_links(neighbourhoods, debug=False):
    """
    Process each neighbourhood graph to add key pedestrian links into the cycle graph.

    Returns:
        results: dict keyed by neighbourhood name with before/after cycle graphs and info
    """
    results = {}

    for city_name, neighbourhood_gdf in neighbourhoods.items():
        if debug:
            print(f"Processing: {city_name}")
        neighbourhood_gdf['ID'] = range(1, len(gdf) + 1)
        ped_nodes, ped_edges, ped_graph = get_neighbourhood_pedestrian_graph(neighbourhood_gdf, debug=debug)
        cycle_nodes, cycle_edges, cycle_graph = get_neighbourhood_street_graph(neighbourhood_gdf, debug=debug)

        exit_nodes = get_exit_nodes({city_name: neighbourhood_gdf}, cycle_graph, buffer_distance=5)
        exit_node_ids = exit_nodes['osmid'].unique()

        combined_graph = nx.compose(ped_graph, cycle_graph)
        combined_routes = {origin: {} for origin in exit_node_ids}
        cycle_routes = {origin: {} for origin in exit_node_ids}
        combined_reachability = pd.DataFrame(False, index=exit_node_ids, columns=exit_node_ids)
        cycle_reachability = pd.DataFrame(False, index=exit_node_ids, columns=exit_node_ids)

        for origin in exit_node_ids:
            for destination in exit_node_ids:
                if origin == destination:
                    combined_reachability.loc[origin, destination] = True
                    combined_routes[origin][destination] = []
                    cycle_reachability.loc[origin, destination] = True
                    cycle_routes[origin][destination] = []
                    continue

                try:
                    combined_path_nodes = nx.shortest_path(combined_graph, source=origin, target=destination, weight='length')
                    combined_reachability.loc[origin, destination] = True
                    combined_routes[origin][destination] = get_edge_path(combined_graph, combined_path_nodes)
                except nx.NetworkXNoPath:
                    combined_reachability.loc[origin, destination] = False
                    combined_routes[origin][destination] = []

                try:
                    cycle_path_nodes = nx.shortest_path(cycle_graph, source=origin, target=destination, weight='length')
                    cycle_reachability.loc[origin, destination] = True
                    cycle_routes[origin][destination] = get_edge_path(cycle_graph, cycle_path_nodes)
                except nx.NetworkXNoPath:
                    cycle_reachability.loc[origin, destination] = False
                    cycle_routes[origin][destination] = []

        # Save cycle graph before patching
        cycle_graph_before = deepcopy(cycle_graph)

        # Track added pedestrian edges
        patched_edges_set = set()
        for origin in exit_node_ids:
            for destination in exit_node_ids:
                if not cycle_reachability.loc[origin, destination] and combined_reachability.loc[origin, destination]:
                    ped_route_edges = combined_routes[origin][destination]
                    if ped_route_edges:
                        for u, v, k in ped_route_edges:
                            patched_edges_set.add((u, v, k))
                        cycle_graph = add_ped_edges_to_cycle_graph(cycle_graph, ped_graph, ped_route_edges)

        # Save cycle graph after patching
        cycle_graph_after = cycle_graph

        # Save the patched pedestrian edges 
        _, edges_before = ox.graph_to_gdfs(cycle_graph_before)
        _, edges_after  = ox.graph_to_gdfs(cycle_graph_after)
        patched_edges_gdf = edges_after.loc[~edges_after.index.isin(edges_before.index)]
        
        # Store results
        results[city_name] = {
            'cycle_graph_before': cycle_graph_before,
            'cycle_graph_after': cycle_graph_after,
            'ped_graph': ped_graph,
            'exit_nodes': exit_node_ids,
            'combined_routes': combined_routes,
            'cycle_routes_before': cycle_routes,
            'combined_reachability': combined_reachability,
            'cycle_reachability_before': cycle_reachability,
            'patched_edges': patched_edges_gdf
        }

        if debug:
            print(f"Finished processing pedestrian links in: {city_name}")

    return results


def get_neighbourhood_pedestrian_graph(gdf, debug=False):
    """
    Get pedestrian edges within each neighbourhood.

    Parameters:
        gdf (GeoDataFrame): GeoDataFrame containing neighbourhood polygons eithin an ID column.
        debug (bool): If True, plot edges colored by neighbourhood ID.

    Returns:
        nodes (GeoDataFrame): Nodes of the pedestrian graph.
        edges (GeoDataFrame): Edges of the pedestrian graph.
        G (MultiDiGraph): The combined pedestrian graph.
    """

    # Buffer polygons slightly in Mercator projection (to avoid boundary issues)
    gdf_mercator = gdf.to_crs(epsg=3857)
    gdf_mercator = gdf_mercator.buffer(10)  # 10 meters buffer
    gdf_buffered = gpd.GeoDataFrame(geometry=gdf_mercator, crs="EPSG:3857").to_crs(epsg=4326)
    pedestrian_network = nx.MultiDiGraph()
    for i, polygon in enumerate(gdf_buffered.geometry):
        try:
            ped_net = ox.graph_from_polygon(polygon, network_type='walk')
            if len(ped_net) == 0:
                if debug:
                    print(f"Polygon {i}: Empty pedestrian graph returned. Skipping.")
                continue
            pedestrian_network = nx.compose(pedestrian_network, ped_net)
        except ValueError as e:
            if debug:
                print(f"Polygon {i}: Skipping due to ValueError: {e}")
            continue
        except Exception as e:
            if debug:
                print(f"Polygon {i}: Skipping due to other error: {e}")
            continue
    if len(pedestrian_network.nodes) == 0:
        if debug:
            print("No valid pedestrian network data found in any neighbourhood.")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), nx.MultiDiGraph()
    nodes, edges = ox.graph_to_gdfs(pedestrian_network)
    # Filter out edges with highway=steps or indoor=* tag
    if 'highway' in edges.columns:
        edges = edges[~edges['highway'].apply(lambda x: ('steps' in x if isinstance(x, list) else x == 'steps'))]
    if 'indoor' in edges.columns:
        edges = edges[edges['indoor'].isna()]
    edges = gpd.sjoin(edges, gdf[['ID', 'geometry']], how="left", predicate='intersects')
    # Drop edges that do not fall within any neighbourhood polygon
    edges = edges.dropna(subset=['ID'])
    if debug:
        unique_ids = edges['ID'].dropna().unique()
        np.random.seed(42)
        random_colors = {ID: mcolors.to_hex(np.random.rand(3)) for ID in unique_ids}
        edges['color'] = edges['ID'].map(random_colors)
        edges['color'] = edges['color'].fillna('#808080')  # Gray fallback

        fig, ax = plt.subplots(figsize=(10, 10))
        edges.plot(ax=ax, color=edges['color'])
        ax.set_title('Pedestrian Edges Colored by Neighbourhood ID')
        plt.show()
    # Rebuild graph from filtered nodes and edges
    u_nodes = edges.index.get_level_values('u')
    v_nodes = edges.index.get_level_values('v')
    unique_nodes = set(u_nodes).union(v_nodes)
    nodes = nodes.loc[nodes.index.intersection(unique_nodes)]
    G = ox.graph_from_gdfs(nodes, edges)

    return nodes, edges, G
