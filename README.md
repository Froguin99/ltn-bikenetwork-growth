
# Growing Urban Bicycle Networks - with an LTN twist

This is code modified from the scientific paper [*Growing urban bicycle networks*](https://www.nature.com/articles/s41598-022-10783-y) by [M. Szell](http://michael.szell.net/), S. Mimar, T. Perlman, [G. Ghoshal](http://gghoshal.pas.rochester.edu/), and [R. Sinatra](http://www.robertasinatra.com/). It adapts the code to work with Low Traffic Neighbourhoods, in order to reduce the amount of kilometers of investment required whilst still providing a connected network plan. The LTNs are sourced from this project: [https://github.com/Froguin99/LTN-Detection](https://github.com/Froguin99/LTN-Detection). 

The code downloads and pre-processes data from OpenStreetMap, prepares points of interest, runs simulations, measures and saves the results, creates videos and plots. 

**Orignal Paper**: [https://www.nature.com/articles/s41598-022-10783-y](https://www.nature.com/articles/s41598-022-10783-y)  

**Recent conferance paper**: [https://zenodo.org/records/15231749](https://zenodo.org/records/15231749)

[![Example of using demand based growth on Newcastle Upon Tyne, United Kingdom](readmevideo.gif)]()
*Example of using demand based growth on Newcastle Upon Tyne, United Kingdom*

## Instructions

### 1. Git clone the project _without_ the full history 

Run from your terminal:

```
git clone -b main --single-branch https://github.com/Froguin99/bikenwgrowth-with-LTNs --depth 1
```

### 2. Install the conda environment `growbikenet`

In your terminal, navigate to the project folder `bikenwgrowth` and use [`conda`](https://docs.conda.io/projects/conda/en/latest/index.html)
or [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
or [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to run:

```
mamba env create -f environment.yml
mamba activate growbikenet
```
**Note for MacOS users** - please use `environment_mac.yml` when installing, this should work excatly the same but if you run in to any issues please let me know, as I'm not a apple user!

#### Environment creation from command line

If the above doesn't work, you can manually create the environment from your command line (not recommended):

```
mamba create --override-channels -c conda-forge -n growbikenet python=3.12 osmnx=1.9.4 python-igraph watermark haversine rasterio tqdm geojson
mamba activate growbikenet
mamba install -c conda-forge ipywidgets
pip install opencv-python
pip install --user ipykernel
```

#### Set up Jupyter kernel

If you want to use the environment `growbikenet` in Jupyter, run:

```bash
python -m ipykernel install --user --name=growbikenet
```

This allows you to run Jupyter with the kernel `growbikenet` (Kernel > Change Kernel > growbikenet)


### 3. Run the code locally

Single (or few/small) cities can be run locally by a manual, step-by-step execution of Jupyter notebooks:

1. Populate [`parameters/cities.csv`](parameters/cities.csv), see below. Leave default values to run the code on two small cities. Currently only local authority districts in the North East of the UK can be used. However, if you'd like to get further places working, raise an issue of the tracker and we can work on it!
2. Navigate to the [`code`](code/) folder.
3. Run notebooks 01, 02, 03 once to download and prepare all networks and POIs.  
4. Run notebooks 04, 05, 06 for each parameter set (see below), set in [`parameters/parameters.py`](parameters/parameters.py)



## Folder structure and output
The main folder/repo is `bikenwgrowth`, containing Jupyter notebooks (`code/`), preprocessed data (`data/`), parameters (`parameters/`), result plots (`plots/`), HPC server scripts and jobs (`scripts/`).

Most of the generated data output (network plots, videos, results, exports, logs) makes up many GBs and is stored in the separate external folder `bikenwgrowth_external`. To set up different paths, edit [`code/path.py`](code/path.py)


## Populating cities.csv

### Checking nominatimstring  
* Go to e.g. [https://nominatim.openstreetmap.org/ui/search.html?q=paris%2C+france](https://nominatim.openstreetmap.org/ui/search.html?q=paris%2C+france) and enter the search string. If a correct polygon (or multipolygon) pops up it should be fine. If not leave the field empty and acquire a shape file, see below.

### Acquiring shape file  
* Go to [Overpass](https://overpass-turbo.eu/), to the city, and run:
    `relation["boundary"="administrative"]["name:en"="Copenhagen Municipality"]({{bbox}});(._;>;);out skel;`
* Export: Download as GPX
* Use QGIS to create a polygon, with Vector > Join Multiple Lines, and Processing Toolbox > Polygonize (see [Stackexchange answer 1](https://gis.stackexchange.com/questions/98320/connecting-two-line-ends-in-qgis-without-resorting-to-other-software) and [Stackexchange answer 2](https://gis.stackexchange.com/questions/207463/convert-a-line-to-polygon))
