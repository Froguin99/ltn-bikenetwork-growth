# Growing Urban Bicycle Networks - with an LTN twist

This is code modified from the scientific paper [*Growing urban bicycle networks*](https://www.nature.com/articles/s41598-022-10783-y) by [M. Szell](http://michael.szell.net/), S. Mimar, T. Perlman, [G. Ghoshal](http://gghoshal.pas.rochester.edu/), and [R. Sinatra](http://www.robertasinatra.com/). It adapts the code to work with Low Traffic Neighbourhoods, in order to reduce the amount of kilometers of investment required whilst still providing a connected network plan. The LTNs are sourced from this project: [https://github.com/Froguin99/LTN-Detection](https://github.com/Froguin99/LTN-Detection).

The code downloads and pre-processes data from OpenStreetMap, prepares points of interest, runs simulations, measures and saves the results, creates videos and plots.

**Orignal Paper**: [https://www.nature.com/articles/s41598-022-10783-y](https://www.nature.com/articles/s41598-022-10783-y)

**Recent conference paper**: [https://zenodo.org/records/15231749](https://zenodo.org/records/15231749)

[![Example of using demand based growth on Newcastle Upon Tyne, United Kingdom](readmevideo.gif)]()
*Example of using demand based growth on Newcastle Upon Tyne, United Kingdom*

## Instructions

### 1. Git clone the project

Run from your terminal:

```sh
git clone https://github.com/Froguin99/ltn-bikenetwork-growth.git
```

and then move inside with:

```sh
cd ltn-bikenetwork-growth
```

### 2. Set up Jupyter kernel in `main` environment

If you want to use this project's `main` environment in Jupyter, run:

```bash
calkit nb check-kernel -e main
```

> [!NOTE]
> [Calkit](https://github.com/calkit/calkit) (a research project management tool)
> can be installed with either `pip install calkit-python` or
> `uv tool install calkit-python`.

This allows you to run Jupyter with the kernel
`ltn-bikenetwork-growth: main`
(Kernel > Change Kernel > ltn-bikenetwork-growth: main).

### 3. Run the code locally

Single (or few/small) cities can be run locally by a manual, step-by-step execution of Jupyter notebooks.

A notebook can be executed from the command line in the appropriate
environment with, for example:

```bash
calkit nb execute -e main code/01_prepare_neighbourhoods.ipynb
```

An executed copy of the notebook will be created in
`.calkit/notebooks/executed`.

1. Populate [`parameters/cities.csv`](parameters/cities.csv), see below. Currently only local authority districts in the North-East of the UK can be used. However, if you'd like to get further places working, raise an issue of the tracker and we can work on it!
2. Run notebooks 01, 02, 03 once to download and prepare all networks and POIs.
3. Run notebooks 04, 05 to run the processing and analysis for location.
4. Run 06 once more than one location has been processed and analysed to get further analysis.

## Folder structure and output
The main folder/repo is `bikenwgrowth`, containing Jupyter notebooks (`code/`), preprocessed data (`data/`), parameters (`parameters/`), result plots (`plots/`), HPC server scripts and jobs (`scripts/`).

Most of the generated data output (network plots, videos, results, exports, logs) makes up many GBs and is stored in the separate external folder `bikenwgrowth_external`. To set up different paths, edit [`code/path.py`](code/path.py)


## Populating cities.csv

`cities.csv` holds the location of the place to analyse. As we use demand data bespoke to England and Wales throughout the analysis, only locations within these areas should be used. The file takes places in the format `placeid;nominatimstring;countryid;name`, so to run Newcastle we would use `newcastle;Newcastle Upon Tyne;gbr;Newcastle Upon Tyne`, whilst North Tyneside would take the form `north_tyneside;North Tyneside;gbr;North Tyneside`. **note** Currently only one location can be input at a time. Multi-location analysis will be included in a future update, but for now it is recommended to run through the code up to (but not including) notebook `06` with each place at a time.

## Parameters
The `parameters.yml` contains values which can be changed to alter the analysis. It is not recommended to change any of the values currently.

### Checking nominatimstring
* Go to e.g. [https://nominatim.openstreetmap.org/ui/search.html?q=paris%2C+france](https://nominatim.openstreetmap.org/ui/search.html?q=paris%2C+france) and enter the search string. If a correct polygon (or multipolygon) pops up it should be fine. If not leave the field empty and acquire a shape file, see below.

### Acquiring shape file
* Go to [Overpass](https://overpass-turbo.eu/), to the city, and run:
    `relation["boundary"="administrative"]["name:en"="Copenhagen Municipality"]({{bbox}});(._;>;);out skel;`
* Export: Download as GPX
* Use QGIS to create a polygon, with Vector > Join Multiple Lines, and Processing Toolbox > Polygonize (see [Stackexchange answer 1](https://gis.stackexchange.com/questions/98320/connecting-two-line-ends-in-qgis-without-resorting-to-other-software) and [Stackexchange answer 2](https://gis.stackexchange.com/questions/207463/convert-a-line-to-polygon))
