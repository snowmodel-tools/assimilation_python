#!/usr/bin/env python
# coding: utf-8


import geopandas as gpd
import requests


################# EDIT ##########################
domain = 'WY'
#################################################


#assimilation files 
assimPath = 'extra/'

#path to CSO domain
domains_resp = requests.get("https://raw.githubusercontent.com/snowmodel-tools/preprocess_python/master/CSO_domains.json")
domains = domains_resp.json()
    
# #start date 'YYYY-MM-DD'   
# st = domains[domain]['st']
# #end date
# ed = domains[domain]['ed']

#Snotel bounding box
Bbox = domains[domain]['Bbox']

# CSO projection
stn_proj = domains[domain]['stn_proj']

# CSO projection
mod_proj = domains[domain]['mod_proj']

# dem path
dem_path = 'topo_vege/DEM_'+domain+'.tif'

# nlcd path
lc_path = 'topo_vege/NLCD2016_'+domain+'.tif'

##################### SM ######################
#path to SM
SMpath = '/nfs/attic/dfh/Aragon2/WY_scratch/jan2021_snowmodel-dfhill_elev/'
#path to SM .f files
codepath = SMpath+'code'
#path to.inc file
incFile = SMpath+'code/snowmodel.inc'

################# Calibration ##################


################# Assimilation #################


#path to SM outputs on scratch
SMout_scrach = SMpath+domain+'/'

#evaluation snotel sites 
snotel_eval_sites = gpd.read_file(SMpath+ 'extra/eval_snotel.geojson')

#assimilation snotel stations
snotel_assim_sites = gpd.read_file(SMpath+ 'extra/assim_snotel_sites.geojson')

#gdat path
gdat_out_path = '/scratch/Nina/WY_gdat/'

############# General CSO info ##################
#CSO data path
CSOpath = SMpath+'extra/cso-data.geojson'

#geojson of all snotel stations in domain
gdf = gpd.read_file(SMpath+'extra/CSO_SNOTEL_sites.geojson')
