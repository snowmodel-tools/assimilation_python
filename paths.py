#!/usr/bin/env python
# coding: utf-8


import geopandas as gpd
import requests


################# EDIT ##########################
domain = 'WY'
#################################################

############# General CSO info ##################
#CSO data path
CSOpath = 'cso-data.geojson'

#assimilation files 
assimPath = '/nfs/attic/dfh/Aragon2/CSOassim/'+domain+'/'

# Figure path
figpath = '/nfs/attic/dfh/Aragon2/CSOfigs/'+domain+'/'

#geojson of all snotel stations in domain
gdf = gpd.read_file('/nfs/attic/dfh/Aragon2/CSOdata/'+domain+'/CSO_SNOTEL_sites.geojson')

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

# number of columns in domain
nx = int(domains[domain]['ncols'])
# number of rows in domain
ny = int(domains[domain]['nrows'])

# dem path
dem_path = '/nfs/attic/dfh/Aragon2/CSOdmn/'+domain+'/DEM_'+domain+'.tif'

# nlcd path
lc_path = '/nfs/attic/dfh/Aragon2/CSOdmn/'+domain+'/NLCD2016_'+domain+'.tif'

##################### SM ######################
#path to SM
SMpath = '/nfs/attic/dfh/Aragon2/WY_scratch/jan2021_snowmodel-dfhill_all/'
#path to SM .f files
codepath = SMpath+'code'
#path to.inc file
incFile = SMpath+'code/snowmodel.inc'

################# Calibration ##################


################# Assimilation #################

#path to SM outputs on scratch
SMout_scrach = SMpath+domain+'/'

#path to baseline SM run
SM_noassim = SMpath + 'outputs/wo_assim/swed.gdat'

#evaluation snotel sites 
snotel_eval_sites = gpd.read_file(assimPath + 'eval_snotel.geojson')

#assimilation snotel stations
snotel_assim_sites = gpd.read_file(assimPath + 'assim_snotel_sites.geojson')