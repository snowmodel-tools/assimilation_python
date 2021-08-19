#!/usr/bin/env python
# coding: utf-8



import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import json
import rasterio as rio
import gdal
import richdem as rd
from scipy import ndimage
from rasterstats import point_query


#########################################################################
############################ USER INPUTS ################################
#########################################################################
# NOTE: to runn assim, set irun_data_assim = 1 in .par file


# DOMAIN
# choose the modeling domain
domain = 'WY'

# PATHS
dataPath = '/nfs/attic/dfh/Aragon2/CSOdmn/'+domain+'/'
#path to dem .tif
dem_path = dataPath + 'DEM_'+domain+'.tif'
#path to landcover .tif
lc_path = dataPath + 'NLCD2016_'+domain+'.tif'
#path to SnowModel
SMpath = '/nfs/attic/dfh/Aragon2/CSOsm/jan2021_snowmodel-dfhill_'+domain+'/'

# TIME
# choose if want to set 'manual' or 'auto' date 
date_flag = 'auto'
# If you choose 'manual' set your dates below  
# This will start on the 'begin' date at 0:00 and the last iteration will 
# be at 18:00 on the day of the 'end' date below.
st_dt = '2019-02-01'
ed_dt = '2019-02-05'

# ASSIM VARIABLE
#can be 'all',elev','slope','tc','delta_day','M', 'lc', 'aspect'
setvar = 'all'
#########################################################################


# Date setup function
def set_dates(st_dt,ed_dt,date_flag):
    if date_flag == 'auto':
        # ###automatically select date based on today's date 
        hoy = date.today()
        antes = timedelta(days = 3)
        #end date 3 days before today's date
        fecha = hoy - antes
        eddt = fecha.strftime("%Y-%m-%d") 
        #start date
        if fecha.month <10:
            styr = fecha.year - 1
        else:
            styr = fecha.year
        stdt = str(styr)+'-10-01'
    elif date_flag == 'manual':
        stdt = st_dt
        eddt = ed_dt
    return stdt, eddt

stdt, eddt = set_dates(st_dt,ed_dt,date_flag)


# Function to get SWE from CSO Hs

def swe_calc(gdf):
    #convert snow depth to mm to input into density function
    H = gdf.depth.values*10
    #Get temp info at each point
    TD = np.array([point_query([val], 'td_final.txt')[0] for val in gdf.geometry])
    #Get pr info at each point
    PPTWT = np.array([point_query([val], 'ppt_wt_final.txt')[0] for val in gdf.geometry])
    #Determine day of year
    dates = pd.to_datetime(gdf.timestamp, format='%Y-%m-%dT%H:%M:%S').dt.date.values
    DOY = [date.toordinal(date(dts.year,dts.month,dts.day))-date.toordinal(date(dts.year,9,30)) for dts in dates]
    DOY = np.array([doy + 365 if doy < 0 else doy for doy in DOY])
    #Apply regression equation 
    a = [0.0533,0.948,0.1701,-0.1314,0.2922] #accumulation phase
    b = [0.0481,1.0395,0.1699,-0.0461,0.1804]; #ablation phase
    SWE = a[0]*H**a[1]*PPTWT**a[2]*TD**a[3]*DOY**a[4]*(-np.tanh(.01*\
            (DOY-180))+1)/2 + b[0]*H**b[1]*PPTWT**b[2]*TD**b[3]*DOY**b[4]*\
            (np.tanh(.01*(DOY-180))+1)/2;
    #convert swe to m to input into SM
    gdf['SWE'] = SWE/1000
    gdf['DOY'] = DOY
    return gdf


# ## Function to build geodataframe of CSO point observations 

def get_cso(st, ed, domain):
    ''' 
    st = start date 'yyyy-mm-dd'
    ed = end date 'yyyy-mm-dd'
    domain = string label of defined CSO domain
    '''
    
    #path to CSO domains
    domains_resp = requests.get("https://raw.githubusercontent.com/snowmodel-tools/preprocess_python/master/CSO_domains.json")
    domains = domains_resp.json()

    Bbox = domains[domain]['Bbox']
    stn_proj = domains[domain]['stn_proj']
    mod_proj = domains[domain]['mod_proj']
    
    #Issue CSO API observations request and load the results into a GeoDataFrame
    params = {
      "bbox": f"{Bbox['lonmin']},{Bbox['latmax']},{Bbox['lonmax']},{Bbox['latmin']}",
      "start_date": st,
      "end_date": ed,
      "format": "geojson",
      "limit": 5000,
    }

    csodata_resp = requests.get("https://api.communitysnowobs.org/observations", params=params)
    csodatajson = csodata_resp.json()
    #turn into geodataframe
    gdf = gpd.GeoDataFrame.from_features(csodatajson, crs=stn_proj)
    
    mask = (gdf['timestamp'] >= st) & (gdf['timestamp'] <= ed)
    gdf = gdf.loc[mask]
    gdf=gdf.reset_index(drop=True)
    print('Total number of CSO in domain = ',len(gdf))

    #ingdf = extract_meta(gdf,domain,dem_path,lc_path)
    ingdf = swe_calc(gdf)
    
    return ingdf

CSOgdf = get_cso(stdt, eddt, domain)


# # Function to format & export for SM 


def make_SMassim_file(new,outFpath):
    '''
    new = dataframe with subset of CSO data 
    
    outFpath = output path to formated assim data for SM 
    '''
    print('Generating assim file')
    f= open(outFpath,"w+")
    new['Y'] = pd.DatetimeIndex(new['timestamp']).year

    tot_obs=len(new)
    uq_day = np.unique(new.dt)
    num_days = len(uq_day)
    f.write('{:02.0f}\n'.format(num_days))
    for j in range(len(uq_day)):
        obs = new[new['dt']==uq_day[j]]
        d=new.D[new['dt']==uq_day[j]].values
        m=new.M[new['dt']==uq_day[j]].values
        y=new.Y[new['dt']==uq_day[j]].values
        date = str(y[0])+' '+str(m[0])+' '+str(d[0])
        obs_count = str(len(obs))
        f.write(date+' \n')
        f.write(obs_count+' \n')
        for k in range(len(obs)):
            ids = 100+k
            x= obs.geometry.x[obs.index[k]]
            y=obs.geometry.y[obs.index[k]]
            swe=obs.swe[obs.index[k]]
            f.write('{:3.0f}\t'.format(ids)+'{:10.0f}\t'.format(x)+'{:10.0f}\t'.format(y)+'{:3.2f}\n'.format(swe))
    f.close() 


# function to edit SnowModel Files other than .par
# for assim - have to adjust .inc file to specify # of obs being assimilated
def replace_line(file_name, line_num, text):
    ''' 
    file_name = file to edit
    line_num = line number in file to edit
    text = nex text to put in
    '''
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


#edit par file for correct number of timesteps 
parFile = SMpath + 'snowmodel.par'
value = str((datetime.strptime(eddt,'%Y-%m-%d')-datetime.strptime(stdt,'%Y-%m-%d')).days*4+4)
print('Number of timesteps =',value)
replace_line(parFile,11,value +'			!max_iter - number of model time steps\n')

#function to make SM assim file based on selected landscape characteristic
#var can be 'all',elev','slope','tc','delta_day','M', 'lc', 'aspect'

def SMassim_ensemble(gdf,var,SMpath):
    '''
    gdf: this is the geodataframe containing all CSO obs taken over the time period of interest
    var: this is the landscape characteristic that will be made into an assimilation ensemble 
        'all': assimilate all inputs to SM
        'elev': assimilate each of n elevation bands. 
            Default = breaks elevation range into 5 bands
        'slope': assimilate each of n slope bands. 
            Default = breaks slope range into 5 bands
        'tc': assimilate each of n terrain complexity score bands. 
            Default = breaks tc score range into 5 bands
        'delta_day': sets a minimum number of days between assimilated observations. 
            -> only 1 observation is selected each day
        'M': assimilate data from each month
        'lc': assimilate data from each land cover class
        'aspect': assimilate data from each aspect N, E, S, W
    '''
    #create directory with initiation date for ensemble if it doesn't exist
    outFpath = SMpath+'swe_assim/swe_obs_test.dat'
    codepath = SMpath+'/code/'
    incFile = SMpath+'code/snowmodel.inc'
    if var == 'all':
        new = gdf
        make_SMassim_file(new,outFpath)
        #edit .inc file
        replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(new)+1)+')\n')
        #compile SM
        get_ipython().run_line_magic('cd', '$codepath')
        get_ipython().system(' ./compile_snowmodel.script')
        #run snowmodel 
        get_ipython().run_line_magic('cd', '$SMpath')
        get_ipython().system(' ./snowmodel')


# Run SM with CSO assim

SMassim_ensemble(CSOgdf,setvar,SMpath)