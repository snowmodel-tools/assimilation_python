#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np 
import pandas as pd
from datetime import date, timedelta
import glob
import xarray as xr
import rioxarray 


# In[99]:


# define directory for tif files
clim_dir = '/nfs/attic/dfh/data/snodas/snodas_tif/clim/'
hs_dir = '/nfs/attic/dfh/data/snodas/snodas_tif/daily_Hs/'


# In[6]:


sdate = date(2003,1,1)   # start date
edate = date(2020,12,31)   # end date
dates = pd.date_range(sdate,edate,freq='d')
# Create DataFrame
df = pd.DataFrame({'day':dates.day,'month':dates.month})
data = df.drop_duplicates()
data.reset_index(inplace=True,drop=True)


# In[30]:


hs_dir = '/nfs/attic/dfh/data/snodas/snodas_tif/daily_Hs/'
for idx in data.index.tolist():
    print(f"{data.month[idx]:02d}"+f"{data.day[idx]:02d}")
    geotiff_list = glob.glob(hs_dir+'us_ssmv11036tS__T0001TTNATS*'+f"{data.month[idx]:02d}"+f"{data.day[idx]:02d}"+'05HP001.tif',recursive = True)

    # Create variable used for time axis
    time_var = xr.Variable('count', np.arange(0,len(geotiff_list),1))

    # Load in and concatenate all individual GeoTIFFs
    geotiffs_da = xr.concat([xr.open_rasterio(i) for i in geotiff_list],
                            dim=time_var)

    # Covert our xarray.DataArray into a xarray.Dataset
    geotiffs_ds = geotiffs_da.to_dataset('band')

    # Rename the variable to a more useful name
    geotiffs_ds = geotiffs_ds.rename({1: 'Hs'})
    
    print('std')
    # calculate standard deviation
    std = geotiffs_ds.std(dim = 'count',keep_attrs=True)
    #specify crs
    std.rio.write_crs("epsg:4326", inplace=True)
    #define filename
    outFname = clim_dir+ f"{data.month[idx]:02d}"+f"{data.day[idx]:02d}"+'1036std.tif'
    std["Hs"].rio.to_raster(outFname)

    print('q1')
    # calculate 1st quantile
    q1 = geotiffs_ds.quantile(.25,dim = 'count',keep_attrs=True)
    #specify crs
    q1.rio.write_crs("epsg:4326", inplace=True)
    #define filename
    outFname = clim_dir+ f"{data.month[idx]:02d}"+f"{data.day[idx]:02d}"+'1036q1.tif'
    q1["Hs"].rio.to_raster(outFname)

    print('q3')
    # calculate 3rd quantile
    q3 = geotiffs_ds.quantile(.75,dim = 'count',keep_attrs=True)
    #specify crs
    q3.rio.write_crs("epsg:4326", inplace=True)
    #define filename
    outFname = clim_dir+ f"{data.month[idx]:02d}"+f"{data.day[idx]:02d}"+'1036q3.tif'
    q3["Hs"].rio.to_raster(outFname)

