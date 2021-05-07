#!/usr/bin/env python
# coding: utf-8

# In[2]:


from collections import OrderedDict
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry as sgeom
import ulmo
import json
import datetime
from paths import *


# In[ ]:


# functions to get SNOTEL stations as geodataframe
def sites_asgdf(ulmo_getsites, crs=stn_proj):
    """ Convert ulmo.cuahsi.wof.get_sites response into a point GeoDataframe
    """
    
    # Note: Found one SNOTEL site that was missing the location key
    sites_df = pd.DataFrame.from_records([
        OrderedDict(code=s['code'], 
        longitude=float(s['location']['longitude']), 
        latitude=float(s['location']['latitude']), 
        name=s['name'], 
        elevation_m=s['elevation_m'])
        for _,s in ulmo_getsites.items()
        if 'location' in s
    ])

    sites_gdf = gpd.GeoDataFrame(
        sites_df, 
        geometry=gpd.points_from_xy(sites_df['longitude'], sites_df['latitude']),
        crs=crs
    )
    return sites_gdf

def get_snotel(Bbox, mod_proj):
    # Convert the bounding box dictionary to a shapely Polygon geometry using sgeom.box
    box_sgeom = sgeom.box(Bbox['lonmin'], Bbox['latmin'], Bbox['lonmax'], Bbox['latmax'])
    box_gdf = gpd.GeoDataFrame(geometry=[box_sgeom], crs=stn_proj)
    
    # WaterML/WOF WSDL endpoint url 
    wsdlurl = "http://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL"

    # get dictionary of snotel sites 
    sites = ulmo.cuahsi.wof.get_sites(wsdlurl,user_cache=True)

    #turn sites to geodataframe 
    snotel_gdf = sites_asgdf(sites)
    
    #clip snotel sites to domain bounding box
    gdf = gpd.sjoin(snotel_gdf, box_gdf, how="inner")
    gdf.drop(columns='index_right', inplace=True)
    gdf.reset_index(drop=True, inplace=True)

    #add columns with projected coordinates 
    CSO_proj = gdf.to_crs(mod_proj)
    gdf['easting'] = CSO_proj.geometry.x
    gdf['northing'] = CSO_proj.geometry.y
    
    return gdf


# In[ ]:


# functions to get SWE timeseries at SNOTEL stations
def fetch(sitecode, variablecode, start_date, end_date):
    print(sitecode, variablecode, start_date, end_date)
    values_df = None
    try:
        #Request data from the server
        site_values = ulmo.cuahsi.wof.get_values(
            wsdlurl, 'SNOTEL:'+sitecode, variablecode, start=start_date, end=end_date
        )
        #Convert to a Pandas DataFrame   
        values_df = pd.DataFrame.from_dict(site_values['values'])
        #Parse the datetime values to Pandas Timestamp objects
        values_df['datetime'] = pd.to_datetime(values_df['datetime'])
        #Set the DataFrame index to the Timestamps
        values_df.set_index('datetime', inplace=True)
        #Convert values to float and replace -9999 nodata values with NaN
        values_df['value'] = pd.to_numeric(values_df['value']).replace(-9999, np.nan)
        #Remove any records flagged with lower quality
        values_df = values_df[values_df['quality_control_level_code'] == '1']
    except:
        print("Unable to fetch %s" % variablecode)   
    return values_df

# returns swe timeseries in 
def get_swe(gdf,st, ed):
    stn_swe = pd.DataFrame(index=pd.date_range(start=st, end=ed))

    for sitecode in gdf.code:
        try:
            swe = fetch(sitecode, variablecode='SNOTEL:WTEQ_D', start_date=st, end_date=ed)
            #check for nan values
            if len(swe.value[np.isnan(swe.value)]) > 0:
                #check if more than 10% of data is missing
                if len(swe.value[np.isnan(swe.value)])/len(swe) > .1:
                    print('More than 10% of days missing')
                    gdf.drop(CSO_gdf.loc[gdf['code']==sitecode].index, inplace=True)
                    continue
            stn_swe[sitecode] = swe.value
        except:
            gdf.drop(gdf.loc[gdf['code']==sitecode].index, inplace=True)     
            
    #convert SNOTEL units[in] to SnowModel units [m]
    for sitecode in CSO_gdf.code:
        # overwrite the original values (no use for the original values in inches)
        stn_swe[sitecode] = 0.0254 * stn_swe[sitecode]
    return stn_swe


# In[ ]:


#compute model performance metrics
def calc_metrics(mod_swe,stn_swe):
    swe_stats = []
    

    #remove days with zero SWE at BOTH the station and the SM pixel
    idx = np.where((stn_swe != 0) | (mod_swe != 0))
    mod_swe = mod_swe[idx]
    stn_swe = stn_swe[idx]

    #remove days where station has nan values 
    idx = np.where(~np.isnan(stn_swe))
    mod_swe = mod_swe[idx]
    stn_swe = stn_swe[idx]

    #R-squared value
    r = np.corrcoef(stn_swe, mod_swe)[0,1]**2
    swe_stats.append(r)

    #mean bias error
    mbe = (sum(mod_swe - stn_swe))/mod_swe.shape[0]
    swe_stats.append(mbe)

    #root mean squared error
    rmse = np.sqrt((sum((mod_swe - stn_swe)**2))/mod_swe.shape[0])
    swe_stats.append(rmse)

    # Nash-Sutcliffe model efficiency coefficient, 1 = perfect, assumes normal data 
    nse_top = sum((mod_swe - stn_swe)**2)
    nse_bot = sum((stn_swe - np.mean(stn_swe))**2)
    nse = 1-(nse_top/nse_bot)
    swe_stats.append(nse)

    # Kling-Gupta Efficiency, 1 = perfect
    kge_std = (np.std(mod_swe)/np.std(stn_swe))
    kge_mean = (np.mean(mod_swe)/np.mean(stn_swe))
    kge_r = np.corrcoef(stn_swe,mod_swe)[1,0]
    kge = 1 - (np.sqrt((kge_r-1)**2)+((kge_std-1)**2)+(kge_mean-1)**2)
    swe_stats.append(kge)   
        
    return swe_stats


# In[ ]:


def make_SMassim_file(new,outFpath):
    f= open(outFpath,"w+")
    new['Y'] = pd.DatetimeIndex(new['dt']).year

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
            x= obs.x[obs.index[k]]
            y=obs.y[obs.index[k]]
            swe=obs.swe[obs.index[k]]
            f.write('{:3.0f}\t'.format(ids)+'{:10.0f}\t'.format(x)+'{:10.0f}\t'.format(y)+'{:3.2f}\n'.format(swe))
    f.close() 


# In[ ]:
def make_SMassim_file_snotel(sample,new,outFpath):
    f= open(outFpath,"w+")

    tot_obs=np.shape(sample)[0]*np.shape(sample)[1]
    uq_day = np.shape(sample)[0]
    stn = list(sample.columns)
    f.write('{:02.0f}\n'.format(uq_day))
    for j in range(uq_day):
        d=sample.index[j].day
        m=sample.index[j].month
        y=sample.index[j].year
        date = str(y)+' '+str(m)+' '+str(d)
        stn_count = np.shape(sample)[1]
        f.write(date+' \n')
        f.write(str(stn_count)+' \n')
        ids = 100
        for k in stn:
            ids = ids + 1 
            x = new.easting.values[new.code.values == k][0]
            y = new.northing.values[new.code.values == k][0]
            swe = sample[k][j]
            f.write('{:3.0f}\t'.format(ids)+'{:10.0f}\t'.format(x)+'{:10.0f}\t'.format(y)+'{:3.2f}\n'.format(swe))
    f.close() 

# In[ ]:


# function to edit SnowModel Files other than .par
# for assim - have to adjust .inc file to specify # of obs being assimilated
def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


# In[ ]:





# In[ ]:


#function to make SM assim file based on selected landscape characteristic
#var can be 'elev','slope','tc','delta_day','M', 'lc', 'aspect'

## need to add if var == 'all'
def SMassim_ensemble(gdf,var):
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
    import datetime
    gdf['dt'] = pd.to_datetime(gdf['timestamp'], format='%Y-%m-%dT%H:%M:%S')
    gdf['dt'] = pd.to_datetime(gdf['dt']).dt.date
    today = datetime.date.today()
    hoy = today.strftime("%Y-%m-%d")
    path = SMpath+domain+'/'+'assim_'+var+'_'+hoy
    #create directory with initiation date for ensemble if it doesn't exist
    get_ipython().system('mkdir -p $path')
    outFpath = SMpath+'swe_assim/swe_obs_test.dat'
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
        get_ipython().run_line_magic('cd', '$assimPath')
        #move swed.gdat file x
        oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
        nSWEpath = path + '/cso_all'+'_swed.gdat'
        get_ipython().system('mv $oSWEpath $nSWEpath    ')
    elif var == 'elev':
        edges = np.histogram_bin_edges(gdf.dem_elev,bins=5, range=(gdf.dem_elev.min(),gdf.dem_elev.max()))
        print('edges:',edges)
        labs = np.arange(0,len(edges)-1,1)
        print('labels:',labs)
        bins = pd.cut(gdf['dem_elev'], edges,labels=labs)
        gdf['elev_bin']=bins    
        for i in range(len(labs)):
            new = gdf[gdf.elev_bin == i]
            make_SMassim_file(new,outFpath)
            #edit .inc file
            replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(new)+1)+')\n')
            #compile SM
            get_ipython().run_line_magic('cd', '$codepath')
            get_ipython().system(' ./compile_snowmodel.script')
            #run snowmodel 
            get_ipython().run_line_magic('cd', '$SMpath')
            get_ipython().system(' ./snowmodel')
            get_ipython().run_line_magic('cd', '$assimPath')
            #move swed.gdat file 
            oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
            nSWEpath = path + '/cso_elev_'+str(i)+'_swed.gdat'
            get_ipython().system('mv $oSWEpath $nSWEpath    ')
    elif var == 'slope':
        edges = np.histogram_bin_edges(gdf.slope,bins=5, range=(gdf.slope.min(),gdf.slope.max()))
        print('edges:',edges)
        labs = np.arange(0,len(edges)-1,1)
        print('labels:',labs)
        bins = pd.cut(gdf['slope'], edges,labels=labs)
        gdf['slope_bin']=bins    
        for i in range(len(labs)):
            new = gdf[gdf.slope_bin == i]
            make_SMassim_file(new,outFpath)
            #edit .inc file
            replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(new)+1)+')\n')
            #compile SM
            get_ipython().run_line_magic('cd', '$codepath')
            get_ipython().system(' ./compile_snowmodel.script')
            #run snowmodel 
            get_ipython().run_line_magic('cd', '$SMpath')
            get_ipython().system(' ./snowmodel')
            get_ipython().run_line_magic('cd', '$assimPath')
            #move swed.gdat file 
            oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
            nSWEpath = path + '/cso_slope_'+str(i)+'_swed.gdat'
            get_ipython().system('mv $oSWEpath $nSWEpath    ')
    elif var == 'tc':
        edges = np.histogram_bin_edges(gdf.tc,bins=5, range=(gdf.tc.min(),gdf.tc.max()))
        print('edges:',edges)
        labs = np.arange(0,len(edges)-1,1)
        print('labels:',labs)
        bins = pd.cut(gdf['tc'], edges,labels=labs)
        gdf['tc_bin']=bins    
        for i in range(len(labs)):
            new = gdf[gdf.tc_bin == i]
            make_SMassim_file(new,outFpath)
            #edit .inc file
            replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(new)+1)+')\n')
            #compile SM
            get_ipython().run_line_magic('cd', '$codepath')
            get_ipython().system(' ./compile_snowmodel.script')
            #run snowmodel 
            get_ipython().run_line_magic('cd', '$SMpath')
            get_ipython().system(' ./snowmodel')
            get_ipython().run_line_magic('cd', '$assimPath')
            #move swed.gdat file 
            oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
            nSWEpath = path + '/cso_tc_'+str(i)+'_swed.gdat'
            get_ipython().system('mv $oSWEpath $nSWEpath  ')
    elif var == 'delta_day':
        import datetime
        gdf = gdf.sort_values(by='dt',ascending=True)
        gdf = gdf.reset_index(drop=True)
        Delta = [3,5,7,10]
        for z in range(len(Delta)):
            delta = Delta[z]
            idx = [0]
            st = gdf.dt[0]
            for i in range(1,len(gdf)-1):
                date = gdf.dt.iloc[i]
                gap = (date - st).days
                if gap<=delta:
                    continue 
                else:
                    idx.append(i)
                    st = date
            new = gdf[gdf.index.isin(idx)]
            make_SMassim_file(new,outFpath)
            #edit .inc file
            replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(new)+1)+')\n')
            #compile SM
            get_ipython().run_line_magic('cd', '$codepath')
            get_ipython().system(' ./compile_snowmodel.script')
            #run snowmodel 
            get_ipython().run_line_magic('cd', '$SMpath')
            get_ipython().system(' ./snowmodel')
            get_ipython().run_line_magic('cd', '$assimPath')
            #move swed.gdat file 
            oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
            nSWEpath = path + '/cso_day_delta'+str(delta)+'_swed.gdat'
            get_ipython().system('mv $oSWEpath $nSWEpath               ')
    else: #works for 'M', 'lc', 'aspect'
        uq = np.unique(gdf[var])
        for i in range(len(uq)):
            new = gdf[gdf[var] == uq[i]]
            make_SMassim_file(new,outFpath)
            #edit .inc file
            replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(new)+1)+')\n')
            #compile SM
            get_ipython().run_line_magic('cd', '$codepath')
            get_ipython().system(' ./compile_snowmodel.script')
            #run snowmodel 
            get_ipython().run_line_magic('cd', '$SMpath')
            get_ipython().system(' ./snowmodel')
            get_ipython().run_line_magic('cd', '$assimPath')
            #move swed.gdat file 
            oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
            nSWEpath = path + '/cso_'+var+'_'+str(uq[i])+'_swed.gdat'
            get_ipython().system('mv $oSWEpath $nSWEpath    ')


# In[ ]:


#function to make SM assim file based on selected landscape characteristic
#var can be 'elev','slope','tc','delta_day','M', 'lc', 'aspect'

## need to add if var == 'all'
def SMassim_ensemble_snotel(gdf,snotel_gdf,swes,var):
    '''
    gdf: this is the geodataframe containing all snotel stations
    swes: this is a dataframe containing all snotel swe
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
    import datetime
    today = datetime.date.today()
    hoy = today.strftime("%Y-%m-%d")
    path = SMpath+domain+'/'+'assim_'+var+'_'+hoy
    #create directory with initiation date for ensemble if it doesn't exist
    get_ipython().system('mkdir -p $path')
    outFpath = SMpath+'swe_assim/swe_obs_test.dat'
    if var == 'all':
        new = snotel_gdf
        sample = swes
        make_SMassim_file_snotel(sample,new,outFpath)
        #edit .inc file
        replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(sample)+1)+')\n')
        #compile SM 
        get_ipython().run_line_magic('cd', '$codepath')
        get_ipython().system(' ./compile_snowmodel.script')
        #run snowmodel 
        get_ipython().run_line_magic('cd', '$SMpath')
        get_ipython().system(' ./snowmodel')
        get_ipython().run_line_magic('cd', '$assimPath')
        #move swed.gdat file 
        oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
        nSWEpath = path + '/snotel_all'+'_swed.gdat'
        get_ipython().system('mv $oSWEpath $nSWEpath    ')
        
    elif var == 'elev':
        edges = np.histogram_bin_edges(gdf.dem_elev,bins=5, range=(gdf.dem_elev.min(),gdf.dem_elev.max()))
        print('edges:',edges)
        labs = np.arange(0,len(edges)-1,1)
        print('labels:',labs)
        bins = pd.cut(snotel_gdf['dem_elev'], edges,labels=labs)
        snotel_gdf['elev_bin']=bins 
        for lab in labs:
            new = snotel_gdf[snotel_gdf.elev_bin == lab]
            if len(new) == 0:
                continue
            else:
                sample = swes[np.intersect1d(swes.columns, new.code.values)] 
                make_SMassim_file_snotel(sample,new,outFpath)
                #edit .inc file
                replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(sample)+1)+')\n')
                #compile SM
                get_ipython().run_line_magic('cd', '$codepath')
                get_ipython().system(' ./compile_snowmodel.script')
                #run snowmodel 
                get_ipython().run_line_magic('cd', '$SMpath')
                get_ipython().system(' ./snowmodel')
                get_ipython().run_line_magic('cd', '$assimPath')
                #move swed.gdat file 
                oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
                nSWEpath = path + '/snotel_elev_'+str(lab)+'_swed.gdat'
                get_ipython().system('mv $oSWEpath $nSWEpath                  ')

    elif var == 'slope':
        edges = np.histogram_bin_edges(gdf.slope,bins=5, range=(gdf.slope.min(),gdf.slope.max()))
        print('edges:',edges)
        labs = np.arange(0,len(edges)-1,1)
        print('labels:',labs)
        bins = pd.cut(snotel_gdf['slope'], edges,labels=labs)
        snotel_gdf['slope_bin']=bins 
        for lab in labs:
            new = snotel_gdf[snotel_gdf.slope_bin == lab]
            if len(new) == 0:
                continue
            else:
                sample = swes[np.intersect1d(swes.columns, new.code.values)] 
                make_SMassim_file_snotel(sample,new,outFpath)
                #edit .inc file
                replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(sample)+1)+')\n')
                #compile SM
                get_ipython().run_line_magic('cd', '$codepath')
                get_ipython().system(' ./compile_snowmodel.script')
                #run snowmodel 
                get_ipython().run_line_magic('cd', '$SMpath')
                get_ipython().system(' ./snowmodel')
                get_ipython().run_line_magic('cd', '$assimPath')
                #move swed.gdat file 
                oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
                nSWEpath = path + '/snotel_slope_'+str(lab)+'_swed.gdat'
                get_ipython().system('mv $oSWEpath $nSWEpath                                 ')

    elif var == 'tc':
        edges = np.histogram_bin_edges(gdf.tc,bins=5, range=(gdf.tc.min(),gdf.tc.max()))
        print('edges:',edges)
        labs = np.arange(0,len(edges)-1,1)
        print('labels:',labs)
        bins = pd.cut(snotel_gdf['tc'], edges,labels=labs)
        snotel_gdf['tc_bin']=bins 
        for lab in labs:
            new = snotel_gdf[snotel_gdf.tc_bin == lab]
            if len(new) == 0:
                continue
            else:
                sample = swes[np.intersect1d(swes.columns, new.code.values)] 
                make_SMassim_file_snotel(sample,new,outFpath)
                #edit .inc file
                replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(sample)+1)+')\n')
                #compile SM
                get_ipython().run_line_magic('cd', '$codepath')
                get_ipython().system(' ./compile_snowmodel.script')
                #run snowmodel 
                get_ipython().run_line_magic('cd', '$SMpath')
                get_ipython().system(' ./snowmodel')
                get_ipython().run_line_magic('cd', '$assimPath')
                #move swed.gdat file 
                oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
                nSWEpath = path + '/snotel_tc_'+str(lab)+'_swed.gdat'
                get_ipython().system('mv $oSWEpath $nSWEpath     ')
                    
    elif var == 'delta_day':
        new = snotel_gdf
        Delta = [3,5,7,10]
        for dels in Delta:
            sample = swes.iloc[::dels,:]
            make_SMassim_file_snotel(sample,new,outFpath)
            #edit .inc file
            replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(sample)+1)+')\n')
            #compile SM
            get_ipython().run_line_magic('cd', '$codepath')
            get_ipython().system(' ./compile_snowmodel.script')
            #run snowmodel 
            get_ipython().run_line_magic('cd', '$SMpath')
            get_ipython().system(' ./snowmodel')
            get_ipython().run_line_magic('cd', '$assimPath')
            #move swed.gdat file 
            oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
            nSWEpath = path + '/snotel_day_delta'+str(dels)+'_swed.gdat'
            get_ipython().system('mv $oSWEpath $nSWEpath  ')
   
    elif var == 'M':
        new = snotel_gdf
        mo = np.unique(swes.index.month)
        for m in mo:
            sample = swes[swes.index.month == m]
            make_SMassim_file_snotel(sample,new,outFpath)
            #edit .inc file
            replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(sample)+1)+')\n')
            #compile SM
            get_ipython().run_line_magic('cd', '$codepath')
            get_ipython().system(' ./compile_snowmodel.script')
            #run snowmodel 
            get_ipython().run_line_magic('cd', '$SMpath')
            get_ipython().system(' ./snowmodel')
            get_ipython().run_line_magic('cd', '$assimPath')
            #move swed.gdat file 
            oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
            nSWEpath = path + '/snotel_day_delta'+str(dels)+'_swed.gdat'
            get_ipython().system('mv $oSWEpath $nSWEpath  ')              
    else: #works for 'lc', 'aspect'
        uq = np.unique(gdf[var])
        for lab in uq:
            new = snotel_gdf[snotel_gdf[var] == lab]
            if len(new) == 0:
                continue
            else:
                sample = swes[np.intersect1d(swes.columns, new.code.values)] 
                make_SMassim_file_snotel(sample,new,outFpath)
                #edit .inc file
                replace_line(incFile, 30, '      parameter (max_obs_dates='+str(len(sample)+1)+')\n')
                #compile SM
                get_ipython().run_line_magic('cd', '$codepath')
                get_ipython().system(' ./compile_snowmodel.script')
                #run snowmodel 
                get_ipython().run_line_magic('cd', '$SMpath')
                get_ipython().system(' ./snowmodel')
                get_ipython().run_line_magic('cd', '$assimPath')
                #move swed.gdat file 
                oSWEpath = SMpath + 'outputs/wi_assim/swed.gdat'
                nSWEpath = path + '/snotel_'+var+'_'+str(lab)+'_swed.gdat'
                get_ipython().system('mv $oSWEpath $nSWEpath     ')


# In[ ]:


# function to save SM outputs from each ensemble into one .nc at the evaluation sites

