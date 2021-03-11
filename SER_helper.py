#--------------------------------------------- Packages ---------------------------------------------#
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import os
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
import psycopg2 as ps
import time
import requests
import streamlit as st
import re
from dateutil.relativedelta import relativedelta
import calendar

#--------------------------------------------- Database Query ---------------------------------------------#
customer_eudb = ''
customer_usdb = ''
device_db = ''

class Databases:
        def __init__(self, url, query):
            self.custConn = ps.connect(url)
            self.custCurs = self.custConn.cursor()
            self.devConn = ps.connect(url)
            self.devCurs = self.devConn.cursor()
            self.custCurs.execute(query)
            self.tenantList = [item for item in self.custCurs.fetchall()]
        def destroy(self):
            self.custConn.close()
            self.devConn.close()

#--------------------------------------------- Extra Functions ---------------------------------------------#
def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def clean(list_of_lists):
    """Flatten list of lists and remove 'nan' values
    """
    flat_list = list(flatten(list_of_lists))
    clean_list = [item for item in flat_list if str(item) != 'nan']
    return clean_list

#--------------------------------------------- Locations ---------------------------------------------#
tenants_needing_locations = ['madrid', 'borough-of-hillingdon', 'borough-of-hounslow', 'borough-of-ealing', 'borough-of-richmond', 'trondheim']

def overpassQuery(location_load_status, metadata, coordinates, admin_level):
    
    load_status = metadata[metadata['coordinates'] == coordinates].index.values[0] + 1
    try:
        location_load_status.info(f"Finding Location for Device {load_status} of {len(metadata)}")
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"[out:json];is_in({coordinates});area._[admin_level = '{admin_level}'];out;"
        data = requests.get(overpass_url, params={'data': overpass_query})
        return data.json()['elements'][0]['tags']['name']
    except:
        time.sleep(15)
        overpassQuery(location_load_status, metadata, coordinates, admin_level)

def madrid_locations(metadata):
    location_load_status = st.empty()
    if os.path.exists('madrid_locations.csv'):
        locations_file = pd.read_csv('madrid_locations.csv', sep = ",")
        metadata = pd.merge(metadata, locations_file[['serial_no','location']],on='serial_no', how='left')
        metadata['location'] = metadata.apply(lambda x: overpassQuery(location_load_status, metadata, x['coordinates'],9) if pd.isnull(x['location']) else x['location'],axis = 1)
        output_df = pd.concat([metadata['serial_no'], metadata['location']], axis = 1)
        output_df.to_csv('madrid_locations.csv', index = False, header = True)
    else:
        metadata['location'] = metadata.apply(lambda x: overpassQuery(location_load_status, metadata, x['coordinates'], 9), axis=1)
        metadata['location'] = metadata.apply(lambda x: overpassQuery(location_load_status, metadata, x['coordinates'], 9) if pd.isna(x['location']) else x['location'], axis = 1) # Extra contingency for None values returned by query
        output_df = pd.concat([metadata['serial_no'], metadata['location']], axis = 1)
        output_df.to_csv('madrid_locations.csv', index = False, header = True)
    location_load_status.success("Locations Obtained")
    return metadata   

def trondheim_locations(metadata):
    location_load_status = st.empty()
    if os.path.exists('trondheim_locations.csv'):
        locations_file = pd.read_csv('trondheim_locations.csv', sep = ",")
        metadata = pd.merge(metadata, locations_file[['serial_no','location']],on='serial_no', how='left')
        metadata['location'] = metadata.apply(lambda x: overpassQuery(location_load_status, metadata, x['coordinates'],9) if pd.isnull(x['location']) else x['location'],axis = 1)
        output_df = pd.concat([metadata['serial_no'], metadata['location']], axis = 1)
        output_df.to_csv('trondheim_locations.csv', index = False, header = True)
    else:
        metadata['location'] = metadata.apply(lambda x: overpassQuery(location_load_status, metadata, x['coordinates'], 9), axis=1)
        metadata['location'] = metadata.apply(lambda x: overpassQuery(location_load_status, metadata, x['coordinates'], 9) if pd.isna(x['location']) else x['location'], axis = 1) # Extra contingency for None values returned by query
        output_df = pd.concat([metadata['serial_no'], metadata['location']], axis = 1)
        output_df.to_csv('trondheim_locations.csv', index = False, header = True)
    location_load_status.success("Locations Obtained")
    return metadata


def borough_of_hillingdon_locations(metadata):
    """Gather location data from device names and clean up local errors
    """
    location_status = st.empty()
    location_status.info("Obtaining Locations")
    # make locations friendly name and split at '-'
    metadata['location'] = metadata['container_name'].apply(lambda x: x.split("-", 1)[0])
    # remove gNUM and rNUM from location name
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"(g|r)[0-9]+", "", x))
    # remove NUM from location name
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"[0-9]+", "",x))
    # remove 'new' (regarding '.new' and 'new.') from location name
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"new+", "", x))
    # remove '.' (regarding '.new' and 'new.') from location name
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"\.+", "", x))
    # remove spaces from start of location name (eg. ' Clyne')
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"^ ", "", x))
    # remove spaces from end of location name (eg. 'Clyne ')
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r" $", "", x))
    # local errors
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("welbeck", "wellbeck", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("sutcliffe", "suttcliffe", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("aukland", "auckland", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("aspeng", "aspen", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("devonshie", "devonshire", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("greeatfields", "greatfields", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("greenwsy", "greenway", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("barnill", "barnhill", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("clynes", "clyne", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub("clyne", "clynes", x))
    # capitalize first letter
    metadata['location'] = metadata['location'].apply(lambda x: x.title())
    location_status.success("Locations Obtained")
    return metadata

def borough_of_hounslow_locations(metadata):
    """Gather location data from device names and clean up local errors
    """
    location_status = st.empty()
    location_status.info("Obtaining Locations")
    # make locations friendly name and split at '-'
    metadata['location'] = metadata['container_name'].apply(lambda x: x.split("-", 1)[0])
    # remove gNUM and rNUM from location name
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"(g|r)[0-9]+", "", x))
    # remove NUM from location name
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"[0-9]+", "",x))
    # remove 'f' if last character
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"f$", "", x))
    # remove space if last character
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r" $", "", x))
    # remove space if first character
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"^ ", "", x))
    # local errors
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"haverfieldf", "haverfield", x))
    # capitalize first letter
    metadata['location'] = metadata['location'].apply(lambda x: x.title())
    location_status.success("Locations Obtained")
    return metadata

def borough_of_ealing_locations(metadata):
    """Gather location data from device names and clean up local errors
    """
    location_status = st.empty()
    location_status.info("Obtaining Locations")
    # make locations friendly name and split at '-'
    metadata['location'] = metadata['container_name'].apply(lambda x: x.split("-", 1)[0])
    # local errors
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"stcatherines", "st catherines", x))
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"stcathrines", "st catherines", x))
    # capitalize first letter
    metadata['location'] = metadata['location'].apply(lambda x: x.title()) 
    location_status.success("Locations Obtained")
    return metadata

def borough_of_richmond_locations(metadata):
    """Gather location data from device names and clean up local errors
    """
    location_status = st.empty()
    location_status.info("Obtaining Locations")
    # make locations friendly name and split at '-'
    metadata['location'] = metadata['container_name'].apply(lambda x: x.split("-", 1)[0])
    # remove NUM+LETTER from location name
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"[0-9][a-z]", "",x))
    # remove NUM from location name
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"[0-9]", "", x))
    # remove '.' from location name
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r"\.+", "", x))
    # remove spaces from end of location name (eg. 'Clyne ')
    metadata['location'] = metadata['location'].apply(lambda x: re.sub(r" $", "", x))
    # capitalize first letter
    metadata['location'] = metadata['location'].apply(lambda x: x.title())
    location_status.success("Locations Obtained")
    return metadata

#--------------------------------------------- Adjust Services ---------------------------------------------#
def round_to_second(timestamp):
    return pd.Timestamp(timestamp).round('s')

def service_adjuster(services, metadata, serial_no, original_service_ts, index, readings_dict):
    readings = readings_dict[serial_no]
    readings = readings[original_service_ts - timedelta(hours = 1):original_service_ts + timedelta(hours = 1)]
    if len(readings) > 0:
        highest_value_ts = readings['fill'].idxmax()
        highest_value = readings.at[highest_value_ts, 'fill']
        
        lowest_value_ts = readings['fill'].idxmin()
        lowest_value = readings.at[lowest_value_ts, 'fill']
        return highest_value * 100, lowest_value * 100, round_to_second(lowest_value_ts)
    else:
        return np.nan, np.nan, np.nan

#--------------------------------------------- Timestamp Calculations ---------------------------------------------#
def previous_service_timestamps(serial_no, services):
    device_services = services[services.serial_no == serial_no][:]
    device_services_idxs = device_services.index.values
    idx_dict = dict(zip(device_services_idxs, range(0,len(device_services_idxs))))
    output_series = device_services.apply(lambda x: services.at[device_services_idxs[idx_dict[x.name]-1], 'adjusted_service_ts'] if idx_dict[x.name] != 0 else None, axis = 1)
    return output_series

def previous_after_service_values(serial_no, services):
    device_services = services[services.serial_no == serial_no][:]
    device_services_idxs = device_services.index.values
    idx_dict = dict(zip(device_services_idxs, range(0,len(device_services_idxs))))
    output_series = device_services.apply(lambda x: services.at[device_services_idxs[idx_dict[x.name]-1], 'fill_after_serviced'] if idx_dict[x.name] != 0 else None, axis = 1)
    return output_series

#--------------------------------------------- Query Functions ---------------------------------------------#
def tenant_names(CONTINENT):
    if CONTINENT == 'EU':
        db = customer_eudb
    elif CONTINENT == 'US':
        db = customer_usdb
    else:
        pass

    query = """
    SELECT DISTINCT sub_tenant_name
    from sub_tenants
    ORDER BY sub_tenant_name asc
    """
        
    db_instance = Databases(db, query)
    output = db_instance.tenantList
    del db_instance
    return list(sum(output, ()))

def metadataQuery(TENANT, CONTINENT):
    if CONTINENT == 'EU':
        db = customer_eudb
    elif CONTINENT == 'US':
        db = customer_usdb
    else:
        pass

    query = f"""
    SELECT DISTINCT ON (container_devices.serial_no)
        containers.friendly_name AS container_name,
        container_content_types.friendly_name AS content_type,
        container_types.friendly_name AS container_type,
        container_devices.serial_no,
        containers.id AS container_id,
        container_device_configs.installation_config -> 'device_offset' AS device_offset ,
        container_device_configs.installation_config -> 'device_total_distance' AS total_distance,
        container_types.volume AS capacity_volume,
        container_types.width,
        containers.latitude,
        containers.longitude
    FROM
        containers
    JOIN 
        container_devices ON containers.id = container_devices.container_id
    JOIN    
        container_types ON containers.container_type_id = container_types.id
    JOIN 
        container_content_types ON containers.content_type_id = container_content_types.id
    JOIN 
        container_device_configs ON container_device_configs.serial_no = container_devices.serial_no
    WHERE 
        containers.tenant IN ('{TENANT}')
    ORDER BY
        container_devices.serial_no, container_device_configs.created_at DESC
    """
    db_instance = Databases(db, query)
    response = db_instance.tenantList
    del db_instance
    
    df = pd.DataFrame(response, columns = ["container_name", "content_type", "container_type", "serial_no", 
                                            "container_id", "device_offset", "total_distance", "capacity_volume",
                                            "width", "latitude", "longitude"])
    
    df['coordinates'] = df['latitude'].astype(str) + ', ' + df['longitude'].astype(str)
    df['capacity_height'] = df['total_distance'] - df['device_offset']
    df['capacity_volume'] = df.apply(lambda x: (x['width'] * x['width'] * x['capacity_height'])/10**6 
                                    if (pd.isna(x['capacity_volume']) and not pd.isna(x['width']))
                                    else (0 if pd.isna(x['width']) else x['capacity_volume']), axis = 1)
    
    return df

def servicesQuery(TENANT, CONTINENT, START, END):
    if CONTINENT == 'EU':
        db = customer_eudb
    elif CONTINENT == 'US':
        db = customer_usdb
    else:
        pass
    
    query = f"""
    SELECT 
        csts.detected_by,
        container_devices.serial_no,
        csts.fill_level_percentage_when_serviced AS fill_after_serviced,
        EXTRACT(epoch FROM csts.empty_timestamp) AS original_service_ts
    FROM 
        container_service_time_series csts
    JOIN 
        container_devices on csts.container_id = container_devices.container_id
    WHERE 
        csts.tenant = '{TENANT}'
    AND 
        EXTRACT(epoch FROM csts.empty_timestamp) BETWEEN '{START.timestamp()}' AND '{END.timestamp()}'
    ORDER BY 
        csts.empty_timestamp ASC
    """
    db_instance = Databases(db, query)
    response = db_instance.tenantList
    del db_instance
    
    df = pd.DataFrame(response, columns = ["detected_by", "serial_no", "fill_after_serviced", "original_service_ts"])
    df['original_service_ts'] = df['original_service_ts'].map(dt.fromtimestamp)
    
    return df

def readingsQueryGenerator(services, serial_no):
    query_condition = ""
    condition_id = 1
    for i in services[services.serial_no == serial_no].original_service_ts:
        under = (i - timedelta(hours = 2)).timestamp()
        over = i.timestamp()
        if condition_id == 1:
            query_condition += f"EXTRACT(epoch FROM ds.sampling_timestamp) >= '{under}' AND EXTRACT(epoch FROM ds.sampling_timestamp) <= '{over}'\n"
        else:
            query_condition += f"OR (EXTRACT(epoch FROM ds.sampling_timestamp) >= '{under}' AND EXTRACT(epoch FROM ds.sampling_timestamp) <= '{over}')\n"
        condition_id += 1
    return query_condition

def readingsQuery(serial_no, services, metadata):
    query = f"""
    SELECT 
        DISTINCT ds.id as reading_id,
        EXTRACT (epoch FROM ds.sampling_timestamp) AS reading_ts,
        ds.device_id as serial_no,
        ds.single_distance
    FROM 
        device_samples ds
    WHERE ds.device_id = '{serial_no}'
    AND ({readingsQueryGenerator(services, serial_no)})
    ORDER BY
        reading_ts ASC
    """
    db_instance = Databases(device_db, query)
    response = db_instance.tenantList
    del db_instance

    readings = pd.DataFrame(response, columns = ["reading_id", "reading_ts", "serial_no", "single_distance"])

    # Drop duplicates if they exist
    readings.drop_duplicates(subset = 'reading_ts', keep = 'first', inplace = True)
    readings.reset_index(drop = True, inplace = True)

    readings['reading_ts'] = readings['reading_ts'].apply(dt.fromtimestamp)
    device_offset = metadata[metadata['serial_no'] == serial_no].device_offset.values[0]
    height = metadata[metadata.serial_no == serial_no].capacity_height.values[0]

    readings['fill'] = 1 - (readings['single_distance'] - device_offset)/height
    readings['fill'] = readings.apply(lambda x: 1.0 if x.fill > 1.0 else (0.0 if x.fill < 0.0 else x.fill), axis = 1)

    readings.set_index('reading_ts', inplace = True)

    return readings

#--------------------------------------------- Reporting Functions ---------------------------------------------#
def create_readme_sheet(TENANT, START, END, metadata, services):
    readme_list = [
        ('Tenant', TENANT),
        ('Number of Containers', len(metadata)),
        ('Number of Containers with Services', services['serial_no'].nunique()),
        ('Number of Services', len(services)),
        ('From', START.date()),
        ('To', END.date()),
        ('Report Generation Date', dt.now().date())
    ]
    
    readme_sheet_list = []
    for tup in readme_list:
        readme_sheet_list.append({"Entity": tup[0], "Value": tup[1]})
    return pd.DataFrame(readme_sheet_list)

def container_level_report_sheet(services):
    report_list = []
    for i in services.serial_no.unique():
        report_list.append(
                            (
                            services[services.serial_no == i].container_name.values[0],
                            i,
                            services[services.serial_no == i].container_id.values[0],
                            services[services.serial_no == i].content_type.values[0],
                            len(services[services.serial_no == i]),
                            sum(services[services.serial_no == i].volume_emptied),
                            np.mean(clean(services[services.serial_no == i].fill_before_serviced)),
                            np.mean(clean(services[services.serial_no == i].time_diff_days)),
                            np.mean(clean(services[services.serial_no == i].fill_rate_per_day)),
                            np.mean(clean(services[services.serial_no == i].days_to_full)),
                            np.mean(clean(services[services.serial_no == i].days_overflowing))
                            )
                          )

    report_sheet_list = []
    for tup in report_list:

        report_sheet_list.append({'Container Name': tup[0], 'Serial Number': tup[1], "Container ID": tup[2], 
                                  "Content Type":tup[3], "Number of Services": tup[4], "Total Volume Emptied [L]": tup[5], "Avg Fill at Service": tup[6],
                                  "Avg Days Between Services": tup[7], "Avg Daily Fill Rate": tup[8], "Avg Days to Full": tup[9],
                                  "Avg Days Overflowing": tup[10]})
    return pd.DataFrame(report_sheet_list)

def create_overall_report_sheet(metadata, services):        
    report_list = [
        ('Number of Containers', len(metadata)),
        ('Number of Services', len(services)),
        ('Total Volume Emptied [L]', sum(services['volume_emptied'])),
        ('Avg Fill at Service', np.mean(services['fill_before_serviced']/100)),
        ('Avg Daily Fill Rate', np.mean(clean(services['fill_rate_per_day']))),
        ('Avg Days Between Services', np.mean(clean(services['time_diff_days']))),
        ('Avg Days to Full', np.mean(clean(services['days_to_full']))),
        ('Avg Days Overflowing', np.mean(clean(services['days_overflowing'])))
    ]
    
    report_sheet_list = []
    for tup in report_list:
        report_sheet_list.append({"Entity": tup[0], "Value": tup[1]})
    return pd.DataFrame(report_sheet_list)

def create_key_report_sheet(metadata, services, key):
    report_list = []
    for i in metadata[key].unique():
        report_list.append(
                            (i, len(metadata[metadata[key] == i]), 
                            len(services[services[key] == i]), 
                            sum(services[services[key] == i]['volume_emptied']),
                            np.mean(services[services[key] == i]['fill_before_serviced']/100),
                            np.mean(clean(services[services[key] == i]['time_diff_days'])), 
                            np.mean(clean(services[services[key] == i]['fill_rate_per_day'])),
                            np.mean(clean(services[services[key] == i]['days_to_full'])),
                            np.mean(clean(services[services[key] == i]['days_overflowing'])))
                          )
        
    report_sheet_list = []
    for tup in report_list:
        key = key.replace("_","-").title()
            
        report_sheet_list.append({key: tup[0], 'Number of Containers': tup[1], 'Number of Services': tup[2],
                                    'Total Volume Emptied': tup[3], 'Avg Fill at Service': tup[4], 
                                    'Avg Days Between Services': tup[5], 'Avg Daily Fill Rate': tup[6],
                                    'Avg Days to Full': tup[7], 
                                    'Avg Days Overflowing': tup[8]})
    return pd.DataFrame(report_sheet_list)

def services_chart_generator(services):
    id_columns = ['container_id', 'serial_no', 'container_name', 'content_type', 'container_type']
    week_columns = ['week_start_date', 'week_num']

    services_agg = (
        services.groupby(id_columns + week_columns)
        .agg(
            num_services = ('fill_before_serviced', 'count'),
            mean_fill_percentage = ('fill_before_serviced', 'mean')
        )
        .reset_index()
        .sort_values('week_start_date')
    )

    services_agg.mean_fill_percentage = round(services_agg.mean_fill_percentage,2)

    services_pivot_df = services_agg.pivot(
        index = id_columns,
        columns = week_columns,
        values = ['mean_fill_percentage', 'num_services']
    )

    return services_pivot_df

def top_10(indexes, key, input_df):
    output_df = pd.DataFrame({"Container Name": list(input_df['Container Name'][indexes]), 
                     key: list(input_df[key][indexes]),
                     "Waste Type": list(input_df['Content Type'][indexes])})
    return output_df

#--------------------------------------------- Share Functions ---------------------------------------------#
def share_of_services_sheet(TENANT, services):
    if TENANT == 'trondheim':
        ranges = [25,50,80]
    else:
        ranges = [50,80]    
        
    def percentage_services(low, high, services):
        return len(services[(services.fill_before_serviced >= low) & (services.fill_before_serviced < high)])/len(services)
        
    if len(services) > 0:
        diffs = pd.Series([[ranges[i], ranges[i + 1]] for i in range(len(ranges) - 1)])
        output = []
         
        output.append({f"<{ranges[0]}%":percentage_services(0, ranges[0], services)})
        diffs.apply(lambda row: output.append({f"{row[0]}%-{row[1]}%": percentage_services(row[0], row[1], services)}))
        output.append({f">{ranges[-1]}%": percentage_services(ranges[-1], 101, services)})
        
    else:
        output.append({f"<{ranges[0]}%":""})
        diffs.apply(lambda row: output.append({f"{row[0]}%-{row[1]}%": ""}))
        output.append({f">{ranges[-1]}%": ""})

    final_output = []
    for i in range(len(output)):
        final_output += list(output[i].items())
    
    final_output = [dict(final_output)]
    
    return pd.DataFrame(final_output)

def key_share_of_services_helper(key, value, TENANT, services):
    key_df = share_of_services_sheet(TENANT, services[services[key] == value])
    key_df.insert(loc=0, column = key.replace("_","-").title(), value="")
    key_df.at[0, key.replace("_","-").title()] = value
    return key_df

def key_share_of_services(TENANT, services, key):
    key_df_list = pd.Series(services[key].unique()).apply(lambda x: key_share_of_services_helper(key, x, TENANT, services))
    key_df = pd.concat(list(key_df_list), ignore_index = True)
    return key_df

#--------------------------------------------- Tendency Functions ---------------------------------------------#
month_names_in_a_year = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
month_nums_in_a_year = {y:x for x,y in month_names_in_a_year.items()}

def tendency_share_of_services(TENANT, START, END, services):
    years = pd.date_range(START,END, freq='MS').strftime("%Y").tolist()
    years = list(map(int, years)) 

    months = pd.date_range(START,END, freq='MS').strftime("%m").tolist()
    months = list(map(int, months)) 

    months_list = list(zip(years, months))
    months_list = months_list[:-1]

    monthly_shares_list = []
    for m in months_list:
        services_in_period = services[(dt(m[0],m[1],1) <= services.adjusted_service_ts) & (services.adjusted_service_ts < (dt(m[0],m[1],calendar.monthrange(m[0], m[1])[1]) + timedelta(days=1)))]
        monthly_share_df = share_of_services_sheet(TENANT, services_in_period)
        monthly_share_df.insert(loc=0, column = 'Period', value="")
        monthly_share_df.insert(loc=1, column = 'Overall Average Fill Level at Service', value = "")
        
        monthly_share_df.at[0, 'Period'] = f'{month_nums_in_a_year[m[1]]} {m[0]}'
        monthly_share_df.at[0, 'Overall Average Fill Level at Service'] = np.mean(services_in_period['fill_before_serviced']/100)
      
        monthly_shares_list.append(monthly_share_df)

    output = pd.concat(monthly_shares_list, ignore_index = True)
    return output

#--------------------------------------------- Recommendation Functions ---------------------------------------------#

def recommendations(df, index):
    if df.at[index, 'Avg Days Overflowing'] < 0:
        services_rec = 'REDUCE'
        container_rec = None
    elif df.at[index, 'Avg Days Overflowing'] > (2/24):
        if df.at[index, 'Avg Days Between Services'] < 2:
            container_rec = 'INCREASE'
            services_rec  = None
        else:
            services_rec  = 'INCREASE'
            container_rec = None
    else:
        services_rec  = None
        container_rec = None
    
    return services_rec, container_rec
        
