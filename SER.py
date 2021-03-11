import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import streamlit as st
import SER_helper as sh
import SER_fig_generator as fg
import os
import io
import base64
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title='SER', page_icon = 'favicon.ico')

st.sidebar.markdown(
    """
        **Enter Database Connection Strings:**
    """
)
sh.customer_eudb = st.sidebar.text_input("EU Database")
sh.customer_usdb = st.sidebar.text_input("US Database")
sh.device_db = st.sidebar.text_input("Device Database")

st.markdown(
    """
<style>

div.row-widget.stRadio > div {
    flex-direction: row;
    align-items: stretch;
}

div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
    background-color: #294269;
}

.stButton>button {
    color: #294269;
}

</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
        Choose tenant and period of services.

        **NOTE:
        Services needs to have been added and/or validated
        using the Annotation Tool in the Ops Tool.**
        """
)

CONTINENT = st.sidebar.selectbox("Select Continent: ", ['EU', 'US'])
TENANT = st.sidebar.selectbox("Select Tenant: ", sh.tenant_names(CONTINENT))

st.title("Service Efficiency Report")
st.sidebar.markdown(
    """
        **Select Time Period for Service Efficiency Report:**
    """
)
t_start = st.sidebar.date_input("Select Start Date")
t_end = st.sidebar.date_input("Select End Date")

START_SER = dt(t_start.year, t_start.month, t_start.day)
END_SER = dt(t_end.year, t_end.month, t_end.day, 23, 59, 59)

tendency_radio = st.sidebar.radio('Tendency Report:', ('Included', 'None'), index = 1)

if tendency_radio == 'Included':

    st.sidebar.markdown(
        """
            **Select Time Period for Tendency Report:**
        """
    )
    start_month = st.sidebar.select_slider('Start Month:', options=list(sh.month_names_in_a_year))
    start_year = st.sidebar.number_input('Start Year:', min_value=2020, value=dt.now().year, step=1)

    st.sidebar.write('')

    end_month = st.sidebar.select_slider('End Month:',
                                        options=list(sh.month_names_in_a_year), 
                                        value=sh.month_nums_in_a_year[dt.now().month])
    end_year = st.sidebar.number_input('End Year:', min_value=2020, value=dt.now().year, step=1)

    START_tendency = dt(start_year, sh.month_names_in_a_year[start_month], 1)
    END_tendency = dt(end_year, sh.month_names_in_a_year[end_month], 1) + relativedelta(months=1)

    START = min(START_SER, START_tendency)
    END = max(END_SER, END_tendency)
else:
    START = START_SER
    END = END_SER

if st.sidebar.button("Generate Report"):
    generation_start = dt.now()

    location_obtained = False

    #Gather Metadata
    metadata_status = st.empty()
    metadata_status.info('Obtaining Metadata')  
    metadata = sh.metadataQuery(TENANT, CONTINENT)
    metadata_status.success('Metadata Obtained')

    #Gather Service
    services_status = st.empty()
    services_status.info('Obtaining Services')  
    services = sh.servicesQuery(TENANT, CONTINENT, START, END)
    services_status.success('Services Obtained')

    #Gather Locations
    if TENANT in sh.tenants_needing_locations:
        location_obtained = True
        function_name = f'sh.{TENANT.replace("-", "_")}_locations'
        metadata = eval(function_name + "(metadata)")
        
    # Merge Services and Metadata
    services = pd.merge(services, metadata, on='serial_no', how = 'left')

    readings_load_status = st.empty()
    readings_dict = {}
    load_status = 1
    for i in services.serial_no.unique():
        readings_load_status.info(f'Gathering Readings for Device {load_status} of {services.serial_no.nunique()}')
        device_ts_list = list(services[services.serial_no == i].original_service_ts)
        readings_dict[i] = sh.readingsQuery(i, services, metadata)
        load_status += 1
    readings_load_status.success('Readings Gathered')

    # Adjust Services
    services_adjustment_status = st.empty()
    services_adjustment_status.info('Adjusting Services')
    if len(services) > 0:
        # Get pinned fill before, fill after and adjusted service ts
        list_of_adjustments = services.apply(lambda x: sh.service_adjuster(services, 
                                                                           metadata, 
                                                                           x.serial_no, 
                                                                           x.original_service_ts, 
                                                                           (x.name + 1), 
                                                                           readings_dict), axis = 1)
        df_of_adjustments = pd.DataFrame(list_of_adjustments.tolist())

        services['fill_before_serviced'] = df_of_adjustments[0]
        services['fill_after_serviced'] = df_of_adjustments[1]
        services['adjusted_service_ts'] = df_of_adjustments[2]
    else:
        pass
    services_adjustment_status.success('Services Adjusted')

    # Delete Wrong Services
        # Services where (Fill Before - Fill After) <= 5%
    identical_idxs = np.where(services.fill_before_serviced <= (services.fill_after_serviced + 5.0))[0] 
    services.loc[identical_idxs, ('fill_before_serviced', 'fill_after_serviced', 'adjusted_service_ts')] = np.nan

    delete_list = services[pd.isna(services['fill_before_serviced']) | pd.isna(services['fill_after_serviced'])
                                                                     | pd.isna(services['adjusted_service_ts'])].index.values
    services.drop(delete_list, axis = 0, inplace = True)
    services.reset_index(drop = True, inplace = True)  

    # Split into Tendency Services and SER Services
    if tendency_radio == 'Included':
        tendency_services = services[(services.original_service_ts >= START_tendency) & (services.original_service_ts <= END_tendency)]
    else:
        pass
    
    services = services[(services.original_service_ts >= START_SER) & (services.original_service_ts <= END_SER)]

    generate_status = st.empty()
    generate_status.info('Generating Service Efficiency Report')
    # Calculate More Columns
    services['fill_drop'] = services['fill_before_serviced'] - services['fill_after_serviced']
    services['volume_emptied'] = services['capacity_volume'] * services['fill_drop']/100
    services['week_num'] = services.apply(lambda x: x.adjusted_service_ts.week, axis = 1)
    services['day_of_week'] = services['adjusted_service_ts'].map(dt.weekday) + 1
    services['week_start_date'] = services.apply(lambda x: (x['adjusted_service_ts'] 
                                                            - timedelta(days = x['day_of_week']-1)).date(), axis = 1)
    
    services.sort_values(['serial_no', 'adjusted_service_ts'], ascending = [True, True], inplace=True)
    services.reset_index(inplace = True)
    
    if location_obtained: 
        services = services[["serial_no", "container_id", "fill_before_serviced", "fill_after_serviced", 
                            "fill_drop", "volume_emptied", "adjusted_service_ts", "original_service_ts", "detected_by",
                            "container_name", "content_type", "container_type", "location",
                            "device_offset", "total_distance", "capacity_height", "capacity_volume",
                            "width", "latitude", "longitude", "coordinates",
                            "week_num", "day_of_week", "week_start_date"]]
    else:   
        services = services[["serial_no", "container_id", "fill_before_serviced", "fill_after_serviced", 
                            "fill_drop", "volume_emptied", "adjusted_service_ts", "original_service_ts", "detected_by",
                            "container_name", "content_type", "container_type",
                            "device_offset", "total_distance", "capacity_height", "capacity_volume",
                            "width", "latitude", "longitude", "coordinates",
                            "week_num", "day_of_week", "week_start_date"]]
    
    # Timestamp Calculation Columns
        # Previous Timestamp for each service
    prev_ts_series = pd.Series([], dtype = 'datetime64[ns]')
    prev_fill_after_serviced = pd.Series([], dtype = 'float64')
    for x in services.serial_no.unique():
        prev_ts_series = prev_ts_series.append(sh.previous_service_timestamps(x, services))
        prev_fill_after_serviced = prev_fill_after_serviced.append(sh.previous_after_service_values(x, services))

    services['prev_service_ts'] = prev_ts_series
    services.at[np.where(pd.isna(services.prev_service_ts))[0], 'prev_service_ts'] = services.adjusted_service_ts

    services['prev_fill_after_serviced'] = prev_fill_after_serviced

        # Time between services
    services['time_diff_days'] = (services['adjusted_service_ts'] - services['prev_service_ts']) / pd.to_timedelta(1, unit = 'D')
    services.at[np.where(services.adjusted_service_ts == services.prev_service_ts)[0], 'time_diff_days'] = np.nan
    services.at[np.where(services.adjusted_service_ts == services.prev_service_ts)[0], 'prev_service_ts'] = np.nan

        # Fill rate per day
    services['fill_rate_per_day'] = (services['fill_before_serviced'] - services['prev_fill_after_serviced'])/services['time_diff_days']

        # Theoretical days to full
    services['days_to_full'] = 80/services['fill_rate_per_day']

        # Theoretical days overflowing
    services['days_overflowing'] = services['time_diff_days'] - services['days_to_full']
    
    # Delete Rows in services where days to full is negative (Problem in Madrid)
    services = services.drop(np.where(services.days_to_full < 0)[0]).reset_index(drop = True)
    
    # Report Generation
    readme = sh.create_readme_sheet(TENANT, START_SER, END_SER, metadata, services)
    container_level_report = sh.container_level_report_sheet(services)
    services_chart = sh.services_chart_generator(services)
    overall_report = sh.create_overall_report_sheet(metadata, services)
    wt_report = sh.create_key_report_sheet(metadata, services, 'content_type')
    overall_share = sh.share_of_services_sheet(TENANT, services)
    wt_share = sh.key_share_of_services(TENANT, services, 'content_type')
    if location_obtained:
        location_report = sh.create_key_report_sheet(metadata, services, 'location')
        location_share = sh.key_share_of_services(TENANT, services, 'location')
    else:
        pass

    slowest_serviced_df = sh.top_10(container_level_report['Avg Days Between Services'].nlargest(10).index.values, 'Avg Days Between Services', container_level_report)
    fastest_serviced_df = sh.top_10(container_level_report['Avg Days Between Services'].nsmallest(10).index.values, 'Avg Days Between Services', container_level_report)
    
    slowest_filling_df = sh.top_10(container_level_report['Avg Days to Full'].nlargest(10).index.values, 'Avg Days to Full', container_level_report)
    fastest_filling_df = sh.top_10(container_level_report['Avg Days to Full'].nsmallest(10).index.values, 'Avg Days to Full', container_level_report)

    if tendency_radio == 'Included':
        overall_tendency_df = sh.tendency_share_of_services(TENANT, START, END, tendency_services)
    else:
        pass


    # Recommendations
        # Container Level
    recs = container_level_report.apply(lambda x: sh.recommendations(container_level_report, x.name), axis = 1)
    container_level_report['Services Recommendation'] = list(zip(*recs))[0]
    container_level_report['Container Recommendation'] = list(zip(*recs))[1]

        # Location Level
    if location_obtained:
        location_recs = location_report.apply(lambda x: sh.recommendations(location_report, x.name), axis = 1)
        location_report['Services Recommendation'] = list(zip(*location_recs))[0]
        location_report['Container Recommendation'] = list(zip(*location_recs))[1]
    else:
        pass


    # Write to Excel
    with io.BytesIO() as output:
        with pd.ExcelWriter(output) as writer:
            readme.to_excel(writer, sheet_name = "Report Overview", index = False)
            services_chart.to_excel(writer, sheet_name = "Services Chart", index = True)
            container_level_report.to_excel(writer, sheet_name = "Container Level Report", index = False)
            overall_report.to_excel(writer, sheet_name = "Overall Report", index = False)
            if location_obtained:
                location_report.to_excel(writer, sheet_name = "Location Report", index = False)
            else:
                pass
            wt_report.to_excel(writer, sheet_name = "Waste-Type Report", index = False)
            overall_share.to_excel(writer, sheet_name = "Overall Share", index = False)
            if location_obtained:
                location_share.to_excel(writer, sheet_name = "Location Share", index = False)
            else:
                pass
            wt_share.to_excel(writer, sheet_name = "Waste-Type Share", index = False)
            services.to_excel(writer, sheet_name = "SER Source Data", index = False)
            slowest_serviced_df.to_excel(writer, sheet_name = "Slowest Serviced", index = False)
            fastest_serviced_df.to_excel(writer, sheet_name = "Fastest Serviced", index = False)
            slowest_filling_df.to_excel(writer, sheet_name = "Slowest Filling", index = False)
            fastest_filling_df.to_excel(writer, sheet_name = "Fastest Filling", index = False)
            if tendency_radio == 'Included':
                overall_tendency_df.to_excel(writer, sheet_name = "Tendency Report", index = False)
                tendency_services.to_excel(writer, sheet_name = "Tendency Source Data", index = False)
            else:
                pass
            
        data = output.getvalue()
        
    b64 = base64.b64encode(data).decode("utf-8")
    href = f'<a download="{TENANT}-service_report-{START.date()}-to-{(END).date()}.xlsx" href="data:file/excel;base64,{b64}">Download Service Efficiency Report as Excel File</a>'
    st.write(readme)
    st.markdown(href, unsafe_allow_html=True)
    
    # Figure Generator
    with io.BytesIO() as output:
        writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
        workbook = writer.book

        fg.overall_share_fig(overall_share, writer, workbook, TENANT)
        if location_obtained:
            fg.location_share_fig(location_share, writer, workbook, TENANT)
        
        fg.wt_share_fig(wt_share, writer, workbook, TENANT)
        if location_obtained:
            fg.overflowing_by_location_fig(location_report, writer, workbook)
        
        fg.overflowing_by_wt_fig(wt_report, writer, workbook)

        fg.top_10_fig(slowest_serviced_df, writer, workbook, 'slowest_serviced_fig', 'Longest number of days between services')
        fg.top_10_fig(fastest_serviced_df, writer, workbook, 'fastest_serviced_fig', 'Shortest number of days between services')
        fg.top_10_fig(slowest_filling_df, writer, workbook, 'slowest_filling_fig', 'Longest number of days to full')
        fg.top_10_fig(fastest_filling_df, writer, workbook, 'fastest_filling_fig', 'Shortest number of days to full')

        if tendency_radio == 'Included':
            fg.tendency_fig(overall_tendency_df, writer, workbook, TENANT)
        else:
            pass

        writer.save()       
        data = output.getvalue()
            
        b64 = base64.b64encode(data).decode("utf-8")
        href = f'<a download="{TENANT}-figures-{START.date()}-to-{(END).date()}.xlsx" href="data:file/excel;base64,{b64}">Download Figures as Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)


    generate_status.success('Service Efficiency Report Generated')


    generation_end = dt.now()
    time_taken = generation_end - generation_start
    st.sidebar.markdown(f"Time taken: {divmod(time_taken.seconds, 60)[0]}m and {divmod(time_taken.seconds, 60)[1]}s")
else:
    pass

        
        
    




