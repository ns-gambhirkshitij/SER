import pandas as pd
import numpy as np
import xlsxwriter
import string

title_font = {'name': 'Rubik', 'bold': False, 'size': 18, 'color': '#537EBD'}
normal_font = {'name': 'Rubik', 'bold': False, 'size': 9, 'color': '#537EBD'}
small_font = {'name': 'Rubik', 'bold': False, 'size': 7, 'color': '#537EBD'}
tiny_font = {'name': 'Rubik', 'bold': False, 'size': 6, 'color': '#537EBD'}

def overall_share_fig(overall_share, writer, workbook, TENANT):
    if TENANT == 'trondheim':
        letter = 'D'
    else:
        letter = 'C'
        
    overall_share.to_excel(writer, sheet_name = 'overall_share_fig', index = None)
    worksheet = writer.sheets['overall_share_fig']
    
    chart = workbook.add_chart({'type': 'column'})
    chart.add_series({'categories': f'=overall_share_fig!$A$1:${letter}$1',
                      'values': f'=overall_share_fig!$A$2:${letter}$2',
                      'data_labels': {'value': True, 'num_format': '0%', 
                                      'font': normal_font},
                      'fill': {'color': '#71D1A7'}})
    
    chart.set_title({'name': 'Share of Services', 
                     'name_font': title_font})
    chart.set_legend({'none': True})
    chart.set_y_axis({'visible': False, 
                      'line': {'none': True},
                      'min': 0, 'max': 1, 
                      'num_font': normal_font, 
                      'major_gridlines': {'visible': False}, 
                      'num_format': '0%'})
    chart.set_x_axis({'line': {'color': '#C9D4D7'}, 
                      'num_font': normal_font, 
                      'major_gridlines': {'visible': False},
                      'major_tick_mark': 'none'})
    
    worksheet.insert_chart('E2', chart)

def location_share_fig(location_share, writer, workbook, TENANT):
    
    if TENANT == 'trondheim':
        rng = 6
    else:
        rng = 5
        
    alphabets = np.char.title(list(string.ascii_lowercase))
    alphabets = dict(zip(range(1,len(alphabets)+1), alphabets))


    location_share.to_excel(writer, sheet_name = 'location_share_fig', index = None)
    worksheet = writer.sheets['location_share_fig']
    
    chart = workbook.add_chart({'type': 'column'})
    
    for i in range(2,rng):
        chart.add_series({'name': f'=location_share_fig!${alphabets[i]}1',
                          'categories': f'=location_share_fig!$A$2:$A${len(location_share) + 1}',
                          'values': f'=location_share_fig!${alphabets[i]}$2:${alphabets[i]}${len(location_share) + 1}'})
    
    chart.set_title({'name': 'Share of Services by Location', 
                     'name_font': title_font})
    chart.set_legend({'position': 'bottom',
                       'font': normal_font})
    chart.set_y_axis({'visible': True, 
                      'line': {'none': False},
                      'min': 0, 'max': 1, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': True, 'line': {'color': '#C9D4D7'}}, 
                      'num_format': '0%'})
    chart.set_x_axis({'line': {'color': '#C9D4D7'}, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': False},
                      'major_tick_mark': 'none'})
    
    worksheet.insert_chart('G2', chart)

def wt_share_fig(wt_share, writer, workbook, TENANT):
    
    if TENANT == 'trondheim':
        rng = 6
    else:
        rng = 5
        
    alphabets = np.char.title(list(string.ascii_lowercase))
    alphabets = dict(zip(range(1,len(alphabets)+1), alphabets))


    wt_share.to_excel(writer, sheet_name = 'wt_share_fig', index = None)
    worksheet = writer.sheets['wt_share_fig']
    
    chart = workbook.add_chart({'type': 'column'})
    
    for i in range(2,rng):
        chart.add_series({'name': f'=wt_share_fig!${alphabets[i]}1',
                          'categories': f'=wt_share_fig!$A$2:$A${len(wt_share) + 1}',
                          'values': f'=wt_share_fig!${alphabets[i]}$2:${alphabets[i]}${len(wt_share) + 1}',
                          'data_labels': {'value': True, 'num_format': '0%', 
                                      'font': small_font}})
    
    chart.set_title({'name': 'Share of Services by Waste-Type', 
                     'name_font': title_font})
    chart.set_legend({'position': 'bottom',
                       'font': normal_font})
    chart.set_y_axis({'visible': False, 
                      'line': {'none': False},
                      'min': 0, 'max': 1, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': False, 'line': {'color': '#C9D4D7'}}, 
                      'num_format': '0%'})
    chart.set_x_axis({'line': {'color': '#C9D4D7'}, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': False},
                      'major_tick_mark': 'none'})
    
    worksheet.insert_chart('G2', chart)

def overflowing_by_location_fig(location_report, writer, workbook):
    location_report.to_excel(writer, sheet_name = 'overflowing_by_location_fig', index = None)
    worksheet = writer.sheets['overflowing_by_location_fig']
    
    chart = workbook.add_chart({'type': 'column'})
    
    chart.add_series({'categories': f'=overflowing_by_location_fig!$A$2:$A${len(location_report) + 1}',
                      'values': f'=overflowing_by_location_fig!$I$2:$I${len(location_report) + 1}',
                      'data_labels': {'value': True, 
                                      'num_format': '0.0', 
                                      'font': tiny_font},
                      'fill': {'color': '#71D1A7'}})
    
    chart.set_title({'name': 'Average Days Overflowing by Location', 
                     'name_font': title_font})
    chart.set_legend({'none': True})
    chart.set_y_axis({'visible': False, 
                      'line': {'none': False}, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': False, 'line': {'color': '#C9D4D7'}}})
    chart.set_x_axis({'line': {'color': '#C9D4D7'}, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': True, 'line': {'color': '#C9D4D7'}},
                      'major_tick_mark': 'cross',
                      'label_position': 'low'})
    
    worksheet.insert_chart('M2', chart)
    
def overflowing_by_wt_fig(wt_report, writer, workbook):
    wt_report.to_excel(writer, sheet_name = 'overflowing_by_wt_fig', index = None)
    worksheet = writer.sheets['overflowing_by_wt_fig']
    
    chart = workbook.add_chart({'type': 'column'})
    
    chart.add_series({'categories': f'=overflowing_by_wt_fig!$A$2:$A${len(wt_report) + 1}',
                      'values': f'=overflowing_by_wt_fig!$I$2:$I${len(wt_report) + 1}',
                      'data_labels': {'value': True, 
                                      'num_format': '0.0', 
                                      'font': tiny_font},
                      'fill': {'color': '#71D1A7'}})
    
    chart.set_title({'name': 'Average Days Overflowing by Waste-Type', 
                     'name_font': title_font})
    chart.set_legend({'none': True})
    chart.set_y_axis({'visible': False, 
                      'line': {'none': False}, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': False, 'line': {'color': '#C9D4D7'}}})
    chart.set_x_axis({'line': {'color': '#C9D4D7'}, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': False, 'line': {'color': '#C9D4D7'}},
                      'major_tick_mark': 'none',
                      'label_position': 'low'})
    
    worksheet.insert_chart('K2', chart)
    
def top_10_fig(df, writer, workbook, sheetname, title):
    df.to_excel(writer, sheet_name = sheetname, index = None)
    worksheet = writer.sheets[sheetname]
    
    chart = workbook.add_chart({'type': 'bar'})
    
    custom_data_labels = []
    for i in range(len(df)):
        custom_data_labels.append({'value': df.at[i, 'Waste Type'], 'font': small_font})
    
    chart.add_series({'categories': f'={sheetname}!$A$2:$A${len(df) + 1}',
                      'values': f'={sheetname}!$B$2:$B${len(df) + 1}',
                      'data_labels': {'custom': custom_data_labels},
                      'fill': {'color': '#71D1A7'}})
    
    chart.set_title({'name': title, 
                     'name_font': title_font})
    chart.set_legend({'none': True})
    chart.set_y_axis({'visible': True, 
                      'line': {'none': True}, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': False, 'line': {'color': '#C9D4D7'}}})
    chart.set_x_axis({'line': {'color': '#C9D4D7'}, 
                      'num_font': small_font, 
                      'major_gridlines': {'visible': True, 'line': {'color': '#C9D4D7'}},
                      'major_tick_mark': 'none'})
    
    worksheet.insert_chart('K2', chart)

def tendency_fig(overall_tendency_df, writer, workbook, TENANT):

    if TENANT == 'trondheim':
        rng = 6
    else:
        rng = 5

    alphabets = np.char.title(list(string.ascii_lowercase))
    alphabets = dict(zip(range(1,len(alphabets)+1), alphabets))

    overall_tendency_df.to_excel(writer, sheet_name = 'tendency_fig', index = None)
    worksheet = writer.sheets['tendency_fig']

    line_chart = workbook.add_chart({'type': 'line'})
    
    column_letter = 3
    for i in list(overall_tendency_df.columns)[2:]:
        line_chart.add_series({'categories': f'tendency_fig!$A$2:$A${len(overall_tendency_df) + 1}',
                          'values': f'=tendency_fig!${alphabets[column_letter]}$2:${alphabets[column_letter]}${len(overall_tendency_df) + 1}',
                               'name': i})
        column_letter += 1
        
    
    line_chart.set_title({'name': 'Share of Services Tendency',
                         'name_font': title_font})
    line_chart.set_legend({'position': 'bottom',
                       'font': normal_font})
    
    line_chart.set_y_axis({'visible': True, 
                      'line': {'none': True}, 
                      'major_gridlines': {'visible': True, 'line': {'color': '#C9D4D7'}},
                      'num_font': normal_font,
                      'num_format': '0%',
                      'min': 0,
                      'max': 1})
    line_chart.set_x_axis({'line': {'color': '#C9D4D7'}, 
                      'num_font': normal_font, 
                      'major_gridlines': {'visible': False, 'line': {'color': '#C9D4D7'}},
                      'major_tick_mark': 'none'})
    
    column_chart = workbook.add_chart({'type': 'column'})
    
    column_chart.add_series({'categories': f'tendency_fig!$A$2:$A${len(overall_tendency_df) + 1}',
                             'values': f'tendency_fig!$B$2:$B${len(overall_tendency_df) + 1}',
                            'name': '=tendency_fig!$B$1'})
    
    line_chart.combine(column_chart)
    
    worksheet.insert_chart(f'{alphabets[rng+1]}2', line_chart)
    