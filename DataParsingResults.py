"""
This is the script which is trying to gather all the results in one location from all the 
results file. 

Author : Nikhil Kondabala
Date: 01/16/2018 14:51
Version: 0.0.1 

the general process that we are trying to get to is get the filename of the resutls , gather company 
project details. then gather all the regression resutls in one locaiton 
get all the other attributes in one location and make a dataframe out of it. 
then read the excel file, get the last row of the excel sheet and then enter the data into the excel sheet. 
"""

#TODO: write the data to the excel sheet 
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook

def write_resultstofile(df,ws, r_start,c_start):
    # write the regression results to file.
    rows = dataframe_to_rows(df)
    for r_idx, row in enumerate(rows, r_start):
        for c_idx, value in enumerate(row, c_start):
            ws.cell(row=r_idx, column=c_idx, value=value)

def get_datasplitout(filename):
    df = pd.read_excel(filename)
    data=df[~df.iloc[:,0].isnull() | ~df.index.isnull()]
    f = filename.split('/')[-1]
    company= f.split('_')[2]
    project= f.split('_')[3]
    attribute = data.index.unique()
    attribute = attribute[~attribute.isnull()]
    
    # TRY TO GET ALL THE RESULTS WE NEED IN A DATAFRAME 
    def get_aboveandbelow_nominal_stats():
        ab_row = data.index.get_loc('Above nominal Status')
        b_row = data.index.get_loc('Below nominal Stats')
        a_row = data.index.get_loc('Total Bin Stats')
        cols = ['TI_error_mean',	'TI_error_std',	'TI_diff_mean',	'TI_diff_std']

        ab_stats = data.iloc[ab_row+2:ab_row+6,0:4]
        b_stats = data.iloc[b_row+2:b_row+6,0:4]
        a_stats = data.iloc[a_row+2:a_row+6,0:4]
        
        ab_stats.columns = [i+'_AboveNominal' for i in cols]
        b_stats.columns = [i+'_BelowNominal' for i in cols]
        a_stats.columns = [i+'_All' for i in cols]

        stats = ab_stats.join(b_stats)
        stats = stats.join(a_stats)
        TI_diff_cols = stats.columns[stats.columns.str.contains('TI_diff')].tolist()
        TI_error_cols = stats.columns[stats.columns.str.contains('TI_error')].tolist()

        return stats.loc[:,TI_diff_cols] , stats.loc[:,TI_error_cols]

    def get_regressionresults(): #REGRESSION RESULTS IN A DATAFRAME 
        reggression = attribute[attribute.str.contains('WS_regression')]
        reggression_df = data.loc[reggression].copy()
        reggression_df = reggression_df.iloc[:,0:4]

        reggression_df = reggression_df.assign(Project=project)
        reggression_df= reggression_df.assign(Company=company)
        cols = list(reggression_df.columns)
        i = [len(cols)-1, len(cols)-2]
        i+=range(0,len(cols)-2)
        orderedcols = [cols[k] for k in i]

        return reggression_df[orderedcols]

    def get_TI_error_results(): # get the TI mean bias error results by bin    
        TI_error =attribute[attribute.str.contains('TI_error')]
        TI_error_df = data.loc[TI_error].copy()
        TI_error_df.loc['Project'] = project
        TI_error_df.loc['Company'] = company
        rows = TI_error_df.shape[0]
        TI_error_df_1mps = TI_error_df.iloc[range(0,rows,2),:]
        TI_error_df_1mps = TI_error_df_1mps.join(statistics_error)
        TI_error_df_05mps = TI_error_df.iloc[range(1,rows,2),:]
        TI_error_df_05mps = TI_error_df_05mps.join(statistics_error)

        return TI_error_df_1mps , TI_error_df_05mps
    
    def get_TI_diff_results(): # get the TI_difference results based on bin 
        TI_diff =attribute[attribute.str.contains('TI_diff')]
        TI_diff_df = data.loc[TI_diff].copy()
        TI_diff_df.loc['Project'] = project
        TI_diff_df.loc['Company'] = company
        rows = TI_diff_df.shape[0]
        TI_diff_df_1mps = TI_diff_df.iloc[range(0,rows,2),:]
        TI_diff_df_1mps = TI_diff_df_1mps.join(statistics_diff)
        TI_diff_df_05mps = TI_diff_df.iloc[range(1,rows,2),:]
        TI_diff_df_05mps = TI_diff_df_05mps.join(statistics_diff)
        return TI_diff_df_1mps , TI_diff_df_05mps 

    def get_TI_values():# get the TI values by bin, get the representative TI values based on mean TI and Std
        TI_values = attribute[attribute.str.contains('_TI')]
        TI_values_df=data.loc[TI_values].copy()
        TI_values_df.loc['Project'] = project
        TI_values_df.loc['Company'] = company
        totallen = TI_values_df.shape[0]
        numofvar = TI_values.shape[0]

        # make a dataframe with aggregate values for the TI values for representative TI
        TI_values_agg_df = TI_values_df.iloc[range(2,totallen,numofvar),0:3]
        TI_values_agg_df.columns = ['mean_15mps','std_15mps','Rep_TI']
        # get the 1 mps bin TI values and add the aggregate representative ti values to the end of df
        TI_values_1mpsbin_df = TI_values_df.iloc[range(0,totallen,numofvar),:]
        # get the .05 mps bin TI values and add the aggregate representative ti values to the end of df
        TI_values_05mpsbin_df = TI_values_df.iloc[range(1,totallen,numofvar),:]
        
        return TI_values_agg_df, TI_values_1mpsbin_df, TI_values_05mpsbin_df
    statistics_diff, statistics_error = get_aboveandbelow_nominal_stats()
    regression_stats = get_regressionresults()
    TI_values_agg_df, TI_values_1mpsbin_df, TI_values_05mpsbin_df = get_TI_values()
    TI_diff_df_1mps , TI_diff_df_05mps = get_TI_diff_results()
    TI_error_df_1mps , TI_error_df_05mps = get_TI_error_results()
    
    # write the results to the file 
    wb = Workbook()
    ws = wb.active
    # regression results
    write_resultstofile(regression_stats,ws,1,1)
    wb.save('./deleteme.xlsx')

if __name__=="__main__":
    filename = r"C:/Users/nikhil.kondabala/Documents/GitHub/CFARS_SS/InternalData/filtereddata/Phase1Tests_ResultsMatrix_Apex_10317_v01_20181130.xlsx"
    get_datasplitout(filename)


