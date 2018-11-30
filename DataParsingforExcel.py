import CFARS_SS_Phase1_Analysis as m
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook


def writedatainbins(inputdata, results_filename):
    wb = Workbook()
    
    for i in range(2,21):
        sheetname = '{}mps_bin'.format(i)
        wb.create_sheet(sheetname)
        ws = wb[sheetname]
        df = inputdata[inputdata['bins']==i]
        m.write_resultstofile(df,ws, 1,1)
    wb.save(results_filename)

if __name__ == '__main__':
    input_filename, config_file, results_filename = m.get_inputfiles()
    inputdata = m.get_inputdata(input_filename, config_file)
    CFARScolumns = m.get_CFARScolumns(config_file)
    CFARScolumns.append('bins')
    inputdata = inputdata[CFARScolumns]
    writedatainbins(inputdata, results_filename)