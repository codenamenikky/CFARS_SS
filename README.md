# CFARS_SS

This script is created to run the CFARS_SS Phase 1 data analysis on your internal data. Given every company has a different input file a config file is created to make the script understand the input file. The data analysis is done based on what was requested in the excel template for results. please clone this repo to run it internally. If you have issues please create a git hub issue. The main script is CFARS_SS_Phase1_Analysis.py, the configuration file template is also provided in the repo. 

#### Requirements

This is a python script, so you need Python2 to run this script. If you are new to python Anaconda is a good place to start, just make sure you choose Python 2.
The following packages are required to run this script. 

1.pandas 
2.sklearn
3.openpyxl
 
you can run 
'''python
pip install -r requirements.txt 
'''
with the requriements file in the

#### How to run the script. 
please follow the following steps to run the script. YouTube Video --> https://youtu.be/wtyHk7J1NV4 

	1. Get the code. (2 unique ways to get the code) https://github.com/codenamenikky/CFARS_SS
		1. Download directly from github
		2. Clone the repo to your github desktop application
	2. Get Python
		1. Get Anaconda (preferred) https://www.anaconda.com/
		2. get modules [pandas, sklearn, openpyxl] (they come installed with anaconda)
	3.  Configuration file for your data. 
		1.get the headers from your internal data file 
		2.paste the header into the configuration template 
		3.change the header information with the appropriate names 
	4. Run code on your data and configuration file
		1. Open cmd and change the directory to your CFARS_SS code folder
		2. python CFARS_SS_Phase1_Analysis.py "inputdata.xlsx" "configuration_template.xlsx" "results.xlsx"

    * ![Screen Capture](https://github.com/codenamenikky/CFARS_SS/blob/master/cmdScreenCapture.PNG)


    
