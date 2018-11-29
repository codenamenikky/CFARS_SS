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
please follow the following steps to run the script. 
* clone the git repo to a local directory
* install python if you don't already have python2
* install the required packages in python 
* open cmd
* navigate to the git folder 
* run the following code. 
    * python CFARS_SS_Phase1_Analaysis.py "input_filename" "config_filename" "results_filename"
    * please note that you need to replace "input_filename" with your fully qualified(should include the file extension) filename, please do the same for the config_filename and results_filename, the results filename can only be a xlsx 
    * ![Screen Capture](https://github.com/codenamenikky/CFARS_SS/blob/master/cmdScreenCapture.PNG)


    