# info254_sfo_police

## Introduction

This github repository is for the final project for UC Berkeley's INFO 254 Course.

Our goal is to predict the prop. of reported crime in San Fransisco, and to analyse the results to determine potential sources and effects of bias in the results. 

## Data

Our data are from https://datasf.org/opendata/

We combine data from the following datasets:

* Police Department Incident Reports: Historical 2003 to May 2018 -
    * https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry
* Police Department Incident Reports: 2018 to Present
    * https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783
* Police Department Calls for Service (FUTURE RESEARCH)
    * https://data.sfgov.org/Public-Safety/Police-Department-Calls-for-Service/hz9m-tj6z
* Census 2010: Blocks for San Francisco
   * https://data.sfgov.org/Geographic-Locations-and-Boundaries/Census-2010-Blocks-for-San-Francisco/2uzy-uv2r
* Census 2010: Tracts for San Francisco
   * https://data.sfgov.org/Geographic-Locations-and-Boundaries/Census-2010-Tracts-for-San-Francisco/rarb-5ahf
* NOAA National Center for Environmental Information Data Request - Weather Data
   * https://www.ncdc.noaa.gov/cdo-web/
* US Census ACS 5-Year estimates (2013-2017) 
   * https://nhgis.org/ 


## Process

Please run the notebooks in the following order:

* dataset_creation.ipynb
* dataset_modification.ipynb
* dataset_feature_engineering.ipynb
* dataset_visulization.ipynb
* model_tuning.ipynb
* RNN_with_cv.ipynb
* model_result_visualization.ipynb
* model_results_census_vis.ipynb


## Authors 
John Iselin - Goldman School of Public Policy 
Takuma Kinoshita - Goldman School of Public Policy 
Naoyuki Komada - Goldman School of Public Policy 
Ted Kumagai - University of California, Berkeley
