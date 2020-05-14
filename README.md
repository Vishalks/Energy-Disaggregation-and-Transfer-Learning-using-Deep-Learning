# Energy-Disaggregation-and-Transfer-Learning-using-Deep-Learning
This repository contains my implementation for Energy Disaggregation of appliances from mains consumption using stacked ensemble deep learning

#Installation
This project is implemented in python 3.7 and tensorflow 2.0. Follow these steps to setup your environment:
1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")

2. After Anaconda has been installed, open up the terminal (Unix) or Anaconda prompt (Windows).
Create a new environment where NILMTK and related dependencies are installed.
	```bash
	conda create --name nilmtk-env 
	```
  
3. Add conda-forge to list of channels to be searched for packages.
	```bash
	conda config --add channels conda-forge
	```

4. Activate the new *nilmtk-env* environment.

	```bash
	conda activate nilmtk-env
	```

5. Install the NILMTK package

	```bash
	conda install -c nilmtk nilmtk
	```
  
6. Install the NILMTK package

	```bash
	conda install -c nilmtk nilmtk
	```
  
7. Run your Python IDE from this environment, for example:

	```bash
	jupyter notebook
	```
	or

	```bash
	spyder
	```
  
8. Download the REDD dataset from [here](http://redd.csail.mit.edu/data/low_freq.tar.bz2) using redd/disaggregatetheenergy as username/password. Unzip it in the same directory as RNN-example.ipynb.
9. Give path of the low_freq directory that you will get after the unzipping process above to the convert_redd function in the very first cell of notebook. Give appliance name that you want to run the disaggregation on and run the notebook. It might take some time to load data and convert into h5 format. After the notebook is run, you can see the actual vs predicted appliance consumption along with prediction metrics.
