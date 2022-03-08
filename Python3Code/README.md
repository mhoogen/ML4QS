1. Installation and Setup

In order to make sure that the provided code runs on all platforms (Windows, Max and Linux). We strongly advise to make use of Anaconda (or Miniconda).
Using Anaconda, you can run the code in your favorite IDE and install the ML4QS requirements in a separate virtual environment. Naturally, if you are familiar with virtual environments and requirements files one can choose the setup of your choice (Docker Start Up Guide Below), but please be aware that this setup is not tested and you have to solve the problems you encounter yourself. 
Download and install Anaconda from: 
https://docs.anaconda.com/anaconda/install/ 
Clone the ML4QS repository by running:

```bash
git clone https://www.github.com/mhoogen/ML4QS.git
```
Windows:

Open the Anaconda Prompt as Adminstrator (when you right click on the Anaconda Prompt, you see the option 'as Adminstrator'). 

Mac/Linux:

Open the terminal


Navigate to the Python3Code folder using cd <path to your ML4QS/Python3Code folder> 
Enter the following commands to create and activate the environment:
```bash
conda create --name myenv python=3.8.8
conda activate myenv
```

Run the following commands to install the required dependencies:

```bash
pip3 install -r requirements.txt 
```

```bash
pip3 install -r requirements_git.txt 
```
If you have any more questions or can't seem to get the code working on your system, post your question on the Tech Support FAQ on the Canvas message board and we will address your issue ASAP if it is not already answered there.




2. Dataset Start-Up

To get started with your coursework:
Download the crowdsignals.io dataset from http://www.cs.vu.nl/~mhoogen/ml4qs/crowdsignals.zip. 
Create a subdirectory in the Python3Code directory called: ‘datasets’
Extract the downloaded dataset in the newly created directory ‘datasets’
Your file structure should now look like: Python3Code/datasets/crowdsignals/csv-participant-one/acc.csv … (multiple csv files)
Redirect to the main directory: Python3Code

Run the following command to run the very first chapter. (this might take a while): 
```bash
python3 crowdsignals_ch2.py
```
Study section ‘3. Code Instructions’ while you are waiting for Chapter2 to finish.

3. Code Instructions

Pre-processing the data (Chapters 3, 4 and 5) can be done in multiple ways. It depends on the characteristics of the dataset which method suits best. 
Therefore at chapters 3, 4 and 5 several data-preprocessing methods are first studied on a single variable. To prevent having to continuously select (and unselect) subsets of code, we strongly advise to use the Argument Parser included in the code for these specific chapters. However, if one prefers to select and unselect specific sections of the code and hard code parameter settings this will also work. 

To study the effect of a specific method on the data, one can add the name of this method to the --mode argument. At the bottom of each script, one can find all possible methods. 

Each chapter also has a --mode argument called ‘final’.  This selection contains the optimal methods selection and is used on all the variables, this results in the dataset needed for the next chapter. 

Also, in some cases one can add parameter settings as well through the argument parser. 

For example to solely run the LOF outlier detection method of Chapter3 with parameter-argument: K=4:
```bash
python3 crowdsignals_ch3_outliers.py --mode='LOF' --K=4
```
And to finish chapter3-outliers, and move to chapter3-rest:
```bash
python3 crowdsignals_ch3_outliers.py --mode='final'
```

!! IMPORTANT !!

Running chapter 4 might give the error: ValueError: numpy.ndarray changed, may indicate binary incompatiblity. Expected 88 from C header, got 80 from PyObject. This is related to an incompatibility of Numpy. A quick fix is to temporarily update NumPy with pip install numpy==1.20.3, execute crowdsignals ch4.py and downgrade with pip install numpy==1.18.2 afterwards. Make sure to use version 1.18.2 for all chapters except chapter 4.


4. OPTIONAL (Docker fanatics)

Installing with Docker allows you to set up the course materials in a separate virtual container running Ubuntu. This guarantees compatibility and prevents system-specific issues. Docker may require some configuration to efficiently use your system resources in the Docker Desktop app.
Download and install Docker (Docker Desktop is the easiest to work with).


clone/download the ML4QS data from the ML4QS repository by running git clone https://www.github.com/mhoogen/ML4QS.git.


Open a command prompt or terminal window and navigate to the ML4QS folder by entering cd <path to your ML4QS/Python3Code folder>.


On Windows: Enter the following command prompt to build and run the Docker image.:
```bash
 start start_docker.bat
 ```


On MacOS/Linux: Enter the following command to build and run the Docker image.  
```bash
chmod +x start_docker.sh
./start_docker.sh 
```


Once the docker image is finished building, you should be able to launch your Docker container by running the batch/shell script. The Python3Code directory has been attached to the container as a volume, so you can access or write to any file or directory in this volume. Run ls in the Docker terminal to list the contents of the folder. Once you've downloaded the crowdsignals.io data and placed it in the appropriate directory, you should be able to run python3 crowdsignals_ch2.py to execute the first script in Docker.
Note: Since Docker runs Ubuntu in headless mode, commands such as plt.show() will not display any figures. However, the figures are numbered and saved automatically in the Python3Code/figures directory, where you can view them as the script runs. See the FAQ document on the course discussion page for more information on running Docker.

