1. Installation and Setup

In order to make sure that the provided code runs on all platforms (Windows, Max and Linux). We strongly advise to make use of Anaconda (or Miniconda).
Using Anaconda, you can run the code in your favorite IDE and install the ML4QS requirements in a separate virtual environment. Naturally, if you are familiar with virtual environments and requirements files one can choose the setup of your choice (Docker Start Up Guide Below), but please be aware that this setup is not tested and you have to solve the problems you encounter yourself. 
Download and install Anaconda from: 
https://docs.anaconda.com/anaconda/install/ 
Clone the ML4QS repository by running:

```bash
git clone https://www.github.com/mhoogen/ML4QS.git
```
## Windows:

Open the Anaconda Prompt as Adminstrator (when you right click on the Anaconda Prompt, you see the option 'as Adminstrator'). 
Run the following command:

```bash
conda create --name myenv python=3.8.8
conda activate myenv
```

Then, navigate back to the Python3Code folder using cd <path to your ML4QS/Python3Code folder>.

Run the following commands to install the required dependencies:

```bash
pip3 install -r requirements.txt 
```

```bash
pip3 install -r requirements_git.txt 
```
It could be the case that you run into an error when installing pybrain/pyflux. Two possible solutions are given here:
1. An error stating: '... error microsoft visual c++ 14.0 or greater is required'. In this case, you need to install Visual Studio Build Tools via the following link: https://visualstudio.microsoft.com/visual-cpp-build-tools/ . Once installed, you need to open it, click on modify and mark 'Desktop development with C++'. Afterwards, you might need to reboot. More information can be found via: https://docs.microsoft.com/en-us/answers/questions/136595/error-microsoft-visual-c-140-or-greater-is-require.html.

2. If there is a different pyflux error, installing pyflux via a wheel might help. 
Download a pyflux wheel (based on your python version and desktop) and pip install it 
in the current working directory. Follow the steps to perform.\
Step 1:
Download the pyflux wheel file from this Github repository via the folder pyflux_wheel. There are two files in this folder.
We work with python 3.8.8 so therefor the number 38 is in the file. You either pick the 32 or 64 file.
Check your desktop settings (64 or 32). You can check that [here](https://support.microsoft.com/en-us/windows/32-bit-and-64-bit-windows-frequently-asked-questions-c6ca9541-8dce-4d48-0415-94a3faa2e13d)  \
Step 2:
Put the wheel file in the current working directory \
Step 3:
Install the wheel with the following command:
```bash
pip install pyflux‑0.4.17‑cp38‑cp38‑win_amd64.whl
```
Or if you have a 32 desktop: 
```bash
pip install pyflux‑0.4.17‑cp38‑cp38‑win32.whl
```
## Mac/Linux:

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
In case you run into an error installing PyFlux, please run ```xcode-select --install``` and then re-run ```pip3 install -r requirements_git.txt```.

If you have any more questions or can't seem to get the code working on your system, post your question on the Tech Support FAQ on the Canvas message board and we will address your issue ASAP if it is not already answered there.




2. Dataset Start-Up

To get started with your coursework:
Download the crowdsignals.io dataset from https://www.cs.vu.nl/~mhoogen/ml4qs/crowdsignals.zip. 
Create a subdirectory in the Python3Code directory called: ‘datasets’
Extract the downloaded dataset in the newly created directory ‘datasets’
Your file structure should now look like: Python3Code/datasets/crowdsignals/csv-participant-one/acc.csv … (multiple csv files)

Important: It might be the case that unzipping the 'crowdsignals.zip' will not automatically create a crowdsignals folder with a subfolder 'csv-participant-one' folder. In this case, you need to manually create a folder crowdsignals and copy the 'csv-participant-one' folder in it.

Redirect to the main directory: Python3Code

Run the following command to run the very first chapter. (this might take a while. You need to close the figures that pop-up so the code can run through!): 
```bash
python3 crowdsignals_ch2.py
```
For Windows it might be the case that you need to run the script via:
```bash
python crowdsignals_ch2.py
```

Study section ‘3. Code Instructions’ while you are waiting for Chapter2 to finish.

3. Code Instructions

Pre-processing the data (Chapters 3, 4 and 5) can be done in multiple ways. It depends on the characteristics of the dataset which method suits best. 
Therefore at chapters 3, 4 and 5 several data-preprocessing methods are first studied on a single variable. To prevent having to continuously select (and unselect) subsets of code, we strongly advise to use the Argument Parser included in the code for these specific chapters. However, if one prefers to select and unselect specific sections of the code and hard code parameter settings this will also work. 

To study the effect of a specific method on the data, one can add the name of this method to the --mode argument. At the bottom of each script, one can find all possible methods. 

Each chapter also has a --mode argument called ‘final’.  This selection contains the optimal methods selection and is used on all the variables, this results in the dataset needed for the next chapter. Hence, you need to run each crowdsignals_ch(<chapter number>).py file with the mode='final' before you can move on to the next python script!

Also, in some cases one can add parameter settings as well through the argument parser. 

For example to solely run the LOF outlier detection method of Chapter3 with parameter-argument: K=4:
```bash
python3 crowdsignals_ch3_outliers.py --mode='LOF' --K=4
```
And to finish chapter3-outliers, and move to chapter3_rest:
```bash
python3 crowdsignals_ch3_outliers.py --mode='final'
```
Finally, you can run
```bash
python3 crowdsignals_ch3_rest.py --mode='final'
```
to finish Chapter 3.

4. OPTIONAL (Docker fanatics)

Installing with Docker allows you to set up the course materials in a separate virtual container running Ubuntu. This guarantees compatibility and prevents system-specific issues. Docker may require some configuration to efficiently use your system resources in the Docker Desktop app.
Download and install Docker (Docker Desktop is the easiest to work with). You can download 'Docker Desktop' via this link: https://www.docker.com/products/docker-desktop/.


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

