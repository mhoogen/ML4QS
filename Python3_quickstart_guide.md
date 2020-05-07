You have two main options to install the source code and packages:

Option 1. Install with Docker

Installing with Docker allows you to set up the course materials in a separate virtual container running Ubuntu. This guarantees compatibility and prevents system-specific issues. Docker may require some configuration to efficiently use your system resources in the Docker Desktop app.

1. Download and install Docker (Docker Desktop is the easiest to work with).
2. clone/download the ML4QS data from the ML4QS repository by running `git clone https://www.github.com/mhoogen/ML4QS.git`.
3. Open a command prompt or terminal window and navigate to the ML4QS folder by entering `cd path/to/ML4QS`.
3. <i>On Windows</i>: Enter `start Python3_start_docker.bat` into command prompt to build and run the Docker image.
   
   <i>On MacOS/Linux</i>: Enter `./Python3_start_docker.sh` into terminal to build and run the Docker image. You may have to set permissions for the file first using `chmod +x Python3_start_docker.sh`.
   
Once the docker image is finished building, you should be able to launch your Docker container by running the batch/shell script. The Python3Code directory has been attached to the container as a volume, so you can access or write to any file or directory in this volume. Run `ls` in the Docker terminal to list the contents of the folder. Once you've downloaded the crowdsignals.io data and placed it in the appropriate directory, you should be able to run `python3 crowdsignals_ch2.py` to execute the first script in Docker.

Option 2. Install with Anaconda 

Using Anaconda, you can run the code in your favorite IDE and install the ML4QS requirements in a separate virtual environment. 

1. Download and install the latest version of Anaconda from https://www.anaconda.com/distribution/.
2. Create a new virtual environment using the Anaconda Navigator or the Anaconda Prompt (Windows) or terminal (MacOS/Linux). Enter `conda create --name <my_env_name>` to create a new environment, then activate it by entering `conda activate <my_env_name>`. 
3. Navigate to your ML4QS folder using `cd path/to/ML4QS`. 
4. Run `pip install -r Python3_requirements.txt` to install the required dependencies. 

If you have any more questions or can't seem to get the code working on your system, post your question on the Tech Support FAQ on the Canvas message board, or send me an e-mail at t.m.maaiveld@student.vu.nl.
