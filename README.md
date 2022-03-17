# Two Coins Flipping by EM Algorithm
## Description
A project using EM algorithm to estimate parameters of Two Coins Flipping problem.
The description of problem can be found in <a href="https://ra-training.s3-us-west-1.amazonaws.com/DoBatzoglou_2008_EMAlgo.pdf"> this article </a>.

## Project structure:
 - main.py: running the whole pipeline: crawling data, estimating parameters, representing the results.
 - algorithms.py: storing functions of EM algorithms.
 - network_services: providing the connector to data retrieval.
 - tuner.py: tuning the hyper-parameters for EM algorithms.
 - unit_test.py: unit test.

## Run
### Installation
Install python dependencies:
    
    pip install -r requirements.txt

### Tuning hyper-parameter
Run command:

    python tuner.py

### 

### Main pipeline
Apply the best hyper-parameters into main.py. Then run command:

    python main.py