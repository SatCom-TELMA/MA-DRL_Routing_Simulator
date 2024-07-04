# Multi-Agent Deep Reinforcement Learning (MA-DRL) Routing Simulator for satellite networks

Contained in this repository is the code used for simulating data transmissions through satellite constellations and evaluating the latency results through post-processing of the data generated in the simulations.

## MA-DRL routing demonstration in a moving Kepler constellation from Malaga, Spain to Los Angeles, USA

<!-- [![Demo Video](Video/MA-DRL_Movement_screenshot.png)](https://raw.githubusercontent.com/SatNEx-Malaga/MA-DRL_Routing_Simulator/main/Video/MA-DRL_Movement.mp4) -->
[![Demo Video](Video/MA-DRL_Movement_screenshot.png)](https://drive.google.com/file/d/1So7jtUwEdobJLzztXv6JmxP2B79PQDU0/preview)

> Click the image above to download the demo video.

The simulations are built using the event based discrete time simulation framework Simpy.


## Requirements
In order to run the simulators and the post-processing script, certain non-standard python libraries are required. For a full list of the necessary libraries, see "requirements.txt".

## Description
The simulator simulates individual data blocks propagating through a satellite constellation from a source gateway to destination gateway. 
### Data generation
The data blocks are generated at the source gateways independently for each destination gateway. Based on the maximal generation rate of a gateway, each destination receives an equal fraction of the data generation. The fraction to each gateway is determined by the maximum amount of gateways there can be active (this is defined in the inputRL.csv file). If the maximum amount of gateways is set to 18 but only 9 is active, then each gateway will receive 8/17 ((numberOfActive -1) / (totalNumber - 1)) of the maximal generation rate. 
### Pathing
For the non-RL simulations, the gateway adds the path of the data block when it is created. The path is found using a Dijkstra shortest path algorithm at the start of the simulation and everytime the constellation moves. For the RL versions, a path is not created with the data block. Instead, it is built as the block propagates through the network.
### Data propagation
The transmission of data blocks is handled through simpy process functions which monitor transmission queues. Each gateway (which are connected to a satellite) has one FIFO transmit queue where generated data blocks are placed in. Each satellite has one FIFO transmission queue for each satellite link (usually 4, 2 inter plane satellite links and 2 intra plane links). A data block is transmitted by starting a reception process on the receiver. This process waits out the propagation time and determines which transmission queue the block should be in based on the next step in the path of the data block. For the RL version of the simulator, the next step of the path is determined in this process before the block is placed in a queue.
### Constellation movement
The constellation movement is handled in discrete time steps. The constellation is assumed stationary for some amount of time (this time delta can be set in the simulator through the "movementTime" variable in the "main()" function) after which the constellation is moved according to the time delta. In the current setup of the simulation, neither the RL and non-RL versions move the constellation. The current simulation time of 1 second is not considered long enough to warrant movement of the constellation. Furthermore, in the RL-version, one of the required methods in the constellation movement process ("updateSatelliteProcessesRL()") is not working correctly and should be looked at before considering constellation movement with either Q-Learning or Deep Q-Learning.

## Installation

To set up the environment and install all required packages, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/YourUsername/YourRepositoryName.git
    cd YourRepositoryName
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

Make sure you have `Python 3.9` installed. It is recommended to use version 3.9.12.


## Usage


### Simulator
The simulator is split into two separate files:
* The "SimulationRL.py" file is for simulations where Q-Learning or Deep Q-Learning is used as the path metrics.
* The "Simulation.py" file is for the remaining path metrics as well as for gathering the data regarding the link rates.

The parameters for the simulations are controlled using the "input.csv" and "inputRL.csv" files. The parameters consist of:
* Locations: The locations specified for this parameter controls which gateways are used from the "Gateways.csv" file. The accepted inputs for this parameter is:
    * "All": All gateways are used. The order in which they are activated follows the "gateways.csv" file exactly.
    * The names of specific locations: The names must match those found the "Gateways.csv" files up until the first comma. For example, an accepted name is "Aalborg", but "Aalborg Denmark" is not. The order in which they are activated follows the order in the "inputs(RL).csv" file.
* Pathing: The path metric which is used in the shortest path algorithm. The accepted inputs are for inputs.csv: ("dataRate", "latency", "slant_range", and "hop") and for inputsRL.csv: ("Q-Learning", and "Deep Q-Learning").
* Constellation: The satellite constellation which is used in the simulations. All results are gathered using the Kepler constellation, however, Iridium_NEXT, OneWeb, and Starlink are also implemented.
* Fraction: Limits the amount of data that can be generated at each gateway to a fraction of the calculated capacity.
* Test type: There are two test types which can be run for the "Simulation.py" (this parameter is not used in "SimulationRL.py):
    * Latency: The main simulation mode where data is being generated at the active gateways
    * Rates: A secondary simulation mode where no data is generated. Instead, the link data rates for all uplinks, downlink, and satellite links are recorded. In this mode, the constellation moves large distances to cover most of the possible link data rates.
* Test Length: The simulation time in seconds. When the Test type is set to "Rates" this parameter instead defines the amount of constellation movements.

Furthermore, the "inputsRL.py" has a number of additional parameters which among other things cover the hyperparameters of the Q-learning and Deep Q-Learning agents.

Both versions of the simulator is set up to produce and save a wide range of telemetry data for the simulation runs. These will be saved in a Results/ folder which is created in the parent folder of the "Code/" folder. If alternative telemetry data is desired, after a simulation run, then look for the "blocks.npy" files which are saved in the "Results/Congestion_Test/(path metric) (test length)s/" folder. These are numpy arrays of all the data blocks which were successfully received at their destination during the simulation. The blocks are saved as instances of the class "BlocksForPickle" which can be found in simulation(RL).py.

The population maps used in the simulators are found at https://sedac.ciesin.columbia.edu/data/collection/gpw-v4/sets/browse
There is a guide to get the data from the population maps at https://towardsdatascience.com/visualising-global-population-datasets-with-python-c87bcfc8c6a6


### Post-Processing

Some post processing results can be found in `./Post-Processing/Post-Results.ipynb` notebook.


## Known Issues
The updateSatelliteProcessesRL() method in the SimulationRL.py file does not work correctly. It is supposed to remake the transmit processes on the satellite when the constellation moves, however, when the constellation movement necessitates that the satellites form new links, the processes are not remade correctly, and the code encounters an error.

The data generation at the gateways was not handled correctly causing too few data blocks to be generated. Instead of sending ((numberOfActive - 1) / (totalNumber - 1)) to each destination, ((numberOfActive - 1) / (totalNumber)) was sent. The code has been changed to generate the correct amount of data, so to reproduce results from the report, this must be changed back. The specific code is found in the "timeToFullBlock()" method in the Gateway class. This line currently reads: flow = self.totalAvgFlow / (len(self.totalLocations) - 1) (which is the correct behaviour) and should be changed to: flow = self.totalAvgFlow / (len(self.totalLocations)).
