import time
import pandas as pd
import math
import numpy as np
import geopy.distance
import simpy
import numba
import networkx as nx
from PIL import Image
from scipy.optimize import linear_sum_assignment
import pickle
import random
import os
import folium
from IPython.display import display
from typing import List, Tuple
from datetime import datetime
import seaborn as sns
import gc
import cProfile
from collections import defaultdict
import glob
import builtins

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
import matplotlib.cm as cm


###############################################################################
################################    Log file    ###############################
###############################################################################

import sys
import atexit

class Logger(object):
    def __init__(self, filename='logfile.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        atexit.register(self.close)  # Register the close method to be called when the program exits

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()


###############################################################################
########################    Deep Learning Framework    ########################
###############################################################################

import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential, losses
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, Reshape, Input, Conv2D, Flatten
from collections import deque

# Forcing TensorFlow to use GPU - No worth using GPU for reinforcement learning in this case 
#                                 since the training is done every step with small buffers
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     print('GPU(s) available:')
#     print(physical_devices)
# else:
#     print('No GPU available')

###############################################################################
###############################    Constants    ###############################
###############################################################################

# HOT PARAMS - This parameters should be revised before every simulation
pathings    = ['hop', 'dataRate', 'dataRateOG', 'slant_range', 'Q-Learning', 'Deep Q-Learning']
pathing     = pathings[5]# dataRateOG is the original datarate. If we want to maximize the datarate we have to use dataRate, which is the inverse of the datarate

FL_Test     = False     # If True, it plots the model divergence the model divergence between agents
plotSatID   = True      # If True, plots the ID of each satellite
plotAllThro = True      # If True, it plots throughput plots for each single path between gateways. If False, it plots a single figure for overall Throughput
plotAllCon  = True      # If True, it plots congestion maps for each single path between gateways. If False, it plots a single figure for overall congestion

movementTime= 10        # Every movementTime seconds, the satellites positions are updated and the graph is built again
                        # If do not want the constellation to move, set this parameter to a bigger number than the simulation time
ndeltas     = 5805.44/20#1 Movement speedup factor. Every movementTime sats will move movementTime*ndeltas space. If bigger, will make the rotation distance bigger

Train       = True      # Global for all scenarios with different number of GTs. if set to false, the model will not train any of them
explore     = True      # If True, makes random actions eventually, if false only exploitation
importQVals = False     # imports either QTables or NN from a certain path
onlinePhase = False     # when set to true, each satellite becomes a different agent. Recommended using this with importQVals=True and explore=False
if onlinePhase:         # Just in case
    explore     = False
    importQVals = True
else:
    FL_Test = False

w1          = 20        # rewards the getting to empty queues
w2          = 20        # rewards getting closes phisycally   
w4          = 5         # Normalization for the distance reward, for the traveled distance factor 

gamma       = 0.99       # greedy factor. Smaller -> Greedy. Optimized params: 0.6 for Q-Learning, 0.99 for Deep Q-Learning

GTs = [8]               # number of gateways to be tested
# Gateways are taken from https://www.ksat.no/ground-network-services/the-ksat-global-ground-station-network/ (Except for Malaga and Aalborg)
# GTs = [i for i in range(2,9)] # This is to make a sweep where scenarios with all the gateways in the range are considered

# Physical constants
rKM = 500               # radio in km of the coverage of each gateway
Re  = 6378e3            # Radius of the earth [m]
G   = 6.67259e-11       # Universal gravitational constant [m^3/kg s^2]
Me  = 5.9736e24         # Mass of the earth
Te  = 86164.28450576939 # Time required by Earth for 1 rotation
Vc  = 299792458         # Speed of light [m/s]
k   = 1.38e-23          # Boltzmann's constant
eff = 0.55              # Efficiency of the parabolic antenna

# Downlink parameters
f       = 20e9  # Carrier frequency GEO to ground (Hz)
B       = 500e6 # Maximum bandwidth
maxPtx  = 10    # Maximum transmission power in W
Adtx    = 0.26  # Transmitter antenna diameter in m
Adrx    = 0.26  #0.33 Receiver antenna diameter in m
pL      = 0.3   # Pointing loss in dB
Nf      = 2     #1.5 Noise figure in dB
Tn      = 290   #50 Noise temperature in K
min_rate= 10e3  # Minimum rate in kbps

# Uplink Parameters
balancedFlow= False         # if set to true all the generated traffic at each GT is equal
totalFlow   = 2*1000000000  # Total average flow per GT when the balanced traffc option is enabled. Malaga has 3*, LA has 3*, Nuuk/500
avUserLoad  = 8593 * 8      # average traffic usage per second in bits

# Block
BLOCK_SIZE   = 64800

# Movement and structure
# movementTime= 0.05      # Every movementTime seconds, the satellites positions are updated and the graph is built again
#                         # If do not want the constellation to move, set this parameter to a bigger number than the simulation time
# ndeltas     = 5805.44/20#1 Movement speedup factor. This number will multiply deltaT. If bigger, will make the rotation distance bigger
saveISLs    = True     # save ISLs map
const_moved = False     # Movement flag. If up, it means it has moved
matching    = 'Greedy'  # ['Markovian', 'Greedy']
minElAngle  = 30        # For satellites. Value is taken from NGSO constellation design chapter.
mixLocs     = False     # If true, every time we make a new simulation the locations are going to change their order of selection
rotateFirst = False     # If True, the constellation starts rotated by 1 movement defined by ndeltas

# State pre-processing
coordGran   = 20            # Granularity of the coordinates that will be the input of the DNN: (Lat/coordGran, Lon/coordGran)
diff        = True          # If up, the state space gives no coordinates about the neighbor and destination positions but the difference with respect to the current positions
diff_lastHop= True          # If up, this state is the same as diff, but it includes the last hop where the block was in order to avoid loops
reducedState= False         # if set to true the DNN will receive as input only the positional information, but not the queueing information
notAvail    = 0             # this value is set in the state space when the satellite neighbour is not available

# Learning Hyperparameters
ddqn        = True      # Activates DDQN, where now there are two DNNs, a target-network and a q-network
# importQVals = False     # imports either QTables or NN from a certain path
plotPath    = False     # plots the map with the path after every decision
alpha       = 0.25      # learning rate for Q-Tables
alpha_dnn   = 0.01      # learning rate for the deep neural networks
# gamma       = 0.99       # greedy factor. Smaller -> Greedy. Optimized params: 0.6 for Q-Learning, 0.99 for Deep Q-Learning
epsilon     = 0.1       # exploration factor for Q-Learning ONLY
tau         = 0.1       # rate of copying the weights from the Q-Network to the target network
learningRate= 0.001     # Default learning rate for Adam optimizer
plotDeliver = False     # create pictures of the path every 1/10 times a data block gets its destination
# plotSatID   = False     # If True, plots the ID of each satellite
GridSize    = 8         # Earth divided in GridSize rows for the grid. Used to be 15
winSize     = 20        # window size for the representation in the plots
markerSize  = 50        # Size of the markers in the plots
nTrain      = 2         # The DNN will train every nTrain steps
noPingPong  = True      # when a neighbour is the destination satellite, send there directly without going through the dnn (Change policy)

# Queues & State
infQueue    = 5000      # Upper boundary from where a queue is considered as infinite when obserbing the state
queueVals   = 10        # Values that the observed Queue can have, being 0 the best (Queue of 0) and max the worst (Huge queue or inexistent link).
latBias     = 90        # This value is added to the latitude of each position in the state space. This can be done to avoid negative numbers
lonBias     = 180       # Same but with longitude

# rewards
ArriveReward= 50        # Reward given to the system in case it sends the data block to the satellite linked to the destination gateway
# w1          = 20        # rewards the getting to empty queues
# w2          = 20        # rewards getting closes phisycally   
# w4          = 5         # Normalization for the distance reward, for the traveled distance factor  
againPenalty= -10       # Penalty if the satellite sends the block to a hop where it has already been
unavPenalty = -10       # Penalty if the satellite tries to send the block to a direction where there is no linked satellite
biggestDist = -1        # Normalization factor for the distance reward. This is updated in the creation of the graph.
firstMove   = True      # The biggest slant range is only computed the first time in order to avoid this value to be variable
distanceRew = 4          # 1: Distance reward normalized to total distance.
                         # 2: Distance reward normalized to average moving possibilities
                         # 3: Distance reward normalized to maximum close up
                         # 4: Distance reward normalized by max isl distance ~3.700 km for Kepler constellation. This is the one used in the papers.
                         # 5: Only negative rewards proportional to traveled distance normalized by 1.000 km

# Deep Learning
MAX_EPSILON = 0.99      # Maximum value that the exploration parameter can have
MIN_EPSILON = 0.001     # Minimum value that the exploration parameter can have
LAMBDA      = 0.0005    # This value is used to decay the epsilon in the deep learning implementation
decayRate   = 4         # sets the epsilon decay in the deep learning implementatio. If higher, the decay rate is slower. If lower, the decay is faster
Clipnorm    = 1         # Maximum value to the nom of the gradients. Prevents the gradients of the model parameters with respect to the loss function becoming too large
hardUpdate  = 1         # if up, the Q-network weights are copied inside the target network every updateF iterations. if down, this is done gradually
updateF     = 1000      # every updateF updates, the Q-Network will be copied inside the target Network. This is done if hardUpdate is up
batchSize   = 16        # batchSize samples are taken from bufferSize samples to train the network
bufferSize  = 50        # bufferSize samples are used to train the network

# Stop Loss
# Train       = True      # Global for all scenarios with different number of GTs. if set to false, the model will not train any of them
stopLoss    = False     # activates the stop loss function
nLosses     = 50        # Nº of loss samples used for the stop loss
lThreshold  = 0.5       # If the mean of the last nLosses are lower than lossThreshold, the mdoel stops training
TrainThis   = Train     # Local for a single scenario with a certain number of GTs. If the stop loss is activated, this will be set to False and the scenario will not train anymore. 
                        # When another scenario is about to run, TrainThis will be set to Train again

# Other
CurrentGTnumber = -1    # Number of active gateways. This number will be updated every time a gateway is added. In the simulation it will iterate the GTs list

###############################################################################
###############################      Paths      ###############################
###############################################################################

# nnpath      = './pre_trained_NNs/qNetwork_8GTs_6secs_nocon.h5'
# nnpathTarget= './pre_trained_NNs/qTarget_8GTs_6secs_nocon.h5'
# nnpath      = './pre_trained_NNs/qNetwork_3GTs.h5'
# nnpathTarget= './pre_trained_NNs/qTarget_3GTs.h5'
nnpath      = './pre_trained_NNs/qNetwork_2GTs.h5'
nnpathTarget= './pre_trained_NNs/qTarget_2GTs.h5'
# nnpath      = './pre_trained_NNs/qNetwork_2GTs_lastHop.h5'
# nnpathTarget= './pre_trained_NNs/qTarget_2GTs_lastHop.h5'
tablesPath  = './pre_trained_NNs/qTablesExport_8GTs/'

if __name__ == '__main__':
    # nnpath          = f'./pre_trained_NNs/qNetwork_8GTs.h5'
    outputPath      = './Results/{}_{}s_[{}]_Del_[{}]_w1_[{}]_w2_{}_GTs/'.format(pathing, float(pd.read_csv("inputRL.csv")['Test length'][0]), ArriveReward, w1, w2, GTs)
    populationMap   = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'

###############################################################################
#################################    Simpy    #################################
###############################################################################

receivedDataBlocks  = []
createdBlocks       = []
seed                = np.random.seed(1)
upGSLRates          = []
downGSLRates        = []
interRates          = []
intraRate           = []


def getBlockTransmissionStats(timeToSim, GTs, constellationType, earth):
    '''
    General Block transmission stats
    '''
    allTransmissionTimes = []
    largestTransmissionTime = (0, None)
    mostHops = (0, None)
    queueLat = []
    txLat = []
    propLat = []
    # latencies = [queueLat, txLat, propLat]
    blocks = []
    allLatencies= []
    pathBlocks  = [[],[]]
    first       = earth.gateways[0]
    second      = earth.gateways[1]

    earth.pathParam

    for block in receivedDataBlocks:
        time = block.getTotalTransmissionTime()
        hops = len(block.checkPoints)
        blocks.append(BlocksForPickle(block))

        if largestTransmissionTime[0] < time:
            largestTransmissionTime = (time, block)

        if mostHops[0] < hops:
            mostHops = (hops, block)

        allTransmissionTimes.append(time)

        queueLat.append(block.getQueueTime()[0])
        txLat.append(block.txLatency)
        propLat.append(block.propLatency)
        
        # [creation time, total latency, arrival time, source, destination, block ID, queue time, transmission latency, prop latency]
        allLatencies.append([block.creationTime, block.totLatency, block.creationTime+block.totLatency, block.source.name, block.destination.name, block.ID, block.getQueueTime()[0], block.txLatency, block.propLatency])
        # pre-process the received data blocks. create the rows that will be saved in csv
        if block.source == first and block.destination == second:
            pathBlocks[0].append([block.totLatency, block.creationTime+block.totLatency])
            pathBlocks[1].append(block)
        
    # save congestion test data
    # blockPath = f"./Results/Congestion_Test/{pathing} {float(pd.read_csv('inputRL.csv')['Test length'][0])}/"
    print('Saving congestion test data...\n')
    blockPath = outputPath + '/Congestion_Test/'     
    os.makedirs(blockPath, exist_ok=True)
    try:
        global CurrentGTnumber
        np.save("{}blocks_{}".format(blockPath, CurrentGTnumber), np.asarray(blocks),allow_pickle=True)
    except pickle.PicklingError:
        print('Error with pickle and profiling')

    avgTime = np.mean(allTransmissionTimes)
    totalTime = sum(allTransmissionTimes)

    print("\n########## Results #########\n")
    print(f"The simulation took {timeToSim} seconds to run")
    print(f"A total of {len(createdBlocks)} data blocks were created")
    print(f"A total of {len(receivedDataBlocks)} data blocks were transmitted")
    print(f"A total of {len(createdBlocks) - len(receivedDataBlocks)} data blocks were stuck")
    print(f"Average transmission time for all blocks were {avgTime}")
    print('Total latecies:\nQueue time: {}%\nTransmission time: {}%\nPropagation time: {}%'.format(
        '%.4f' % float(sum(queueLat)/totalTime*100),
        '%.4f' % float(sum(txLat)/totalTime*100),
        '%.4f' % float(sum(propLat)/totalTime*100)))

    results = Results(finishedBlocks=blocks,
                      constellation=constellationType,
                      GTs=GTs,
                      meanTotalLatency=avgTime,
                      meanQueueLatency=np.mean(queueLat),
                      meanPropLatency=np.mean(propLat),
                      meanTransLatency=np.mean(txLat),
                      perQueueLatency = sum(queueLat)/totalTime*100,
                      perPropLatency = sum(propLat)/totalTime*100,
                      perTransLatency = sum(txLat)/totalTime*100)

    return results, allLatencies, pathBlocks, blocks


def simProgress(simTimelimit, env):
    timeSteps = 100
    timeStepSize = simTimelimit/timeSteps
    progress = 1
    startTime = time.time()
    yield env.timeout(timeStepSize)
    while True:
        elapsedTime = time.time() - startTime
        estimatedTimeRemaining = elapsedTime * (timeSteps/progress) - elapsedTime
        print("Simulation progress: {}% Estimated time remaining: {} seconds Current simulation time: {}".format(progress, int(estimatedTimeRemaining), env.now), end='\r')
        yield env.timeout(timeStepSize)
        progress += 1


###############################################################################
############################# Federated Learning ##############################
###############################################################################

FL_techs    = ['nothing', 'modelAnticipation', 'plane', 'full', 'combination']
FL_tech     = FL_techs[4]# dataRateOG is the original datarate. If we want to maximize the datarate we have to use dataRate, which is the inverse of the datarate
if FL_tech == 'combination':
    global FL_counter
    FL_counter = 1

if pathing != 'Deep Q-Learning':
    FL_Test = False

if FL_Test:
    CKA_Values = []     # CKA matrix 
    num_samples = 10   # number of random samples to test the divergence between models
    print(f'Federated Learning ongoing: {FL_tech}. Number of random samples to test divergence: {num_samples}')

def generate_test_data(num_samples, include_not_avail=False):
    data = []
    queue_values = np.arange(0, 11)  # Possible queue values from 0 to 10
    # Set probabilities: 0 at 35%, 10 at 20%, and 5% each for values 1-9
    queue_probs = [0.35] + [0.05] * 9 + [0.20]

    for _ in range(num_samples):
        sample = []
        if diff_lastHop:
            sample.append(random.randint(0, 4))
        # Queue Scores for each direction: Up, Down, Right, Left (4 scores each)
        for _ in range(4):
            # Queue scores biased towards 0 and 10
            sample.extend(np.random.choice(queue_values, 4, p=queue_probs))
            
            # Relative positions for each direction: latitude and longitude
            sample.append(np.random.uniform(-2, 2))  # Latitude relative position
            sample.append(np.random.uniform(-2, 2))  # Longitude relative position
        
        # Absolute positions
        sample.append(np.random.uniform(0, 9))  # Absolute latitude normalized
        sample.append(np.random.uniform(0, 18))  # Absolute longitude normalized
        
        # Destination differential coordinates
        sample.append(np.random.uniform(-2, 2))  # Destination differential latitude
        sample.append(np.random.uniform(-2, 2))  # Destination differential longitude
        
        # Optionally include not available values
        if include_not_avail and np.random.rand() < 0.1:  # 10% chance to introduce a -1 value
            idx_to_replace = np.random.choice(len(sample), int(0.1 * len(sample)), replace=False)
            sample[idx_to_replace] = -1
        
        data.append(sample)
    
    return np.array(data)

def get_models(earth):
    models = []
    model_names = []
    for plane in earth.LEO:
        for sat in plane.sats:
            models.append(sat.DDQNA.qNetwork)
            model_names.append(sat.ID)
    return models, model_names

def average_model_weights(models):
    """Average weights of multiple trained models."""
    weights = [model.get_weights() for model in models]
    new_weights = [np.mean(np.array(w), axis=0) for w in zip(*weights)]
    return new_weights

def full_federated_learning(models):
    averaged_weights = average_model_weights(models)
    for model in models:
        model.set_weights(averaged_weights)

def federate_by_plane(models, model_names):
    """Perform Federated Averaging within each orbital plane."""
    plane_dict = {}
    for model, name in zip(models, model_names):
        plane = name.split('_')[0]
        if plane in plane_dict:
            plane_dict[plane].append(model)
        else:
            plane_dict[plane] = [model]
    for plane_models in plane_dict.values():
        averaged_weights = average_model_weights(plane_models)
        for model in plane_models:
            model.set_weights(averaged_weights)

def model_anticipation_federate(models, model_names):
    """Perform Model Anticipation Federated Learning."""
    plane_dict = {}
    # Group models by orbital plane
    for model, name in zip(models, model_names):
        plane = name.split('_')[0]
        if plane not in plane_dict:
            plane_dict[plane] = []
        plane_dict[plane].append((model, name))
    
    # Process each plane for model anticipation
    for plane_models in plane_dict.values():
        # Sort models by their identifiers within the plane
        plane_models.sort(key=lambda x: int(x[1].split('_')[1]))
        for i in range(1, len(plane_models)):
            prev_model_weights = plane_models[i - 1][0].get_weights()
            current_model = plane_models[i][0]
            current_weights = current_model.get_weights()
            # Average weights from the previous model
            new_weights = [(w1 + w2) / 2 for w1, w2 in zip(current_weights, prev_model_weights)]
            current_model.set_weights(new_weights)

def update_sats_models(earth, models, model_names):
    '''Update each satellite model for the updated one'''
    print('Updating satellites models...')
    for model, satID in zip(models, model_names):
        sat = findByID(earth, satID)
        sat.DDQNA.qNetwork = model
        if ddqn:
            sat.DDQNA.qTarget = model

def compute_full_cka_matrix(models, data):
    """Compute the full CKA matrix for a list of models."""
    
    def gram_matrix(X):
        """Calculate the Gram matrix from layer activations."""
        n = X.shape[0]
        X = X - X.mean(axis=0)
        return X @ X.T / n

    def cka(G, H):
        """Compute the CKA metric."""
        return np.trace(G @ H) / np.sqrt(np.trace(G @ G) * np.trace(H @ H))

    def compute_cka(model1, model2, data):
        """Compute the CKA between layers of two models using data."""
        intermediate_model1 = tf.keras.Model(inputs=model1.input, outputs=[layer.output for layer in model1.layers])
        intermediate_model2 = tf.keras.Model(inputs=model2.input, outputs=[layer.output for layer in model2.layers])
        activations1 = intermediate_model1(data)
        activations2 = intermediate_model2(data)
        return np.mean([cka(gram_matrix(np.array(act1)), gram_matrix(np.array(act2))) for act1, act2 in zip(activations1, activations2)])
    
    n = len(models)
    cka_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                cka_matrix[i, j] = 1.0
            else:
                cka_matrix[i, j] = cka_matrix[j, i] = compute_cka(models[i], models[j], data)
    return cka_matrix

def compute_average_cka(cka_matrix):
    """Compute the average CKA value from a CKA matrix."""
    triu_indices = np.triu_indices_from(cka_matrix, k=1)
    return np.mean(cka_matrix[triu_indices])

def perform_FL(earth):#, outputPath):

    # path = outputPath + 'FL' + str(len(earth.gateways)) + 'GTs/'
    # os.makedirs(path, exist_ok=True) 
    print('----------------------------------')
    print(f'Federated Learning. Performing: {FL_tech}')

    data = generate_test_data(num_samples, include_not_avail=False)
    models, model_names = get_models(earth)

    CKA_Values_before = compute_full_cka_matrix(models, data)

    if FL_tech == 'nothing':
        return CKA_Values_before, CKA_Values_before
        
    if FL_tech == 'modelAnticipation':
        model_anticipation_federate(models, model_names)
    elif FL_tech == 'plane':
        federate_by_plane(models, model_names)
    elif FL_tech == 'full':
        full_federated_learning(models)
    elif FL_tech == 'combination':
        global FL_counter
        if FL_counter == 1:
            print(f'Model Anticipation, counter = {FL_counter}')
            FL_counter += 1
            model_anticipation_federate(models, model_names)

        elif FL_counter == 2:
            print(f'Plane FL, counter = {FL_counter}')
            FL_counter += 1
            federate_by_plane(models, model_names)

        elif FL_counter > 2:
            print(f'Global FL, counter = {FL_counter}')
            FL_counter = 1
            full_federated_learning(models)

    CKA_Values_after = compute_full_cka_matrix(models, data)
    update_sats_models(earth, models, model_names)

    print('----------------------------------')
    return CKA_Values_before, CKA_Values_after

def plot_cka_over_time_v0(cka_data, outputPath, nGTs):
    """
    Plots each CKA value over time in milliseconds, connecting 'before' and 'after' points with a line
    and using different colors for each type of dot.
    
    Parameters:
    - cka_data: List of [CKA_before, CKA_after, timestamp] entries.
    """
    path = outputPath + 'FL/'
    os.makedirs(path, exist_ok=True) # create output path

    # Extract times and CKA values for before and after
    times = [entry[2] * 1000 for entry in cka_data]  # Convert time to milliseconds
    cka_before_values = [compute_average_cka(entry[0]) for entry in cka_data]
    cka_after_values = [compute_average_cka(entry[1]) for entry in cka_data]

    # Construct the sequence for line plot: interleave before and after values
    line_times = [time for time in times for _ in (0, 1)]
    line_values = [val for pair in zip(cka_before_values, cka_after_values) for val in pair]

    # Plotting
    plt.figure(figsize=(10, 6))

    # Line connecting all CKA values
    plt.plot(line_times, line_values, label='CKA Value Sequence', color='gray', linestyle='--', alpha=0.7)

    # Dots for 'CKA Before FL' and 'CKA After FL'
    plt.scatter(times, cka_before_values, label='CKA Before FL', color='blue', marker='o')
    plt.scatter(times, cka_after_values, label='CKA After FL', color='green', marker='s')

    # Labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('CKA Value')
    plt.title('CKA Values Over Time (ms) with Sequential Connection and Dot Types')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.tight_layout()
    plt.savefig(path + f'CKA_over_time_{str(nGTs)}_GTs', dpi=300, bbox_inches='tight')   

    # Save mean CKA values over time
    mean_cka_values = np.column_stack((times, cka_before_values, cka_after_values))
    np.savetxt(os.path.join(path, 'mean_cka_values.csv'), mean_cka_values, delimiter=',', 
               header="Time_ms,CKA_Before,CKA_After", comments='')

    # Save individual CKA matrices before and after FL for each timestamp
    for i, entry in enumerate(cka_data):
        np.savetxt(os.path.join(path, f'cka_matrix_before_{i}.csv'), entry[0], delimiter=',')
        np.savetxt(os.path.join(path, f'cka_matrix_after_{i}.csv'), entry[1], delimiter=',')

def plot_cka_over_time(cka_data, outputPath, nGTs):
    """
    Plots each CKA value over time in milliseconds, connecting 'before' and 'after' points with a dashed line
    and using different colors for each type of dot, with quartile ranges represented by error bars.
    
    Parameters:
    - cka_data: List of [CKA_before, CKA_after, timestamp] entries.
    """
    path = outputPath + 'FL/'
    os.makedirs(path, exist_ok=True)  # create output path

    # Extract times and calculate CKA values for before and after
    times = [entry[2] * 1000 for entry in cka_data]  # Convert time to milliseconds
    cka_before_values = [np.mean(entry[0]) for entry in cka_data]
    cka_after_values = [np.mean(entry[1]) for entry in cka_data]

    # Calculate quartile ranges for before and after values
    cka_before_quartiles = [np.percentile(entry[0], [25, 75]) for entry in cka_data]
    cka_after_quartiles = [np.percentile(entry[1], [25, 75]) for entry in cka_data]
    cka_before_25th, cka_before_75th = zip(*cka_before_quartiles)
    cka_after_25th, cka_after_75th = zip(*cka_after_quartiles)

    # Construct the sequence for line plot: interleave before and after mean values
    line_times = [time for time in times for _ in (0, 1)]
    line_values = [val for pair in zip(cka_before_values, cka_after_values) for val in pair]

    # Set y-axis limits with margin to avoid cutting T-caps and ensure the max is exactly 1
    y_min = min(min(cka_before_25th), min(cka_after_25th)) * 0.95
    y_max = 1

    # Plotting
    plt.figure(figsize=(10, 6))

    # Line connecting mean CKA values
    plt.plot(line_times, line_values, label='CKA Value Sequence', color='gray', linestyle='-.', alpha=0.7)

    # Error bars for 'CKA Before FL' and 'CKA After FL' with T-caps
    cka_before_yerr = [np.abs(np.subtract(cka_before_values, cka_before_25th)), 
                       np.abs(np.subtract(cka_before_75th, cka_before_values))]
    cka_after_yerr = [np.abs(np.subtract(cka_after_values, cka_after_25th)), 
                      np.abs(np.subtract(cka_after_75th, cka_after_values))]

    plt.errorbar(times, cka_before_values, yerr=cka_before_yerr, fmt='s', color='blue', 
                 ecolor='blue', capsize=8, capthick=2, label='CKA Before FL Quartiles')
    plt.errorbar(times, cka_after_values, yerr=cka_after_yerr, fmt='s', color='green', 
                 ecolor='green', capsize=8, capthick=2, label='CKA After FL Quartiles')

    # Set x-axis and y-axis limits with a dynamic y-axis minimum
    plt.xlim(min(times) - 20, max(times) + 20)
    # plt.ylim(y_min, y_max)
    plt.ticklabel_format(style='plain', axis='y')  # Disable scientific notation for y-axis

    # Labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('CKA Value')
    plt.title('CKA Values Over Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'CKA_over_time_{str(nGTs)}_GTs.png'), dpi=300, bbox_inches='tight')

    # Save mean CKA values over time
    mean_cka_values = np.column_stack((times, cka_before_values, cka_after_values))
    np.savetxt(os.path.join(path, 'mean_cka_values.csv'), mean_cka_values, delimiter=',', 
               header="Time_ms,CKA_Before,CKA_After", comments='')

    # Save individual CKA matrices before and after FL for each timestamp
    for i, entry in enumerate(cka_data):
        np.savetxt(os.path.join(path, f'cka_matrix_before_{i}.csv'), entry[0], delimiter=',')
        np.savetxt(os.path.join(path, f'cka_matrix_after_{i}.csv'), entry[1], delimiter=',')


###############################################################################
###############################     Classes    ################################
###############################################################################


class Results:
    def __init__(self, finishedBlocks, constellation, GTs, meanTotalLatency, meanQueueLatency, meanTransLatency, meanPropLatency, perQueueLatency, perPropLatency,perTransLatency):

        self.GTs = GTs
        self.finishedBlocks = finishedBlocks
        self.constellation = constellation
        self.meanTotalLatency = meanTotalLatency
        self.meanQueueLatency = meanQueueLatency
        self.meanPropLatency = meanPropLatency
        self.meanTransLatency = meanTransLatency
        self.perQueueLatency = perQueueLatency
        self.perPropLatency = perPropLatency
        self.perTransLatency = perTransLatency


class BlocksForPickle:
    def __init__(self, block):
        self.size = BLOCK_SIZE  # size in bits
        self.ID = block.ID  # a string which holds the source id, destination id, and index of the block, e.g. "1_2_12"
        self.timeAtFull = block.timeAtFull  # the simulation time at which the block was full and was ready to be sent.
        self.creationTime = block.creationTime  # the simulation time at which the block was created.
        self.timeAtFirstTransmission = block.timeAtFirstTransmission  # the simulation time at which the block left the GT.
        self.checkPoints = block.checkPoints  # list of simulation reception times at node with the first entry being the reception time at first sat - can be expanded to include the sat IDs at each checkpoint
        self.checkPointsSend = block.checkPointsSend  # list of times after the block was sent at each node
        self.path = block.path
        self.queueLatency = block.queueLatency  # total time acumulated in the queues
        self.txLatency = block.txLatency  # total transmission time
        self.propLatency = block.propLatency  # total propagation latency
        self.totLatency = block.totLatency  # total latency
        self.QPath = block.QPath # path followed due to Q-Learning


class RFlink:
    def __init__(self, frequency, bandwidth, maxPtx, aDiameterTx, aDiameterRx, pointingLoss, noiseFigure,
                 noiseTemperature, min_rate):
        self.f = frequency
        self.B = bandwidth
        self.maxPtx = maxPtx
        self.maxPtx_db = 10 * math.log10(self.maxPtx)
        self.Gtx = 10 * math.log10(eff * ((math.pi * aDiameterTx * self.f / Vc) ** 2))
        self.Grx = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2))
        self.G = self.Gtx + self.Grx - 2 * pointingLoss
        self.No = 10 * math.log10(self.B * k) + noiseFigure + 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.GoT = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2)) - noiseFigure - 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.min_rate = min_rate

    def __repr__(self):
        return '\n Carrier frequency = {} GHz\n Bandwidth = {} MHz\n Transmission power = {} W\n Gain per antenna: Tx {}  Rx {}\n Total antenna gain = {} dB\n Noise power = {} dBW\n G/T = {} dB/K'.format(
            self.f / 1e9,
            self.B / 1e6,
            self.maxPtx,
            '%.2f' % self.Gtx,
            '%.2f' % self.Grx,
            '%.2f' % self.G,
            '%.2f' % self.No,
            '%.2f' % self.GoT,
        )


class FSOlink:
    def __init__(self, data_rate, power, comm_range, weight):
        self.data_rate = data_rate
        self.power = power
        self.comm_range = comm_range
        self.weight = weight

    def __repr__(self):
        return '\n Data rate = {} Mbps\n Power = {} W\n Transmission range = {} km\n Weight = {} kg'.format(
            self.data_rate / 1e6,
            self.power,
            self.comm_range / 1e3,
            self.weight)


class OrbitalPlane:
    def __init__(self, ID, h, longitude, inclination, n_sat, min_elev, firstID, env, earth):
        self.ID = ID 								# A unique ID given to every orbital plane = index in Orbital_planes, string
        self.h = h									# Altitude of deployment
        self.longitude = longitude					# Longitude angle where is intersects equator [radians]
        self.inclination = math.pi/2 - inclination	# Inclination of the orbit form [radians]
        self.n_sat = n_sat							# Number of satellites in plane
        self.period = 2 * math.pi * math.sqrt((self.h+Re)**3/(G*Me))	# Orbital period of the satellites in seconds
        self.v = 2*math.pi * (h + Re) / self.period						# Orbital velocity of the satellites in m/s
        self.min_elev = math.radians(min_elev)							# Minimum elevation angle for ground comm.
        self.max_alpha = math.acos(Re*math.cos(self.min_elev)/(self.h+Re))-self.min_elev	# Maximum angle at the center of the Earth w.r.t. yaw
        self.max_beta  = math.pi/2-self.max_alpha-self.min_elev								# Maximum angle at the satellite w.r.t. yaw
        self.max_distance_2_ground = Re*math.sin(self.max_alpha)/math.sin(self.max_beta)	# Maximum distance to a servable ground station
        self.earth = earth

        # Adding satellites
        self.first_sat_ID = firstID # Unique ID of the first satellite in the orbital plane

        self.sats = []              # List of satellites in the orbital plane
        for i in range(n_sat):
            self.sats.append(Satellite(self.first_sat_ID + str(i), int(self.ID), int(i), self.h, self.longitude, self.inclination, self.n_sat, env, self))

        self.last_sat_ID = self.first_sat_ID + str(len(self.sats) - 1) # Unique ID of the last satellite in the orbital plane

    def __repr__(self):
        return '\nID = {}\n altitude= {} km\n longitude= {} deg\n inclination= {} deg\n number of satellites= {}\n period= {} hours\n satellite speed= {} km/s'.format(
            self.ID,
            self.h/1e3,
            '%.2f' % math.degrees(self.longitude),
            '%.2f' % math.degrees(self.inclination),
            '%.2f' % self.n_sat,
            '%.2f' % (self.period/3600),
            '%.2f' % (self.v/1e3))

    def rotate(self, delta_t):
        """
        Rotates the orbit according to the elapsed time by adjusting the longitude. The amount the longitude is adjusted
        is based on the fraction the elapsed time makes up of the time it takes the Earth to complete a full rotation.
        """

        # Change in longitude and phi due to Earth's rotation
        self.longitude = self.longitude + 2*math.pi*delta_t/Te
        self.longitude = self.longitude % (2*math.pi)
        # Rotating every satellite in the orbital plane
        for sat in self.sats:
            sat.rotate(delta_t, self.longitude, self.period)


# @profile
class Satellite:
    def __init__(self, ID, in_plane, i_in_plane, h, longitude, inclination, n_sat, env, orbitalPlane, quota = 500, power = 10):
        self.ID = ID                    # A unique ID given to every satellite
        self.orbPlane = orbitalPlane    # Pointer to the orbital plane which the sat belongs to
        self.in_plane = in_plane        # Orbital plane where the satellite is deployed
        self.i_in_plane = i_in_plane    # Index in orbital plane
        self.quota = quota              # Quota of the satellite
        self.h = h                      # Altitude of deployment
        self.power = power              # Transmission power
        self.minElevationAngle = minElAngle# Value is taken from NGSO constellation design chapter

        # Spherical Coordinates before inclination (r,theta,phi)
        self.r = Re+self.h
        self.theta = 2 * math.pi * self.i_in_plane / n_sat
        self.phi = longitude

        # Inclination of the orbital plane
        self.inclination = inclination

        # Cartesian coordinates  (x,y,z)
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)

        self.polar_angle = self.theta               # Angle within orbital plane [radians]
        self.latitude = math.asin(self.z/self.r)   # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y/self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi/2
        elif self.y < 0:
            self.longitude = -math.pi/2
        else:
            self.longitude = 0

        self.waiting_list = {}
        self.applications = []
        self.n_sat = n_sat

        self.ngeo2gt = RFlink(f, B, maxPtx, Adtx, Adrx, pL, Nf, Tn, min_rate)
        self.downRate = 0

        # simpy
        self.env = env
        self.sendBufferGT = ([env.event()], [])  # ([self.env.event()], [DataBlock(0, 0, "0", 0)])
        self.sendBlocksGT = []  # env.process(self.sendBlock())  # simpy processes which send the data blocks
        self.sats = []
        self.linkedGT = None
        self.GTDist = None
        # list of data blocks waiting on their propagation delay.
        self.tempBlocks = []  # This list is used to so the block can have their paths changed when the constellation is moved

        self.intraSats = []
        self.interSats = []
        self.sendBufferSatsIntra = []
        self.sendBufferSatsInter = []
        self.sendBlocksSatsIntra = []
        self.sendBlocksSatsInter = []
        self.newBuffer  = [False]

        self.QLearning  = None  # Q-learning table that will be updated in case the pathing is 'Q-Learning'
        self.DDQNA      = None  # DDQN agent for each satellite. Only used in the online phase
        self.maxSlantRange = self.GetmaxSlantRange()

    def GetmaxSlantRange(self):
        """
        Maximum distance from satellite to edge of coverage area is calculated using the following formula:
        D_max(minElevationAngle, h) = sqrt(Re**2*sin**2(minElevationAngle) + 2*Re*h + h**2) - Re*sin(minElevationAngle)
        This formula is based on the NGSO constellation design chapter page 16.
        """
        eps = math.radians(self.minElevationAngle)

        distance = math.sqrt((Re+self.h)**2-(Re*math.cos(eps))**2) - Re*math.sin(eps)

        return distance

    def __repr__(self):
        return '\nID = {}\n orbital plane= {}, index in plane= {}, h={}\n pos r = {}, pos theta = {},' \
               ' pos phi = {},\n pos x= {}, pos y= {}, pos z= {}\n inclination = {}\n polar angle = {}' \
               '\n latitude = {}\n longitude = {}'.format(
                self.ID,
                self.in_plane,
                self.i_in_plane,
                '%.2f' % self.h,
                '%.2f' % self.r,
                '%.2f' % self.theta,
                '%.2f' % self.phi,
                '%.2f' % self.x,
                '%.2f' % self.y,
                '%.2f' % self.z,
                '%.2f' % math.degrees(self.inclination),
                '%.2f' % math.degrees(self.polar_angle),
                '%.2f' % math.degrees(self.latitude),
                '%.2f' % math.degrees(self.longitude))

    def createReceiveBlockProcess(self, block, propTime):
        """
        Function which starts a receiveBlock process upon receiving a block from a transmitter.
        """
        process = self.env.process(self.receiveBlock(block, propTime))

    def receiveBlock(self, block, propTime):
        """
        Simpy process function:

        This function is used to handle the propagation delay of data blocks. This is done simply by waiting the time
        of the propagation delay and adding the block to the send-buffer afterwards. Since there are multiple buffers,
        this function looks at the next step in the blocks path and adds the block to the correct send-buffer.
        When Q-Learning or Deep learning is used, this function is where the next step in the block's path is found.

        While the transmission delay is handled at the transmitter, the transmitter cannot also wait for the propagation
        delay, otherwise the send-buffer might be overfilled.

        Using this structure, if there are to be implemented limits on the sizes of the "receive-buffer" it could be
        handled by either limiting the amount of these processes that can occur at the same time, or limiting the size
        of the send-buffer.
        """
        # wait for block to fully propagate
        self.tempBlocks.append(block)

        yield self.env.timeout(propTime)

        if block.path == -1:
            return

        # KPI: propLatency receive block from sat
        block.propLatency += propTime

        for i, tempBlock in enumerate(self.tempBlocks):
            if block.ID == tempBlock.ID:
                self.tempBlocks.pop(i)
                break

        try: # ANCHOR Save Queue time csv
            block.queueTime.append((block.checkPointsSend[len(block.checkPointsSend)-1]- block.checkPoints[len(block.checkPoints)-1]))
        except IndexError:  # Either it is the first satellite for the datablock or the datablock has no checkpoints appendeds
            # print('Index error')
            pass

        block.checkPoints.append(self.env.now)

        # if QLearning or Deep Q-Learning we:
        # Compute the next hop in the path and add it to the second last position (Last is the destination gateway)
        # we let the (Deep) Q-model choose the next hop and it will be added to the block.QPath as mentioned
        # if the next hop is the linked gateway it will simply not add anything and will let the model work normally
        if ((self.QLearning) or (self.orbPlane.earth.DDQNA is not None) or (self.DDQNA is not None)):
            if len(block.QPath) > 3: # the block does not come from a gateway
                if self.QLearning:
                    nextHop = self.QLearning.makeAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth, prevSat = (findByID(self.orbPlane.earth, block.QPath[len(block.QPath)-3][0])))
                elif self.DDQNA:
                    nextHop = self.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth, prevSat = (findByID(self.orbPlane.earth, block.QPath[len(block.QPath)-3][0])))
                else:
                    nextHop = self.orbPlane.earth.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth, prevSat = (findByID(self.orbPlane.earth, block.QPath[len(block.QPath)-3][0])))
            else:
                if self.QLearning:
                    nextHop = self.QLearning.makeAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth)
                elif self.DDQNA:
                    nextHop = self.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth)
                else:
                    nextHop = self.orbPlane.earth.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth)

            if nextHop != 0:
                block.QPath.insert(len(block.QPath)-1 ,nextHop)
                pathPlot = block.QPath.copy()
                pathPlot.pop()
            else:
                pathPlot = block.QPath.copy()
            
            # If plotPath plots an image for every action taken. Plots 1/10 of blocks. # ANCHOR plot action satellite
            #################################################################
            if self.orbPlane.earth.plotPaths:
                if int(block.ID[len(block.ID)-1]) == 0:
                    os.makedirs(self.orbPlane.earth.outputPath + '/pictures/', exist_ok=True) # create output path
                    outputPath = self.orbPlane.earth.outputPath + '/pictures/' + block.ID + '_' + str(len(block.QPath)) + '_'
                    # plotShortestPath(self.orbPlane.earth, pathPlot, outputPath)
                    plotShortestPath(self.orbPlane.earth, pathPlot, outputPath, ID=block.ID, time = block.creationTime)
            #################################################################

            path = block.QPath  # if there is Q-Learning the path will be repalced with the QPath
        else:
            path = block.path   # if there is no Q-Learning we will work with the path as normally

        # get this satellites index in the blocks path
        index = None
        for i, step in enumerate(path):
            if self.ID == step[0]:
                index = i

        if not index:
            print(path)

        # check if next step in path is GT (last step in path)
        if index == len(path) - 2:
            # add block to GT send-buffer
            if not self.sendBufferGT[0][0].triggered:
                self.sendBufferGT[0][0].succeed()
                self.sendBufferGT[1].append(block)
            else:
                newEvent = self.env.event().succeed()
                self.sendBufferGT[0].append(newEvent)
                self.sendBufferGT[1].append(block)

        else:
            ID = None
            isIntra = False
            # get ID of next sat
            for sat in self.intraSats:
                id = sat[1].ID
                if id == path[index + 1][0]:
                    ID = sat[1].ID
                    isIntra = True
            for sat in self.interSats:
                id = sat[1].ID
                if id == path[index + 1][0]:
                    ID = sat[1].ID

            if ID is not None:
                sendBuffer = None
                # find send-buffer for the satellite
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if ID == buffer[2]:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if ID == buffer[2]:
                            sendBuffer = buffer
                # ANCHOR save the queue length that the block found at its next hop
                self.orbPlane.earth.queues.append(len(sendBuffer[1]))
                block.queue.append(len(sendBuffer[1]))

                # add block to buffer
                if not sendBuffer[0][0].triggered:
                    sendBuffer[0][0].succeed()
                    sendBuffer[1].append(block)
                else:
                    newEvent = self.env.event().succeed()
                    sendBuffer[0].append(newEvent)
                    sendBuffer[1].append(block)

            else:
                print(
                    "ERROR! Sat {} tried to send block to {} but did not have it in its linked satellite list".format(
                        self.ID, path[index + 1][0]))

    def sendBlock(self, destination, isSat, isIntra = None):
        """
        Simpy process function:

        Sends data blocks that are filled and added to one of the send-buffers, a buffer which consists of a list of
        events and data blocks. Since there are multiple send-buffers, the function finds the correct buffer given
        information regarding the desired destination satellite or GT. The function monitors the send-buffer, and when
        the buffer contains one or more triggered events, the function will calculate the time it will take to send the
        block and trigger an event which notifies a separate process that a block has been sent.

        A process is running this method for each ISL and for the downLink GSL the satellite has. This will usually be
        4 ISL processes and 1 GSL process.
        """

        if isIntra is not None:
            sendBuffer = None
            if isSat:
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
        else:
            sendBuffer = self.sendBufferGT

        while True:
            try:
                yield sendBuffer[0][0]

                # ANCHOR KPI: queueLatency at sat
                sendBuffer[1][0].checkPointsSend.append(self.env.now)

                if isSat:
                    timeToSend = sendBuffer[1][0].size / destination[2]

                    propTime = self.timeToSend(destination)
                    yield self.env.timeout(timeToSend)

                    receiver = destination[1]

                else:
                    propTime = self.timeToSend(self.linkedGT.linkedSat)
                    timeToSend = sendBuffer[1][0].size / self.downRate
                    yield self.env.timeout(timeToSend)

                    receiver = self.linkedGT

                # When the constellations move, the only case where this process can simply continue, is when the
                # receiver is the same, and there is a block already ready to be sent. The only place where the process
                # can continue from, is as a result right here. Furthermore, the only processes this can happen for are
                # the inter-ISL processes.
                # Due to having to remake buffers when the satellites move, it is necessary for the process to "find"
                # the correct buffer again - the process uses a reference to the buffer: "sendBuffer".
                # To avoid remaking the reference every time a block is sent, the list of boolean values: self.newBuffer
                # is used to indicate when the constellation is moved,

                if True in self.newBuffer and not isIntra and isSat: # remake reference to buffer
                    if isIntra is not None:
                        sendBuffer = None
                        if isSat:
                            if isIntra:
                                for buffer in self.sendBufferSatsIntra:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                            else:
                                for buffer in self.sendBufferSatsInter:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                    else:
                        sendBuffer = self.sendBufferGT

                    for index, val in enumerate(self.newBuffer):
                        if val: # each process will one by one remake their reference, and change one value to True.
                                # After all processes has done this, all values are back to False
                            self.newBuffer[index] = False
                            break

                # ANCHOR KPI: txLatency ISL
                sendBuffer[1][0].txLatency += timeToSend
                receiver.createReceiveBlockProcess(sendBuffer[1][0], propTime)

                # remove from own buffer
                if len(sendBuffer[0]) == 1:
                    sendBuffer[0].pop(0)
                    sendBuffer[1].pop(0)
                    sendBuffer[0].append(self.env.event())

                else:
                    sendBuffer[0].pop(0)
                    sendBuffer[1].pop(0)
            except simpy.Interrupt:
                # print(f'Simpy interrupt at sending block at satellite {self.ID} to {destination[1].ID}') # FIXME Are they really lost blocks?
                # self.orbPlane.earth.lostBlocks+=1
                break

    def adjustDownRate(self):

        speff_thresholds = np.array(
            [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
             1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
             2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
             3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
             5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
        lin_thresholds = np.array(
            [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
             1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
             3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
             8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
             16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
             45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])
        db_thresholds = np.array(
            [-100.00000, -2.85000, -2.35000, -2.03000, -1.24000, -0.30000, 0.22000, 1.00000, 1.45000, 2.23000, 3.10000,
             4.03000, 4.68000, 4.73000, 5.13000, 5.50000, 5.97000, 6.55000, 6.84000, 7.41000, 8.10000, 8.38000, 8.43000,
             8.97000, 9.27000, 9.71000, 10.21000, 10.65000, 11.03000, 11.10000, 11.61000, 11.75000, 12.17000, 12.73000,
             13.05000, 13.64000, 13.98000, 14.81000, 15.47000, 15.87000, 16.55000, 16.98000, 17.24000, 18.10000,
             18.59000, 18.84000, 19.57000])

        pathLoss = 10*np.log10((4*math.pi*self.linkedGT.linkedSat[0]*self.ngeo2gt.f/Vc)**2)
        snr = 10**((self.ngeo2gt.maxPtx_db + self.ngeo2gt.G - pathLoss - self.ngeo2gt.No)/10)
        shannonRate = self.ngeo2gt.B*np.log2(1+snr)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.ngeo2gt.B * feasible_speffs[-1]

        self.downRate = speff

    def timeToSend(self, linkedSat):
        """
        Calculates the propagation time of a block going from satellite to satellite.
        """
        distance = linkedSat[0]
        pTime = distance/Vc
        return pTime

    def findIntraNeighbours(self, earth):
        '''
        Finds intra-plane neighbours
        '''
        self.linked = None                                                      # Closest sat linked
        self.upper  = earth.LEO[self.in_plane].sats[self.i_in_plane-1]          # Previous sat in the same plane
        if self.i_in_plane < self.n_sat-1:
            self.lower = earth.LEO[self.in_plane].sats[self.i_in_plane+1]       # Following sat in the same plane
        else:
            self.lower = earth.LEO[self.in_plane].sats[0]                       # last satellite of the plane

    def findInterNeighbours(self, earth):
        '''
        Sets the inter plane neighbors for each satellite that will be used for DRL
        '''
        g = earth.graph
        self.right = None
        self.left  = None
        # Find inter-plane neighbours (right and left)
        for edge in list(g.edges(self.ID)):
            if edge[1][0].isdigit():
                satB = findByID(earth, edge[1])
                dir = getDirection(self, satB)
                if(dir == 3):                                         # Found Satellite at East
                    # if self.right is not None:
                    #     print(f"{self.ID} east satellite duplicated! Replacing {self.right.ID} with {satB.ID}.")
                    self.right  = satB

                elif(dir == 4):                                       # Found Satellite at West
                    # if self.left is not None:
                    #     print(f"{self.ID} west satellite duplicated! Replacing {self.left.ID} with {satB.ID}.")
                    self.left  = satB
                elif(dir==1 or dir==2):
                    pass
                else:
                    print(f'Sat: {satB.ID} direction not found with respect to {self.ID}')
            else:   # it is a GT
                pass
        
    def rotate(self, delta_t, longitude, period):
        """
        Rotates the satellite by re-calculating the sperical coordinates, Cartesian coordinates, and longitude and
        latitude adjusted for the new longitude of the orbit, and fraction the elapsed time makes up of the orbit time
        of the satellite.
        """
        # Updating spherical coordinates upon rotation (these are phi, theta before inclination)
        self.phi = longitude
        self.theta = self.theta + 2*math.pi*delta_t/period
        self.theta = self.theta % (2*math.pi)

        # Calculating x,y,z coordinates with inclination
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)
        self.polar_angle = self.theta  # Angle within orbital plane [radians]
        # updating latitude and longitude after rotation [degrees]
        self.latitude = math.asin(self.z/self.r)  # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y/self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi/2
        elif self.y < 0:
            self.longitude = -math.pi/2
        else:
            self.longitude = 0


class edge:
    def  __init__(self, sati, satj, slant_range, dji, dij, shannonRate):
        '''
        dji && dij are deprecated. We do not use them anymore to decide which neighbour is at the right or left direction. We are using their coordinates.
        It is used in the markovian matching only
        '''
        self.i = sati   # sati ID
        self.j = satj   # satj ID
        self.slant_range = slant_range  # distance between both sats
        self.dji = dji  # direction from sati to satj
        self.dij = dij  # direction from sati to satj
        self.shannonRate = shannonRate  # max dataRate between sat1 and satj

    def  __repr__(self):
        return '\n node i: {}, node j: {}, slant_range: {}, shannonRate: {}'.format(
    self.i,
    self.j,
    self.slant_range,
    self.shannonRate)

    def __cmp__(self, other):
        if hasattr(other, 'slant_range'):    # returns true if has 'weight' attribute
            return self.slant_range.__cmp__(other.slant_range)


class DataBlock:
    """
    Class for outgoing block of data from the gateways.
    Instead of simulating the individual data packets from each user, data is gathered at the GTs in blocks - one for
    each destination GT. Once a block is filled with data it is sent as one unit to the destination GT.
    """

    def __init__(self, source, destination, ID, creationTime):
        self.size = BLOCK_SIZE  # size in bits
        self.destination = destination
        self.source = source
        self.ID = ID            # a string which holds the source id, destination id, and index of the block, e.g. "1_2_12"
        self.timeAtFull = None  # the simulation time at which the block was full and was ready to be sent.
        self.creationTime = creationTime  # the simulation time at which the block was created.
        self.timeAtFirstTransmission = None  # the simulation time at which the block left the GT.
        self.checkPoints = []   # list of simulation reception times at node with the first entry being the reception time at first sat - can be expanded to include the sat IDs at each checkpoint
        self.checkPointsSend = []   # list of times after the block was sent at each node
        self.path = []
        self.queueLatency = (None, None) # total time acumulated in the queues
        self.txLatency = 0      # total transmission time
        self.propLatency = 0    # total propagation latency
        self.totLatency = 0     # total latency
        self.isNewPath = False
        self.oldPath = []
        self.newPath = []
        self.QPath   = []
        self.queue   = []
        self.queueTime= []
        self.oldState  = None
        self.oldAction = None
        # self.oldReward = None

    def getQueueTime(self):
        '''
        The queue latency is computed in two steps:
        First one: time when the block is sent for the first time - time when the the block is created
        Rest of the steps: sum(checkpoint (Arrival time at node) - checkpointsSend (send time at previous node))
        '''
        queueLatency = [0, []]
        queueLatency[0] += self.timeAtFirstTransmission - self.creationTime        # ANCHOR queue first step
        queueLatency[1].append(self.timeAtFirstTransmission - self.creationTime)
        for arrived, sendReady in zip(self.checkPoints, self.checkPointsSend):  # rest of the steps
            queueLatency[0] += sendReady - arrived
            queueLatency[1].append(sendReady - arrived)

        self.queueLatency = queueLatency
        return queueLatency

    def getTotalTransmissionTime(self):
        totalTime = 0
        if len(self.checkPoints) == 1:
            return self.checkPoints[0] - self.timeAtFirstTransmission

        lastTime = self.creationTime
        for time in self.checkPoints:
            totalTime += time - lastTime
            lastTime = time
        # ANCHOR KPI: totLatency
        self.totLatency = totalTime
        return totalTime

    def __repr__(self):
        return'ID = {}\n Source:\n {}\n Destination:\n {}\nTotal latency: {}'.format(
            self.ID,
            self.source,
            self.destination,
            self.totLatency
        )


# @profile
class Gateway:
    """
    Class for the gateways (or concentrators). Each gateway will exist as an instance of this class
    which means that each ground station will have separate processes filling and sending blocks to all other GTs.
    """
    def __init__(self, name: str, ID: int, latitude: float, longitude: float, totalX: int, totalY: int, totalGTs, env, totalLocations, earth):
        self.name   = name
        self.ID     = ID
        self.earth  = earth
        self.latitude   = latitude  # number is already in degrees
        self.longitude  = longitude  # number is already in degrees

        # using the formulas from the set_window() function in the Earth class to the location in terms of cell grid.
        self.gridLocationX = int((0.5 + longitude / 360) * totalX)
        self.gridLocationY = int((0.5 - latitude / 180) * totalY)
        self.cellsInRange = []  # format: [ [(lat,long), userCount, distance], [..], .. ]
        self.totalGTs = totalGTs  # number of GTs including itself
        self.totalLocations = totalLocations # number of possible GTs
        self.totalAvgFlow = None  # total combined average flow from all users in bits per second
        self.totalX = totalX
        self.totalY = totalY

        # cartesian coordinates
        self.polar_angle = (math.pi / 2 - math.radians(self.latitude) + 2 * math.pi) % (2 * math.pi)  # Polar angle in radians
        self.x = Re * math.cos(math.radians(self.longitude)) * math.sin(self.polar_angle)
        self.y = Re * math.sin(math.radians(self.longitude)) * math.sin(self.polar_angle)
        self.z = Re * math.cos(self.polar_angle)

        # satellite linking structure
        self.satsOrdered = []
        self.satIndex = 0
        self.linkedSat = (None, None)  # (distance, sat)
        self.graph = nx.Graph()

        # simpy attributes
        self.env = env  # simulation environment
        self.datBlocks = []  # list of outgoing data blocks - one for each destination GT
        self.fillBlocks = []  # list of simpy processes which fills up the data blocks
        self.sendBlocks = env.process(self.sendBlock())  # simpy process which sends the data blocks
        self.sendBuffer = ([env.event()], [])  # queue of blocks that are ready to be sent
        self.paths = {}  # dictionary for destination: path pairs

        # comm attributes
        self.dataRate = None
        self.gs2ngeo = RFlink(
            frequency=30e9,
            bandwidth=500e6,
            maxPtx=20,
            aDiameterTx=0.33,
            aDiameterRx=0.26,
            pointingLoss=0.3,
            noiseFigure=2,
            noiseTemperature=290,
            min_rate=10e3
        )

    def makeFillBlockProcesses(self, GTs):
        """
        Creates the processes for filling the data blocks and adding them to the send-buffer. A separate process for
        each destination gateway is created.
        """

        self.totalGTs = len(GTs)

        for gt in GTs:
            if gt != self:
                # add a process for each destination which runs the function 'fillBlock'
                self.fillBlocks.append(self.env.process(self.fillBlock(gt)))

    def fillBlock(self, destination):
        """
        Simpy process function:

        Creates a block headed for a given destination, finds the time for a block to be full and adds the block to the
        send-buffer after the calculated time.

        A separate process for each destination gateway will be running this function.
        """
        index = 0
        unavailableDestinationBuffer = []

        while True:
            try:
                # create a new block to be filled
                block = DataBlock(self, destination, str(self.ID) + "_" + str(destination.ID) + "_" + str(index), self.env.now)

                timeToFull = self.timeToFullBlock(block)  # calculate time to fill block

                yield self.env.timeout(timeToFull)  # wait until block is full

                if block.destination.linkedSat[0] is None:
                    unavailableDestinationBuffer.append(block)
                else:
                    while unavailableDestinationBuffer: # empty buffer before adding new block
                        if not self.sendBuffer[0][0].triggered:
                            self.sendBuffer[0][0].succeed()
                            self.sendBuffer[1].append(unavailableDestinationBuffer[0])
                            unavailableDestinationBuffer.pop(0)
                        else:
                            newEvent = self.env.event().succeed()
                            self.sendBuffer[0].append(newEvent)
                            self.sendBuffer[1].append(unavailableDestinationBuffer[0])
                            unavailableDestinationBuffer.pop(0)

                    block.path = self.paths[destination.name]

                    if self.earth.pathParam == 'Q-Learning' or self.earth.pathParam == 'Deep Q-Learning':
                        block.QPath = [block.path[0], block.path[1], block.path[len(block.path)-1]]
                        # We add a Qpath field for the Q-Learning case. Only source and destination will be added
                        # after that, every hop will be added at the second last position.

                    if not block.path:
                        print(self.name, destination.name)
                        exit()
                    block.timeAtFull = self.env.now
                    createdBlocks.append(block)
                    # add block to send-buffer
                    if not self.sendBuffer[0][0].triggered:
                        self.sendBuffer[0][0].succeed()
                        self.sendBuffer[1].append(block)
                    else:
                        newEvent = self.env.event().succeed()
                        self.sendBuffer[0].append(newEvent)
                        self.sendBuffer[1].append(block)
                    index += 1
            except simpy.Interrupt:
                print(f'Simpy interrupt at filling block at gateway{self.name}')
                break

    def sendBlock(self):
        """
        Simpy process function:

        Sends data blocks that are filled and added to the send-buffer which is a list of events and data blocks. The
        function monitors the send-buffer, and when the buffer contains one or more triggered events, the function will
        calculate the time it will take to send the block (yet to be implemented), and trigger an event which notifies
        a separate process that a block has been sent (yet to be implemented).

        After a block is sent, the function will send the next, if any more blocks are ready to be sent.

        (While it is assumed that if a buffer is full and ready to be sent it will always be at the first index,
        the method simpy.AnyOf is used. The end result is the same and this method is simple to implement.
        Furthermore, it allows for handling of such errors where a later index is ready but the first is not.
        this case is, however, not handled.)

        Since there is only one link on the GT for sending, there will only be one process running this method.
        """
        while True:
            yield self.sendBuffer[0][0]     # event 0 of block 0

            # wait until a satellite is linked
            while self.linkedSat[0] is None:
                yield self.env.timeout(0.1)

            # calculate propagation time and transmission time
            propTime = self.timeToSend(self.linkedSat)
            timeToSend = BLOCK_SIZE/self.dataRate

            self.sendBuffer[1][0].timeAtFirstTransmission = self.env.now
            yield self.env.timeout(timeToSend)
            # ANCHOR KPI: txLatency send block from GT
            self.sendBuffer[1][0].txLatency += timeToSend

            if not self.sendBuffer[1][0].path:
                print(self.sendBuffer[1][0].source.name, self.sendBuffer[1][0].destination.name)
                exit()

            self.linkedSat[1].createReceiveBlockProcess(self.sendBuffer[1][0], propTime)

            # remove from own sendBuffer
            if len(self.sendBuffer[0]) == 1:
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)
                self.sendBuffer[0].append(self.env.event())
            else:
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)

    def timeToSend(self, linkedSat):
        distance = linkedSat[0]
        pTime = distance/Vc
        return pTime

    def createReceiveBlockProcess(self, block, propTime):
        """
        Function which starts a receiveBlock process upon receiving a block from a transmitter.
        Adds the propagation time to the block attribute
        """

        process = self.env.process(self.receiveBlock(block, propTime))

    def receiveBlock(self, block, propTime):
        """
        Simpy process function:

        This function is used to handle the propagation delay of data blocks. This is done simply by waiting the time
        of the propagation delay. As a GT will always be the last step in a block's path, there is no need to send the
        block further. After the propagation delay, the block is simply added to a list of finished blocks so the KPIs
        can be tracked at the end of the simulation.

        While the transmission delay is handled at the transmitter, the transmitter cannot also wait for the propagation
        delay, otherwise the send-buffer might be overfilled.
        """
        # wait for block to fully propagate
        yield self.env.timeout(propTime)
        # ANCHOR KPI: propLatency send block from GT
        block.propLatency += propTime

        block.checkPoints.append(self.env.now)

        receivedDataBlocks.append(block)

    def cellDistance(self, cell) -> float:
        """
        Calculates the distance to the specified cell (assumed the center of the cell).
        Calculation is based on the geopy package which uses the 'WGS-84' model for earth shape.
        """
        cellCoord = (math.degrees(cell.latitude), math.degrees(cell.longitude))  # cell lat and long is saved in a format which is not degrees
        gTCoord = (self.latitude, self.longitude)

        return geopy.distance.geodesic(cellCoord,gTCoord).km

    def distance_GSL(self, satellite):
        """
        Distance between GT and satellite is calculated using the distance formula based on the cartesian coordinates
        in 3D space.
        """

        satCoords = [satellite.x, satellite.y, satellite.z]
        GTCoords = [self.x, self.y, self.z]

        distance = math.dist(satCoords, GTCoords)
        return distance

    def adjustDataRate(self):

        speff_thresholds = np.array(
            [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
             1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
             2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
             3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
             5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
        lin_thresholds = np.array(
            [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
             1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
             3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
             8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
             16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
             45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])
        db_thresholds = np.array(
            [-100.00000, -2.85000, -2.35000, -2.03000, -1.24000, -0.30000, 0.22000, 1.00000, 1.45000, 2.23000, 3.10000,
             4.03000, 4.68000, 4.73000, 5.13000, 5.50000, 5.97000, 6.55000, 6.84000, 7.41000, 8.10000, 8.38000, 8.43000,
             8.97000, 9.27000, 9.71000, 10.21000, 10.65000, 11.03000, 11.10000, 11.61000, 11.75000, 12.17000, 12.73000,
             13.05000, 13.64000, 13.98000, 14.81000, 15.47000, 15.87000, 16.55000, 16.98000, 17.24000, 18.10000,
             18.59000, 18.84000, 19.57000])

        pathLoss = 10*np.log10((4*math.pi*self.linkedSat[0]*self.gs2ngeo.f/Vc)**2)
        snr = 10**((self.gs2ngeo.maxPtx_db + self.gs2ngeo.G - pathLoss - self.gs2ngeo.No)/10)
        shannonRate = self.gs2ngeo.B*np.log2(1+snr)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.gs2ngeo.B*feasible_speffs[-1]

        self.dataRate = speff

    def orderSatsByDist(self, constellation):
        """
        Calculates the distance from the GT to all satellites and saves a sorted (least to greatest distance) list of
        all the satellites that are within range of the GT.
        """
        sats = []
        index = 0
        for orbitalPlane in constellation:
            for sat in orbitalPlane.sats:
                d_GSL = self.distance_GSL(sat)
                # ensure that the satellite is within range
                if d_GSL <= sat.maxSlantRange*10: #FIXME this x10 is for small constellations
                    sats.append((d_GSL, sat, [index]))
                index += 1
        sats.sort()
        self.satsOrdered = sats

    def addRefOnSat(self):
        """
        Adds a reference of the GT on a satellite based on the local list of satellites that are within range of the GT.
        This function is used in the greedy version of the 'linkSats2GTs()' method in the Earth class.
        The function uses a local indexing number to choose which satellite to add a reference to. If the satellite
        already has a reference, the GT checks if it is closer than the existing reference. If it is closer, it
        overwrites the reference and forces the other GT to add a reference to the next satellite it its own list.
        """
        if self.satIndex >= len(self.satsOrdered):
            self.linkedSat = (None, None)
            print("No satellite for GT {}".format(self.name))
            return

        # check if satellite has reference
        if self.satsOrdered[self.satIndex][1].linkedGT is None:
            # add self as reference on satellite
            self.satsOrdered[self.satIndex][1].linkedGT = self
            self.satsOrdered[self.satIndex][1].GTDist = self.satsOrdered[self.satIndex][0]

        # check if satellites reference is further away than this GT
        elif self.satsOrdered[self.satIndex][1].GTDist < self.satsOrdered[self.satIndex][0]:
            # force other GT to increment satIndex and check next satellite in its local ordered list
            self.satsOrdered[self.satIndex][1].linkedGT.satIndex += 1
            self.satsOrdered[self.satIndex][1].linkedGT.addRefOnSat()

            # add self as reference on satellite
            self.satsOrdered[self.satIndex][1].linkedGT = self
            self.satsOrdered[self.satIndex][1].GTDist = self.satsOrdered[self.satIndex][0]
        else:
            self.satIndex += 1
            if self.satIndex == len(self.satsOrdered):
                self.linkedSat = (None, None)
                print("No satellite for GT {}".format(self.name))
                return

            self.addRefOnSat()

    def link2Sat(self, dist, sat):
        """
        Links the GT to the satellite chosen in the 'linkSats2GTs()' method in the Earth class and makes sure that the
        data rate for the RFlink to the satellite is updated.
        """
        self.linkedSat = (dist, sat)
        sat.linkedGT = self
        sat.GTDist = dist
        self.adjustDataRate()

    def addCell(self, cellInfo):
        """
        Links a cell to the GT by adding the relevant information of the cell to the local list "cellsInRange".
        """
        self.cellsInRange.append(cellInfo)

    def removeCell(self, cell):
        """
        Unused function
        """
        for i, cellInfo in enumerate(self.cellsInRange):
            if cell.latitude == cellInfo[0][0] and cell.longitude == cellInfo[0][1]:
                cellInfo.pop(i)
                return True
        return False

    def findCellsWithinRange(self, earth, maxDistance):
        """
        This function finds the cells that are within the coverage area of the gateway instance. The cells are
        found by checking cells one at a time from the location of the gateway moving outward in a circle until
        the edge of the circle around the terminal exclusively consists of cells that border cells which are outside the
        coverage area. This is an optimized way of finding the cells within the coverage area, as only a limited number
        of cells outside the coverage is checked.

        The size of the area that is checked for is based on the parameter 'maxDistance' which can be seen as the radius
        of the coverage area in kilometers.

        The function will not "link" the cells and the gateway. Instead, it will only add a reference in the
        cells to the closest GT. As a result, all GTs must run this function before any linking is performed. The
        linking is done in the function: "linkCells2GTs()", in the Earth class, which also runs this function. This is
        done to handle cases where the coverage areas of two or more GTs are overlapping and the cells must only link to
        one of the GTs.

        The information added to the "cellsWithinRange" list is used for generating flows from the cells to each GT.
        """

        # Up right:
        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == earth.total_x: # "roll over" to opposite side of grid.
                x = 0
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:  # "roll over" to opposite side of grid.
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y -= 1  # the y-axis is flipped in the cell grid.
            x += 1

        # Down right:
        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == earth.total_x:  # "roll over" to opposite side of grid.
                x = 0
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == earth.total_y:  # "roll over" to opposite side of grid.
                    y = 0
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y += 1  # the y-axis is flipped in the cell grid.
            x += 1

        # up left:
        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == -1:  # "roll over" to opposite side of grid.
                x = earth.total_x - 1
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:  # "roll over" to opposite side of grid.
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y -= 1  # the y-axis is flipped in the cell grid.
            x -= 1

        # down left:
        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == -1:  # "roll over" to opposite side of grid.
                x = earth
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:  # "roll over" to opposite side of grid.
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y += 1  # the y-axis is flipped in the cell grid.
            x -= 1

    def timeToFullBlock(self, block):
        """
        Calculates the average time it will take to fill up a data block and returns the actual time based on a
        random variable following an exponential distribution.
        Different from the non reinforcement version of the simulator, this does not include different methods for
        setting the fractions of the data generation to each destination gateway.
        """

        # split the traffic evenly among the active gateways while keeping the fraction to each gateway the same
        # regardless of number of active gateways
        flow = self.totalAvgFlow / (len(self.totalLocations) - 1)

        avgTime = block.size / flow  # the average time to fill the buffer in seconds

        time = np.random.exponential(scale=avgTime) # the actual time to fill the buffer after adjustment by exp dist.

        return time

    def getTotalFlow(self, avgFlowPerUser, distanceFunc, maxDistance, capacity = None, fraction = 1.0):
        """
        This function is used as a precursor for the 'timeToFillBlock' method. Based on one of two distance functions
        this function finds the combined average flow from the combined users within the ground coverage area of the GT.

        Calculates the average combined flow from all cells scaling with distance in one of two ways:
            For the step function this means that it essentially just counts the number of users from the local list and
            multiplies with the flowPerUser value.

            For the slope it means that the slope is found using the flowPerUser and maxDistance as the gradient where
            the function gives 0 at the maximum distance.

            If this logic should be changed, it is important that it is done so in accordance with the
            "findCellsWithinRange" method.
        """
        if balancedFlow:
            self.totalAvgFlow = totalFlow
        else:
            totalAvgFlow = 0
            avgFlowPerUser =  avUserLoad

            if distanceFunc == "Step":
                for cell in self.cellsInRange:
                    totalAvgFlow += cell[1] * avgFlowPerUser

            elif distanceFunc == "Slope":
                gradient = (0-avgFlowPerUser)/(maxDistance-0)
                for cell in self.cellsInRange:
                    totalAvgFlow += (gradient * cell[2] + avgFlowPerUser) * cell[1]

            else:
                print("Error, distance function not recognized. Provided function = {}. Allowed functions: {} or {}".format(
                    distanceFunc,
                    "Step",
                    "slope"))
                exit()

            if self.linkedSat[0] is None:
                self.dataRate = self.gs2ngeo.min_rate

            if not capacity:
                capacity = self.dataRate

            if totalAvgFlow < capacity * fraction:
                self.totalAvgFlow = totalAvgFlow
            else:
                self.totalAvgFlow = capacity * fraction
                
        print(self.name + ': ' + str(self.totalAvgFlow/1000000000))

    def __eq__(self, other):
        if self.latitude == other.latitude and self.longitude == other.longitude:
            return True
        else:
            return False

    def __repr__(self):
        return 'Location = {}\n Longitude = {}\n Latitude = {}\n pos x= {}, pos y= {}, pos z= {}'.format(
            self.name,
            self.longitude,
            self.latitude,
            self.x,
            self.y,
            self.z)


# A single cell on earth
class Cell:
    def __init__(self, total_x, total_y, cell_x, cell_y, users, Re=6378e3, f=20e9, bw=200e6, noise_power=1 / (1e11)):
        # X and Y coordinates of the cell on the dataset map
        self.map_x = cell_x
        self.map_y = cell_y
        # Latitude and longitude of the cell as per dataset map
        self.latitude = math.pi * (0.5 - cell_y / total_y)
        self.longitude = (cell_x / total_x - 0.5) * 2 * math.pi
        if self.latitude < -5 or self.longitude < -5:
            print("less than 0")
            print(self.longitude, self.latitude)
            print(cell_x, cell_y)
            # exit()
        # Actual area the cell covers on earth (scaled for)
        self.area = 4 * math.pi * Re * Re * math.cos(self.latitude) / (total_x * total_y)
        # X,Y,Z coordinates to the center of the cell (assumed)
        self.x = Re * math.cos(self.latitude) * math.cos(self.longitude)
        self.y = Re * math.cos(self.latitude) * math.sin(self.longitude)
        self.z = Re * math.sin(self.latitude)

        self.users = users  # Population in the cell
        self.f = f  # Frequency used by the cell
        self.bw = bw  # Bandwidth used for the cell
        self.noise_power = noise_power  # Noise power for the cell
        self.rejected = True  # Usefulfor applications process to show if the cell is rejected or accepted
        self.gateway = None  # (groundstation, distance)

    def __repr__(self):
        return 'Users = {}\n area = {} km^2\n longitude = {} deg\n latitude = {} deg\n pos x = {}\n pos y = {}\n pos ' \
               'z = {}\n x position on map = {}\n y position on map = {}'.format(
                self.users,
                '%.2f' % (self.area / 1e6),
                '%.2f' % math.degrees(self.longitude),
                '%.2f' % math.degrees(self.latitude),
                '%.2f' % self.x,
                '%.2f' % self.y,
                '%.2f' % self.z,
                '%.2f' % self.map_x,
                '%.2f' % self.map_y)

    def setGT(self, gateways, maxDistance = 60):
        """
        Finds the closest gateway and updates the internal attribute 'self.gateway' as a tuple:
        (Gateway, distance to terminal). If the distance to the closest gateway is less than some maximum
        distance, the cell information is added to the gateway.
        """
        closestGT = (gateways[0], gateways[0].cellDistance(self))
        for gateway in gateways[1:]:
            distanceToGT = gateway.cellDistance(self)
            if distanceToGT < closestGT[1]:
                closestGT = (gateway, distanceToGT)
        self.gateway = closestGT

        if closestGT[1] <= maxDistance:
            closestGT[0].addCell([(math.degrees(self.latitude), math.degrees(self.longitude)), self.users, closestGT[1]])
        else:
            self.users = 0
        return closestGT


# Earth consisting of cells
# @profile
class Earth:
    def __init__(self, env, img_path, gt_path, constellation, inputParams, deltaT, totalLocations, getRates = False, window=None, outputPath='/'):
        # Input the population count data
        # img_path = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'
        self.outputPath = outputPath
        self.plotPaths = plotPath
        self.lostBlocks = 0
        self.queues = []
        self.loss   = []
        self.lossAv = []
        self.DDQNA  = None
        self.step   = 0
        self.nMovs  = 0     # number of total movements done by the constellation
        self.epsilon= []    # set of epsilon values
        self.rewards= []    # set of rewards
        self.trains = []    # Set of times when a fit to any dnn has happened
        self.graph  = None
        self.CKA    = []

        pop_count_data = Image.open(img_path)

        pop_count = np.array(pop_count_data)
        pop_count[pop_count < 0] = 0  # ensure there are no negative values

        # total image sizes
        [self.total_x, self.total_y] = pop_count_data.size

        self.total_cells = self.total_x * self.total_y

        # List of all cells stored in a 2d array as per the order in dataset
        self.cells = []
        for i in range(self.total_x):
            self.cells.append([])
            for j in range(self.total_y):
                self.cells[i].append(Cell(self.total_x, self.total_y, i, j, pop_count[j][i]))

        # window is a list with the coordinate bounds of our window of interest
        # format for window = [western longitude, eastern longitude, southern latitude, northern latitude]
        if window is not None:  # if window provided
            # latitude, longitude bounds:
            self.lati = [window[2], window[3]]
            self.longi = [window[0], window[1]]
            # dataset pixel bounds:
            self.windowx = (
            (int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
            self.windowy = (
            (int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))
        else:  # set window size as entire world if no window provided
            self.lati = [-90, 90]
            self.longi = [-179, 180]
            self.windowx = (0, self.total_x)
            self.windowy = (0, self.total_y)

        # import gateways from .csv
        self.gateways = []

        gateways = pd.read_csv(gt_path)

        length = 0
        for i, location in enumerate(gateways['Location']):
            for name in inputParams['Locations']:
                if name in location.split(","):
                    length += 1

        if inputParams['Locations'][0] != 'All':
            for i, location in enumerate(gateways['Location']):
                for name in inputParams['Locations']:
                    if name in location.split(","):
                        lName = gateways['Location'][i]
                        gtLati = gateways['Latitude'][i]
                        gtLongi = gateways['Longitude'][i]
                        self.gateways.append(Gateway(lName, i, gtLati, gtLongi, self.total_x, self.total_y,
                                                                   length, env, totalLocations, self))
                        break
        else:
            for i in range(len(gateways['Latitude'])):
                name = gateways['Location'][i]
                gtLati = gateways['Latitude'][i]
                gtLongi = gateways['Longitude'][i]
                self.gateways.append(Gateway(name, i, gtLati, gtLongi, self.total_x, self.total_y,
                                                           len(gateways['Latitude']), env, totalLocations, self))

        self.pathParam = pathing

        # create data Blocks on all GTs.
        if not getRates:
            for gt in self.gateways:
                gt.makeFillBlockProcesses(self.gateways)

        # create constellation of satellites
        self.LEO = create_Constellation(constellation, env, self)

        if rotateFirst:
            print('Rotating constellation...')
            for constellation in self.LEO:
                constellation.rotate(ndeltas*deltaT)

        # Simpy process for handling moving the constellation and the satellites within the constellation
        self.moveConstellation = env.process(self.moveConstellation(env, deltaT, getRates))

    def set_window(self, window):  # function to change/set window for the earth
        """
        Unused function
        """
        self.lati = [window[2], window[3]]
        self.longi = [window[0], window[1]]
        self.windowx = ((int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
        self.windowy = ((int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))

    def linkCells2GTs(self, distance):
        """
        Finds the cells that are within the coverage areas of all GTs and links them ensuring that a cell only links to
        a single GT.
        """
        start = time.time()

        # Find cells that are within range of all GTs
        for i, gt in enumerate(self.gateways):
            print("Finding cells within coverage area of GT {} of {}".format(i+1, len(self.gateways)), end='\r')
            gt.findCellsWithinRange(self, distance)
        print('\r')
        print("Time taken to find cells that are within range of all GTs: {} seconds".format(time.time() - start))

        start = time.time()

        # Add reference for cells to the GT they are closest to
        for cells in self.cells:
            for cell in cells:
                if cell.gateway is not None:
                    cell.gateway[0].addCell([(math.degrees(cell.latitude),
                                                     math.degrees(cell.longitude)),
                                                    cell.users,
                                                    cell.gateway[1]])

        print("Time taken to add cell information to all GTs: {} seconds".format(time.time() - start))
        print()

    def linkSats2GTs(self, method):
        """
        Links GTs to satellites. One satellite is only allowed to link to one GT.
        """
        sats = []
        for orbit in self.LEO:
            for sat in orbit.sats:
                sat.linkedGT = None
                sat.GTDist = None
                sats.append(sat)

        if method == "Greedy":
            for GT in self.gateways:
                GT.orderSatsByDist(self.LEO)
                GT.addRefOnSat()

            for orbit in self.LEO:
                for sat in orbit.sats:
                    if sat.linkedGT is not None:
                        sat.linkedGT.link2Sat(sat.GTDist, sat)
        elif method == "Optimize":
            # make cost matrix
            SxGT = np.array([[99999 for _ in range(len(sats))] for _ in range(len(self.gateways))])
            for i, GT in enumerate(self.gateways):
                GT.orderSatsByDist(self.LEO)
                for val, entry in enumerate(GT.satsOrdered):
                    SxGT[i][entry[2][0]] = val

            # find assignment of GSL which minimizes the cost from the cost matrix
            rowInd, colInd = linear_sum_assignment(SxGT)

            # link satellites and GTs
            for i, GT in enumerate(self.gateways):
                if SxGT[rowInd[i]][colInd[i]] < len(GT.satsOrdered):
                    sat = GT.satsOrdered[SxGT[rowInd[i]][colInd[i]]]
                    GT.link2Sat(sat[0], sat[1])
                else:
                    GT.linkedSat = (None, None)
                    print("no satellite for GT {}".format(GT.name))

    def getCellUsers(self):
        """
        Used for plotting the population map.
        """
        temp = []
        for i, cellList in enumerate(self.cells):
            temp.append([])
            for cell in cellList:
                temp[i].append(cell.users)
        return temp

    def updateSatelliteProcessesSimpler(self, graph):
        """

        Function from the non-reinforcement implementation. However, due to the paths not existing between transmitter
        and destination gateways (they get created as the blocks travel through the constellation), this version does
        work with Q-Learning and Deep-Learning.

        Can be used for a simpler version of updating the processes on satellites. However, it does not take into
        account that some processes may be able to continue without being stopped. Stopping the processes may lose
        time of the transmission of a block.

        Function which ensures all processes on all satellites are updated after constellation movement. This is done in
        several steps:
            - All blocks waiting to be sent or currently being sent has their paths updated.
            - All processes are stopped and remade according to current links - all transmission progress is lost on
            blocks currently being transmitted.
            - All buffers are emptied and blocks are redistributed to new buffers according to the blocks' arrival time
            at the satellite.
        """

        # update ISL references in all satellites, adjust data rate to GTs and ensure send-processes are correct
        sats = []
        for plane in self.LEO:
            for sat1 in plane.sats:
                sats.append(sat1)
        for plane in self.LEO:
            for sat in plane.sats:

                # remake path for all blocks
                for buffer in sat.sendBufferSatsIntra:
                    for block in buffer[1]:
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.path = path
                for buffer in sat.sendBufferSatsInter:
                    for block in buffer[1]:
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.path = path
                for block in sat.sendBufferGT[1]:
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path to GT:")
                        print(block)
                        exit()
                    block.path = path
                for block in sat.tempBlocks:
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path from Temp:")
                        print(block)
                        exit()
                    block.path = path

                # find neighboring satellites
                neighbors = list(nx.neighbors(graph, sat.ID))
                itt = 0
                neighborSats = []
                for sat2 in sats:
                    if sat2.ID in neighbors:
                        dataRate = nx.path_weight(graph, [sat2.ID, sat.ID], "dataRateOG")
                        distance = nx.path_weight(graph, [sat2.ID, sat.ID], "slant_range")
                        neighborSats.append((distance, sat2, dataRate))
                        itt += 1
                        if itt == len(neighbors):
                            break

                sat.intraSats = []
                sat.interSats = []

                # add new satellites as references
                for neighbor in neighborSats:
                    if neighbor[1].in_plane == sat.in_plane:
                        sat.intraSats.append(neighbor)
                    else:
                        sat.interSats.append(neighbor)

                # stop all processes
                for process in sat.sendBlocksSatsInter:
                    process.interrupt()
                for process in sat.sendBlocksSatsIntra:
                    process.interrupt()
                for process in sat.sendBlocksGT:
                    process.interrupt()
                sat.sendBlocksSatsIntra = []
                sat.sendBlocksSatsInter = []
                sat.sendBlocksGT = []

                # add all blocks to list and reset queues
                blocksToDistribute = []
                for buffer in sat.sendBufferSatsIntra:
                    for block in buffer[1]:
                        blocksToDistribute.append((block.checkPoints[-1], block))
                sat.sendBufferSatsIntra = []
                for buffer in sat.sendBufferSatsInter:
                    for block in buffer[1]:
                        blocksToDistribute.append((block.checkPoints[-1], block))
                sat.sendBufferSatsInter = []
                for block in sat.sendBufferGT[1]:
                    blocksToDistribute.append((block.checkPoints[-1], block))
                sat.sendBufferGT = ([sat.env.event()], [])

                # remake all processes
                if sat.linkedGT is not None:
                    sat.adjustDownRate()
                    # make a process for the GSL from sat to GT
                    sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                for neighbor in sat.intraSats:
                    # make a send buffer for each ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                    sat.sendBufferSatsIntra.append(([sat.env.event()], [], neighbor[1].ID))

                    # make a process for each ISL
                    sat.sendBlocksSatsIntra.append(sat.env.process(sat.sendBlock(neighbor, True, True)))

                for neighbor in sat.interSats:
                    # make a send buffer for each ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                    sat.sendBufferSatsInter.append(([sat.env.event()], [], neighbor[1].ID))

                    # make a process for each ISL
                    sat.sendBlocksSatsInter.append(sat.env.process(sat.sendBlock(neighbor, True, False)))

                # sort blocks by arrival time at satellite
                blocksToDistribute.sort()
                # add blocks to the correct queues based on next step in their path
                # since the blocks list is sorted by arrival time, the order in the new queues is correct
                for block in blocksToDistribute:
                    # get this satellite's index in the blocks path
                    index = None
                    for i, step in enumerate(block[1].path):
                        if sat.ID == step[0]:
                            index = i

                    # check if next step in path is GT (last step in path)
                    if index == len(block[1].path) - 2:
                        # add block to GT send-buffer
                        if not sat.sendBufferGT[0][0].triggered:
                            sat.sendBufferGT[0][0].succeed()
                            sat.sendBufferGT[1].append(block[1])
                        else:
                            newEvent = sat.env.event().succeed()
                            sat.sendBufferGT[0].append(newEvent)
                            sat.sendBufferGT[1].append(block[1])
                    else:
                        # get ID of next sat and find if it is intra or inter
                        ID = None
                        isIntra = False
                        for neighborSat in sat.intraSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID
                                isIntra = True
                        for neighborSat in sat.interSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID

                        if ID is not None:
                            sendBuffer = None
                            # find send-buffer for the satellite
                            if isIntra:
                                for buffer in sat.sendBufferSatsIntra:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                            else:
                                for buffer in sat.sendBufferSatsInter:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer

                            # add block to buffer
                            if not sendBuffer[0][0].triggered:
                                sendBuffer[0][0].succeed()
                                sendBuffer[1].append(block[1])
                            else:
                                newEvent = sat.env.event().succeed()
                                sendBuffer[0].append(newEvent)
                                sendBuffer[1].append(block[1])
                        else:
                            print("buffer for next satellite in path could not be found")

    def updateSatelliteProcessesCorrect(self, graph):
        """

        Function from the non-reinforcement implementation. However, due to the paths not existing between transmitter
        and destination gateways (they get created as the blocks travel through the constellation), this version does
        work with Q-Learning and Deep-Learning.

        Function which ensures all processes on all satellites are updated after constellation movement. This is done in
        several steps:
            - All blocks waiting to be sent or currently being sent has their paths updated.
            - ISLs are updated with references to new inter-orbit satellites (intra-orbit links do not change).
                - This includes updating buffer if ISL is changed
                - It also includes remaking send-process if ISL is changed
                - Despite intra-orbit links not changing, blocks in an intra-orbit buffer may have to be moved.
            - GSL is updated:
                - Depending on new status - whether the satellite has a GSL or not - and past status - whether the
                satellite had a GSL or not - GSL buffer and process is handled accordingly.
            - All blocks not currently being transmitted to a satellite/GT, which is still present as a ISL or GSL, are
            redistributed to send-buffers according to their arrival time at the satellite.

        This function differentiates from the simple version by allowing continued operation of send-processes after
        constellation movement if the link is not broken.
        """
        sats = []
        for plane in self.LEO:
            for sat1 in plane.sats:
                sats.append(sat1)

        for plane in self.LEO:
            for sat in plane.sats:
                # remake path for all blocks
                for buffer in sat.sendBufferSatsIntra:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        if newPath == -1:
                            if len(buffer[0]) == 1:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                                buffer[0].append(sat.env.event())
                            else:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                            continue
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.isNewPath = True
                        block.oldPath = block.path
                        block.newPath = newPath
                        block.path = path
                        index += 1

                for buffer in sat.sendBufferSatsInter:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        if newPath == -1:
                            if len(buffer[0]) == 1:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                                buffer[0].append(sat.env.event())
                            else:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                            continue
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.isNewPath = True
                        block.oldPath = block.path
                        block.newPath = newPath
                        block.path = path
                        index += 1

                index = 0
                while index < len(sat.sendBufferGT[1]):
                    block = sat.sendBufferGT[1][index]
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                    if newPath == -1:
                        if len(sat.sendBufferGT[0]) == 1:
                            sat.sendBufferGT[0].pop(index)
                            sat.sendBufferGT[1].pop(index)
                            sat.sendBufferGT[0].append(sat.env.event())
                        else:
                            sat.sendBufferGT[0].pop(index)
                            sat.sendBufferGT[1].pop(index)
                        continue
                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        
                        ("no path to GT:")
                        print(block)
                        exit()
                    block.isNewPath = True
                    block.oldPath = block.path
                    block.newPath = newPath
                    block.path = path
                    index += 1

                index = 0
                while index < len(sat.tempBlocks):
                    block = sat.tempBlocks[index]
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)

                    if newPath == -1:
                        block.path = -1
                        if len(sat.tempBlocks[0]) == 1:
                            sat.tempBlocks[0].pop(index)
                            sat.tempBlocks[1].pop(index)
                            sat.tempBlocks[0].append(sat.env.event())
                        else:
                            sat.tempBlocks[0].pop(index)
                            sat.tempBlocks[1].pop(index)
                        continue

                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path from Temp:")
                        print(block)
                        exit()
                    block.isNewPath = True
                    block.oldPath = block.path
                    block.newPath = newPath
                    block.path = path
                    index += 1

                # find neighboring satellites
                neighbors = list(nx.neighbors(graph, sat.ID))
                itt = 0
                neighborSatsInter = []
                for sat2 in sats:
                    if sat2.ID in neighbors:
                        # we only care about the satellite if it is an inter-plane ISL
                        # we assume intra-plane ISLs will not change
                        if sat2.in_plane != sat.in_plane:
                            dataRate = nx.path_weight(graph, [sat2.ID, sat.ID], "dataRateOG")
                            distance = nx.path_weight(graph, [sat2.ID, sat.ID], "slant_range")
                            neighborSatsInter.append((distance, sat2, dataRate))
                        itt += 1
                        if itt == len(neighbors):
                            break
                sat.interSats = neighborSatsInter
                # list of blocks to be redistributed
                blocksToDistribute = []

                ### inter-plane ISLs ###

                sat.newBuffer = [True for _ in range(len(neighborSatsInter))]

                # make a list of False entries for each current neighbor
                sameSats = [False for _ in range(len(neighborSatsInter))]

                buffers = [None for _ in range(len(neighborSatsInter))]
                processes = [None for _ in range(len(neighborSatsInter))]

                # go through each process/buffer
                #   - check if the satellite is still there:
                #       - if it is, change the corresponding False to True, handle blocks and add process and buffer references to temporary list
                #       - if it is not, remove blocks from buffer and stop process
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsInter):
                    # check if the satellite is still there
                    isPresent = False
                    for neighborIndex, neighbor in enumerate(neighborSatsInter):
                        if buffer[2] == neighbor[1].ID:
                            isPresent = True
                            sameSats[neighborIndex] = True

                            ## handle blocks
                            # check if there are blocks in the buffer
                            if buffer[1]:
                                # find index of satellite in block's path
                                index = None
                                for i, step in enumerate(buffer[1][0].path):
                                    if sat.ID == step[0]:
                                        index = i
                                        break

                                # check if next step in path corresponds to buffer's satellite
                                if buffer[1][0].path[index + 1][0] == buffer[2]:
                                    # add all but the first block to redistribution list
                                    for block in buffer[1][1:]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))

                                    # add buffer with only first block present to temp list
                                    buffers[neighborIndex] = ([sat.env.event().succeed()], [sat.sendBufferSatsInter[bufferIndex][1][0]], buffer[2])
                                    processes[neighborIndex] = sat.sendBlocksSatsInter[bufferIndex]
                                else:
                                    # add all blocks to redistribution list
                                    for block in buffer[1]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))
                                    # reset buffer
                                    buffers[neighborIndex] = ([sat.env.event()], [], buffer[2])

                                    # reset process
                                    sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                    processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))

                            else: # there are no blocks in the buffer
                                # add buffer and remake process
                                buffers[neighborIndex] = sat.sendBufferSatsInter[bufferIndex]
                                sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))
                                # sendBlocksSatsInter[bufferIndex]

                            break
                    if not isPresent:
                        # add blocks to redistribution list
                        for block in buffer[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))
                        # stop process
                        sat.sendBlocksSatsInter[bufferIndex].interrupt()

                # make buffer and process for new neighbors(s)
                # - go through list of previously false entries:
                #   - check  entry for each neighbor:
                #       - if False, create buffer and process for new neighbor
                # - clear temporary list of processes and buffers
                for entryIndex, entry in enumerate(sameSats):
                    if not entry:
                        buffers[entryIndex] = ([sat.env.event()], [], neighborSatsInter[entryIndex][1].ID)
                        processes[entryIndex] = sat.env.process(sat.sendBlock(neighborSatsInter[entryIndex], True, False))

                # overwrite buffers and processes
                sat.sendBlocksSatsInter = processes
                sat.sendBufferSatsInter = buffers

                ### intra-plane ISLs ###
                # check blocks for each buffer
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsIntra):
                    ## handle blocks
                    # check if there are blocks in the buffer
                    if buffer[1]:
                        # find index of satellite in block's path
                        index = None
                        for i, step in enumerate(buffer[1][0].path):
                            if sat.ID == step[0]:
                                index = i
                                break

                        # check if next step in path corresponds to buffer's satellite
                        if buffer[1][0].path[index + 1][0] == buffer[2]:
                            # add all but the first block to redistribution list
                            for block in buffer[1][1:]:
                                blocksToDistribute.append((block.checkPoints[-1], block))

                            # remove all but the first block and event from the buffer
                            length = len(sat.sendBufferSatsIntra[bufferIndex][1]) - 1
                            for _ in range(length):
                                sat.sendBufferSatsIntra[bufferIndex][1].pop(1)
                                sat.sendBufferSatsIntra[bufferIndex][0].pop(1)

                        else:
                            # add all blocks to redistribution list
                            for block in buffer[1]:
                                blocksToDistribute.append((block.checkPoints[-1], block))
                            # reset buffer
                            sat.sendBufferSatsIntra[bufferIndex] = ([sat.env.event()], [], buffer[2])

                            # reset process
                            sat.sendBlocksSatsIntra[bufferIndex].interrupt()
                            sat.sendBlocksSatsIntra[bufferIndex] = sat.env.process(sat.sendBlock(sat.intraSats[bufferIndex], True, True))

                ### GSL ###
                # check if satellite has a linked GT
                if sat.linkedGT is not None:
                    sat.adjustDownRate()

                    # check if it had a sendBlocksGT process
                    if sat.sendBlocksGT:
                        # check if there are any blocks in the buffer
                        if sat.sendBufferGT[1]:
                            # check if linked GT is the same as the destination of first block in sendBufferGT
                            if sat.sendBufferGT[1][0].destination != sat.linkedGT:
                                sat.sendBlocksGT[0].interrupt()
                                sat.sendBlocksGT = []

                                # remove blocks from queue and add to list of blocks which should be redistributed
                                for block in sat.sendBufferGT[1]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                sat.sendBufferGT = ([sat.env.event()], [])

                                # make new send process for new linked GT
                                sat.sendBlocksGT.append(
                                    sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))
                            else:
                                # keep the first block in the buffer and let process continue
                                for block in sat.sendBufferGT[1][1:]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                length = len(sat.sendBufferGT[1]) - 1
                                for _ in range(length):
                                    sat.sendBufferGT[1].pop(1) # pop all but the first block
                                    sat.sendBufferGT[0].pop(1) # pop all but the first event

                        else:  # there are no blocks in the buffer
                            sat.sendBlocksGT[0].interrupt()
                            sat.sendBlocksGT = []
                            sat.sendBufferGT = ([sat.env.event()], [])
                            # make new send process for new linked GT
                            sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                    else:  # it had no process running
                        # there should be no blocks in the GT buffer, but just in case - if there are none, then the for loop will not run
                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                        # make new send process for new linked GT
                        sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                else:  # no linked GT
                    # check if there is a sendBlocksGT process
                    if sat.sendBlocksGT:
                        sat.sendBlocksGT[0].interrupt()
                        sat.sendBlocksGT = []

                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                # sort blocks by arrival time at satellite
                blocksToDistribute.sort()
                # add blocks to the correct queues based on next step in their path
                # since the blocks list is sorted by arrival time, the order in the new queues is correct
                for block in blocksToDistribute:
                    # get this satellite's index in the blocks path
                    index = None
                    for i, step in enumerate(block[1].path):
                        if sat.ID == step[0]:
                            index = i

                    # check if next step in path is GT (last step in path)
                    if index == len(block[1].path) - 2:
                        # add block to GT send-buffer
                        if not sat.sendBufferGT[0][0].triggered:
                            sat.sendBufferGT[0][0].succeed()
                            sat.sendBufferGT[1].append(block[1])
                        else:
                            newEvent = sat.env.event().succeed()
                            sat.sendBufferGT[0].append(newEvent)
                            sat.sendBufferGT[1].append(block[1])
                    else:
                        # get ID of next sat and find if it is intra or inter
                        ID = None
                        isIntra = False
                        for neighborSat in sat.intraSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID
                                isIntra = True
                        for neighborSat in sat.interSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID

                        if ID is not None:
                            sendBuffer = None
                            # find send-buffer for the satellite
                            if isIntra:
                                for buffer in sat.sendBufferSatsIntra:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                            else:
                                for buffer in sat.sendBufferSatsInter:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer

                            # add block to buffer
                            if not sendBuffer[0][0].triggered:
                                sendBuffer[0][0].succeed()
                                sendBuffer[1].append(block[1])
                            else:
                                newEvent = sat.env.event().succeed()
                                sendBuffer[0].append(newEvent)
                                sendBuffer[1].append(block[1])
                        else:
                            print("buffer for next satellite in path could not be found")

    def updateSatelliteProcessesRL(self, graph):
        """
        Update: This function works now. The issue is that all the inter-plane packets that were in a queue to be sent are discarded
        when the graph is updated and those links stop existing.
        This function does not work correctly! The remaking of processes and queues fails when the satellites move
        enough so that new links must be formed.

        This function takes into account that the paths are not complete and the next step may not have been chosen yet.

        Function which ensures all processes on all satellites are updated after constellation movement. This is done in
        several steps:
            - All blocks waiting to be sent or currently being sent has their paths updated.
            - ISLs are updated with references to new inter-orbit satellites (intra-orbit links do not change).
                - This includes updating buffer if ISL is changed
                - It also includes remaking send-process if ISL is changed
                - Despite intra-orbit links not changing, blocks in an intra-orbit buffer may have to be moved.
            - GSL is updated:
                - Depending on new status - whether the satellite has a GSL or not - and past status - whether the
                satellite had a GSL or not - GSL buffer and process is handled accordingly.
            - All blocks not currently being transmitted to a satellite/GT, which is still present as a ISL or GSL, are
            redistributed to send-buffers according to their arrival time at the satellite.

        This function differentiates from the simple version by allowing continued operation of send-processes after
        constellation movement if the link is not broken.
        """
        # update linked sats
        sats = []
        for plane in self.LEO:
            for sat in plane.sats:
                sats.append(sat)
                if self.pathParam == 'Q-Learning':
                    # Update ISL
                    linkedSats   = getLinkedSats(sat, graph, self)
                    sat.QLearning.linkedSats =  {'U': linkedSats['U'],
                                    'D': linkedSats['D'],
                                    'R': linkedSats['R'],
                                    'L': linkedSats['L']}
                elif self.pathParam == 'Deep Q-Learning':
                    # update ISL. Intra-plane should not change
                    sat.findIntraNeighbours(self)
                    sat.findInterNeighbours(self)


        for plane in self.LEO:
            for sat in plane.sats:
                # get next step for all blocks
                # doing this here assumes that the constellation movement will have a limited effect on the links
                # and that the queue sizes will not change significantly.

                # intra satellite buffers
                for buffer in sat.sendBufferSatsIntra:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]
                        nextHop = None

                        if len(block.QPath) > 3:  # the block does not come from a gateway
                            if sat.QLearning is not None:   # Q-Learning
                                nextHop = sat.QLearning.makeAction(block, sat,
                                                                    sat.orbPlane.earth.gateways[0].graph,
                                                                    sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            elif sat.DDQNA is not None:     # Deep Q-Learning-Online phase
                                nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                                   sat.orbPlane.earth.gateways[0].graph,
                                                                                   sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            elif self.DDQNA is not None:    # Deep Q-Learning-Offline phase
                                # nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                nextHop = self.DDQNA.makeDeepAction(block, sat,
                                                                                   sat.orbPlane.earth.gateways[
                                                                                       0].graph,
                                                                                   sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            else:
                                print(f'No learning model for sat: {sat.ID}')
                        else:
                            if sat.QLearning is not None:   # Q-Learning
                                nextHop = sat.QLearning.makeAction(block, sat,
                                                                    sat.orbPlane.earth.gateways[0].graph,
                                                                    sat.orbPlane.earth)
                            elif sat.DDQNA is not None:     # Deep Q-Learning-Offline phase
                                nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                    sat.orbPlane.earth.gateways[
                                                                        0].graph,
                                                                    sat.orbPlane.earth)
                            elif self.DDQNA is not None:    # Deep Q-Learning-Offline phase
                                # nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                nextHop = self.DDQNA.makeDeepAction(block, sat,
                                                                                   sat.orbPlane.earth.gateways[
                                                                                       0].graph,
                                                                                   sat.orbPlane.earth)
                            else:
                                print(f'No learning model for sat: {sat.ID}')

                        if nextHop is None:
                            print(f'Something wrong with block: {block}')
                        
                        elif nextHop != 0:
                            block.QPath[-2] = nextHop
                            pathPlot = block.QPath.copy()
                            pathPlot.pop()
                        else:
                            pathPlot = block.QPath.copy()

                        # If plotPath plots an image for every action taken. Prints 1/10 of blocks. # ANCHOR plot action earth 1
                        #################################################################
                        if sat.orbPlane.earth.plotPaths:
                            if int(block.ID[len(block.ID) - 1]) == 0:
                                os.makedirs(sat.orbPlane.earth.outputPath + '/pictures/',
                                            exist_ok=True)  # create output path
                                outputPath = sat.orbPlane.earth.outputPath + '/pictures/' + block.ID + '_' + str(
                                    len(block.QPath)) + '_'
                                # plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath)
                                plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath, ID=block.ID, time = block.creationTime)
                        #################################################################

                        # path = block.QPath  # if there is Q-Learning the path will be repalced with the QPath
                        index += 1

                # inter satellite buffers
                for buffer in sat.sendBufferSatsInter:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]

                        if len(block.QPath) > 3:  # the block does not come from a gateway
                            if sat.QLearning:
                                nextHop = sat.QLearning.makeAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[0].graph,
                                                                   sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            elif sat.DDQNA is not None:
                                nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                                  sat.orbPlane.earth.gateways[
                                                                                      0].graph,
                                                                                  sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            else:
                                nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                                                                  sat.orbPlane.earth.gateways[
                                                                                      0].graph,
                                                                                  sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                        else:
                            if sat.QLearning:
                                nextHop = sat.QLearning.makeAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[0].graph,
                                                                   sat.orbPlane.earth)
                            elif sat.DDQNA is not None:
                                nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                                  sat.orbPlane.earth.gateways[
                                                                                      0].graph,
                                                                                  sat.orbPlane.earth)

                            else:
                                nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                                                                  sat.orbPlane.earth.gateways[
                                                                                      0].graph,
                                                                                  sat.orbPlane.earth)

                        if nextHop != 0:
                            block.QPath[-2] = nextHop
                            pathPlot = block.QPath.copy()
                            pathPlot.pop()
                        else:
                            pathPlot = block.QPath.copy()

                        # If plotPath plots an image for every action taken. Prints 1/10 of blocks. # ANCHOR plot action earth 2
                        #################################################################
                        if sat.orbPlane.earth.plotPaths:
                            if int(block.ID[len(block.ID) - 1]) == 0:
                                os.makedirs(sat.orbPlane.earth.outputPath + '/pictures/',
                                            exist_ok=True)  # create output path
                                outputPath = sat.orbPlane.earth.outputPath + '/pictures/' + block.ID + '_' + str(
                                    len(block.QPath)) + '_'
                                # plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath)
                                plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath, ID=block.ID, time = block.creationTime)
                        #################################################################

                        # path = block.QPath  # if there is Q-Learning the path will be repalced with the QPath
                        index += 1

                # down link buffers
                index = 0
                while index < len(sat.sendBufferGT[1]):
                    block = sat.sendBufferGT[1][index]

                    if len(block.QPath) > 3:  # the block does not come from a gateway
                        if sat.QLearning:
                            nextHop = sat.QLearning.makeAction(block, sat,
                                                               sat.orbPlane.earth.gateways[0].graph,
                                                               sat.orbPlane.earth, prevSat=(
                                    findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                        elif sat.DDQNA is not None:
                            nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                              sat.orbPlane.earth.gateways[
                                                                                  0].graph,
                                                                              sat.orbPlane.earth, prevSat=(
                                    findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                        else:
                            nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                                                              sat.orbPlane.earth.gateways[
                                                                                  0].graph,
                                                                              sat.orbPlane.earth, prevSat=(
                                    findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                    else:
                        if sat.QLearning:
                            nextHop = sat.QLearning.makeAction(block, sat,
                                                               sat.orbPlane.earth.gateways[0].graph,
                                                               sat.orbPlane.earth)
                        elif sat.DDQNA is not None:
                            nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                              sat.orbPlane.earth.gateways[
                                                                                  0].graph,
                                                                              sat.orbPlane.earth)
                        else:
                            nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                                                              sat.orbPlane.earth.gateways[
                                                                                  0].graph,
                                                                              sat.orbPlane.earth)

                    if nextHop != 0:
                        block.QPath[-2] = nextHop
                        pathPlot = block.QPath.copy()
                        pathPlot.pop()
                    else:
                        pathPlot = block.QPath.copy()

                    # If plotPath plots an image for every action taken. Prints 1/10 of blocks. # ANCHOR plot action earth 3
                    #################################################################
                    if sat.orbPlane.earth.plotPaths:
                        if int(block.ID[len(block.ID) - 1]) == 0:
                            os.makedirs(sat.orbPlane.earth.outputPath + '/pictures/',
                                        exist_ok=True)  # create output path
                            outputPath = sat.orbPlane.earth.outputPath + '/pictures/' + block.ID + '_' + str(
                                len(block.QPath)) + '_'
                            # plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath)
                            plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath, ID=block.ID, time = block.creationTime)
                    #################################################################

                    # path = block.QPath  # if there is Q-Learning the path will be repalced with the QPath
                    index += 1

                # find neighboring satellites
                neighbors = list(nx.neighbors(graph, sat.ID))
                itt = 0
                neighborSatsInter = []
                for sat2 in sats:
                    if sat2.ID in neighbors:
                        # we only care about the satellite if it is an inter-plane ISL
                        # we assume intra-plane ISLs will not change
                        if sat2.in_plane != sat.in_plane:
                            dataRate = nx.path_weight(graph, [sat2.ID, sat.ID], "dataRateOG")
                            distance = nx.path_weight(graph, [sat2.ID, sat.ID], "slant_range")
                            neighborSatsInter.append((distance, sat2, dataRate))
                        itt += 1
                        if itt == len(neighbors):
                            break
                sat.interSats = neighborSatsInter
                # list of blocks to be redistributed
                blocksToDistribute = []

                ### inter-plane ISLs ###

                sat.newBuffer = [True for _ in range(len(neighborSatsInter))]

                # make a list of False entries for each current neighbor
                sameSats = [False for _ in range(len(neighborSatsInter))]

                buffers = [None for _ in range(len(neighborSatsInter))]
                processes = [None for _ in range(len(neighborSatsInter))]

                # go through each process/buffer
                #   - check if the satellite is still there:
                #       - if it is, change the corresponding False to True, handle blocks and add process and buffer references to temporary list
                #       - if it is not, remove blocks from buffer and stop process
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsInter):
                    # check if the satellite is still there
                    isPresent = False
                    for neighborIndex, neighbor in enumerate(neighborSatsInter):
                        if buffer[2] == neighbor[1].ID:
                            isPresent = True
                            sameSats[neighborIndex] = True

                            ## handle blocks
                            # check if there are blocks in the buffer
                            if buffer[1]:
                                # find index of satellite in block's path
                                index = None
                                for i, step in enumerate(buffer[1][0].QPath):
                                    if sat.ID == step[0]:
                                        index = i
                                        break

                                # check if next step in path corresponds to buffer's satellite
                                if buffer[1][0].QPath[index + 1][0] == buffer[2]:
                                    # add all but the first block to redistribution list
                                    for block in buffer[1][1:]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))

                                    # add buffer with only first block present to temp list
                                    buffers[neighborIndex] = ([sat.env.event().succeed()], [sat.sendBufferSatsInter[bufferIndex][1][0]], buffer[2])
                                    processes[neighborIndex] = sat.sendBlocksSatsInter[bufferIndex]
                                else:
                                    # add all blocks to redistribution list
                                    for block in buffer[1]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))
                                    # reset buffer
                                    buffers[neighborIndex] = ([sat.env.event()], [], buffer[2])

                                    # reset process
                                    sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                    processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))

                            else: # there are no blocks in the buffer
                                # add buffer and remake process
                                buffers[neighborIndex] = sat.sendBufferSatsInter[bufferIndex]
                                sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))
                                # sendBlocksSatsInter[bufferIndex]

                            break
                    if not isPresent:
                        # add blocks to redistribution list
                        for block in buffer[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))
                        # stop process
                        sat.sendBlocksSatsInter[bufferIndex].interrupt()

                # make buffer and process for new neighbors(s)
                # - go through list of previously false entries:
                #   - check  entry for each neighbor:
                #       - if False, create buffer and process for new neighbor
                # - clear temporary list of processes and buffers
                for entryIndex, entry in enumerate(sameSats):
                    if not entry:
                        buffers[entryIndex] = ([sat.env.event()], [], neighborSatsInter[entryIndex][1].ID)
                        processes[entryIndex] = sat.env.process(sat.sendBlock(neighborSatsInter[entryIndex], True, False))

                # overwrite buffers and processes
                sat.sendBlocksSatsInter = processes
                sat.sendBufferSatsInter = buffers

                ### intra-plane ISLs ###
                # check blocks for each buffer
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsIntra):
                    ## handle blocks
                    # check if there are blocks in the buffer
                    if buffer[1]:
                        # find index of satellite in block's path
                        index = None
                        for i, step in enumerate(buffer[1][0].QPath):
                            if sat.ID == step[0]:
                                index = i
                                break

                        # check if next step in path corresponds to buffer's satellite
                        if buffer[1][0].QPath[index + 1][0] == buffer[2]:
                            # add all but the first block to redistribution list
                            for block in buffer[1][1:]:
                                blocksToDistribute.append((block.checkPoints[-1], block))

                            # remove all but the first block and event from the buffer
                            length = len(sat.sendBufferSatsIntra[bufferIndex][1]) - 1
                            for _ in range(length):
                                sat.sendBufferSatsIntra[bufferIndex][1].pop(1)
                                sat.sendBufferSatsIntra[bufferIndex][0].pop(1)

                        else:
                            # add all blocks to redistribution list
                            for block in buffer[1]:
                                blocksToDistribute.append((block.checkPoints[-1], block))
                            # reset buffer
                            sat.sendBufferSatsIntra[bufferIndex] = ([sat.env.event()], [], buffer[2])

                            # reset process
                            sat.sendBlocksSatsIntra[bufferIndex].interrupt()
                            sat.sendBlocksSatsIntra[bufferIndex] = sat.env.process(sat.sendBlock(sat.intraSats[bufferIndex], True, True))

                ### GSL ###
                # check if satellite has a linked GT
                if sat.linkedGT is not None:
                    sat.adjustDownRate()

                    # check if it had a sendBlocksGT process
                    if sat.sendBlocksGT:
                        # check if there are any blocks in the buffer
                        if sat.sendBufferGT[1]:
                            # check if linked GT is the same as the destination of first block in sendBufferGT
                            if sat.sendBufferGT[1][0].destination != sat.linkedGT:
                                sat.sendBlocksGT[0].interrupt()
                                sat.sendBlocksGT = []

                                # remove blocks from queue and add to list of blocks which should be redistributed
                                for block in sat.sendBufferGT[1]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                sat.sendBufferGT = ([sat.env.event()], [])

                                # make new send process for new linked GT
                                sat.sendBlocksGT.append(
                                    sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))
                            else:
                                # keep the first block in the buffer and let process continue
                                for block in sat.sendBufferGT[1][1:]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                length = len(sat.sendBufferGT[1]) - 1
                                for _ in range(length):
                                    sat.sendBufferGT[1].pop(1) # pop all but the first block
                                    sat.sendBufferGT[0].pop(1) # pop all but the first event

                        else:  # there are no blocks in the buffer
                            sat.sendBlocksGT[0].interrupt()
                            sat.sendBlocksGT = []
                            sat.sendBufferGT = ([sat.env.event()], [])
                            # make new send process for new linked GT
                            sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                    else:  # it had no process running
                        # there should be no blocks in the GT buffer, but just in case - if there are none, then the for loop will not run
                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                        # make new send process for new linked GT
                        sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                else:  # no linked GT
                    # check if there is a sendBlocksGT process
                    if sat.sendBlocksGT:
                        sat.sendBlocksGT[0].interrupt()
                        sat.sendBlocksGT = []

                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                # sort blocks by arrival time at satellite
                try:
                    blocksToDistribute.sort()
                except Exception as e:
                    print(f"Caught an exception: {e}")
                    print(f'Something wrong with: \n{blocksToDistribute}')
                # add blocks to the correct queues based on next step in their path
                # since the blocks list is sorted by arrival time, the order in the new queues is correct
                for block in blocksToDistribute:
                    # get this satellite's index in the blocks path
                    index = None
                    for i, step in enumerate(block[1].QPath):
                        if sat.ID == step[0]:
                            index = i

                    # check if next step in path is GT (last step in path)
                    if index is None:
                        print(f'Satellite {sat.ID} not found in the QPath: {block[1].QPath}') # FIXME This should not happen. Debugging I realized when this happens the previous satellite is twice in last positions of QPath, instead of prevSat and currentSat. The current sat was the linked to the gateways bu after the movement it is not anymore.
                        self.lostBlocks += 1
                    elif index == len(block[1].QPath) - 2:
                        # add block to GT send-buffer
                        if not sat.sendBufferGT[0][0].triggered:
                            sat.sendBufferGT[0][0].succeed()
                            sat.sendBufferGT[1].append(block[1])
                        else:
                            newEvent = sat.env.event().succeed()
                            sat.sendBufferGT[0].append(newEvent)
                            sat.sendBufferGT[1].append(block[1])
                    else:
                        # get ID of next sat and find if it is intra or inter
                        ID = None
                        isIntra = False
                        for neighborSat in sat.intraSats:
                            id = neighborSat[1].ID
                            if id == block[1].QPath[index + 1][0]:
                                ID = neighborSat[1].ID
                                isIntra = True
                        for neighborSat in sat.interSats:
                            id = neighborSat[1].ID
                            if id == block[1].QPath[index + 1][0]:
                                ID = neighborSat[1].ID

                        if ID is not None:
                            sendBuffer = None
                            # find send-buffer for the satellite
                            if isIntra:
                                for buffer in sat.sendBufferSatsIntra:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                            else:
                                for buffer in sat.sendBufferSatsInter:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer

                            # add block to buffer
                            if not sendBuffer[0][0].triggered:
                                sendBuffer[0][0].succeed()
                                sendBuffer[1].append(block[1])
                            else:
                                newEvent = sat.env.event().succeed()
                                sendBuffer[0].append(newEvent)
                                sendBuffer[1].append(block[1])
                        else:
                            print("buffer for next satellite in path could not be found")

    def updateGTPaths(self):
        """
        Updates all paths for all GTs going to all other GTs and ensures that all blocks waiting to be sent has the
        correct path.
        """
        # make new paths for all GTs
        for GT in self.gateways:
            for destination in self.gateways:
                if GT != destination:
                    if destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                        path = getShortestPath(GT.name, destination.name, self.pathParam, GT.graph)
                        GT.paths.update({destination.name: path})


                    else:
                        GT.paths.update({destination.name: []})
                        print("no path from gateway!!")

            # update paths for all blocks in send-buffer
            for block in GT.sendBuffer[1]:
                block.path = GT.paths[block.destination.name]
                block.isNewPath = True
                block.QPath = [block.path[0], block.path[1], block.path[len(block.path) - 1]]
                # We add a Qpath field for the Q-Learning case. Only source and destination will be added
                # after that, every hop will be added at the second last position.

    def getGSLDataRates(self):
        upDataRates = []
        downDataRates = []
        for GT in self.gateways:
            if GT.linkedSat[0] is not None:
                upDataRates.append(GT.dataRate)

        for orbit in self.LEO:
            for satellite in orbit.sats:
                if satellite.linkedGT is not None:
                    downDataRates.append(satellite.downRate)

        return upDataRates, downDataRates

    def getISLDataRates(self):
        interDataRates = []
        highRates = 0
        for orbit in self.LEO:
            for satellite in orbit.sats:
                for satData in satellite.interSats:
                    if satData[2] > 3e9:
                        highRates += 1
                    interDataRates.append(satData[2])
        return interDataRates

    def moveConstellation(self, env, deltaT=3600, getRates = False):
        """
        Simpy process function:

        Moves the constellations in terms of the Earth's rotation and moves the satellites within the constellations.
        The movement is based on the time that has passed since last constellation movement and is defined by the
        "deltaT" variable.

        After the satellites have been moved a process of re-linking all links, both GSLs and ISLs, is conducted where
        the paths for all blocks are re-made, the blocks are moved (if necessary) to the correct buffers, and all
        processes managing the send-buffers are checked to ensure they will still work correctly.
        """

        # Get the data rate for a intra plane ISL - used for testing
        if getRates:
            intraRate.append(self.LEO[0].sats[0].intraSats[0][2])

        while True:
            print('Creating/Moving constellation: Updating satellites position and links.')
            if getRates:
                # get data rates for all inter plane ISLs and all GSLs (up and down) - used for testing
                upDataRates, downDataRates = self.getGSLDataRates()
                inter = self.getISLDataRates()

                for val in upDataRates:
                    upGSLRates.append(val)

                for val in downDataRates:
                    downGSLRates.append(val)

                for val in inter:
                    interRates.append(val)

            yield env.timeout(deltaT)

            # clear satellite references on all GTs
            for GT in self.gateways:
                GT.satsOrdered = []
                GT.linkedSat = (None, None)

            # rotate constellation and satellites
            for plane in self.LEO:
                plane.rotate(ndeltas*deltaT)

            # relink satellites and GTs
            self.linkSats2GTs("Optimize")

            # create new graph and add references to all GTs for every rotation
            # prevGraph = self.graph
            graph = createGraph(self, matching=matching)
            self.graph = graph
            for GT in self.gateways:
                GT.graph = graph

            if self.pathParam == 'Deep Q-Learning' or self.pathParam == 'Q-Learning':
                self.updateSatelliteProcessesRL(graph)
            else:
                self.updateSatelliteProcessesCorrect(graph)

            self.updateGTPaths()
            self.nMovs += 1
            if saveISLs:
                print('Constellation moved! Saving ISLs map...')
                islpath = outputPath + '/ISL_maps/'
                os.makedirs(islpath, exist_ok=True) 
                self.plotMap(plotGT = True, plotSat = True, edges=True, save = True, outputPath=islpath, n=self.nMovs)
                plt.close()

            # Perform Federated Learning
            if FL_Test:
                global const_moved
                const_moved = True
                CKA_before, CKA_after = perform_FL(self)#, outputPath)
                self.CKA.append([CKA_before, CKA_after, env.now])

    def testFlowConstraint1(self, graph):
        highestDist = (0,0)
        for GT in self.gateways:
            if 1/GT.linkedSat[0] > highestDist[0]:
                highestDist = (1/GT.linkedSat[0], GT)

        lowestDist = (1/highestDist[0], highestDist[1])

        toolargeDists = []

        for (u,v,c) in graph.edges.data("slant_range"):
            if c > lowestDist[0]:
                toolargeDists.append((u,v,c))

        print("number of edges with too large distance: {}".format(len(toolargeDists)))

    def testFlowConstraint2(self, graph):
        edgeWeights = nx.get_edge_attributes(graph, "slant_range")
        totalFailed = 0

        for GT in self.gateways[1:]:
            failed = False
            path = getShortestPath(self.gateways[0].name, GT.name, self.pathParam, graph)
            try:
                firstStep = GT.linkedSat[0]
            except KeyError:
                firstStep = edgeWeights[(path[1][0], path[0][0])]
                print(f'Keyerror in: {GT.name}')


            for index in range(1, len(path) - 2):
                try:
                    if edgeWeights[(path[index][0], path[index+1][0])] > firstStep:
                        failed = True
                except KeyError:
                    print(f'Keyerror 2 in: {GT.name}')
                    if edgeWeights[(path[index+1][0], path[index][0])] > firstStep:
                        failed = True
            if failed:
                print("{} could not create a path which adheres to flow constraints".format(GT.name))
                totalFailed += 1

        print("number of GT paths that cannot meet flow restraints: {}".format(totalFailed))

    def plotMap(self, plotGT = True, plotSat = True, path = None, bottleneck = None, save = False, ID=None, time=None, edges=False, arrow_gap=0.008, outputPath='', paths=None, fileName="map.png", n = None):
        if paths is None:
            plt.figure()
        else:
            plt.figure(figsize=(6, 3))

        legend_properties = {'size': 10, 'weight': 'bold'}
        markerscale = 1.5
        usage_threshold = 10   # In percentage

        # Compute the link usage
        def calculate_link_usage(paths):
            link_usage = {}
            for path in paths:
                for i in range(len(path) - 1):
                    start_node, end_node = path[i], path[i+1]
                    link_str = '{}_{}'.format(start_node[0], end_node[0])

                    # Coordinates for plotting
                    coordinates = [(start_node[1], start_node[2]), (end_node[1], end_node[2])]

                    if link_str in link_usage:
                        link_usage[link_str]['count'] += 1
                    else:
                        link_usage[link_str] = {'count': 1, 'coordinates': coordinates}
            return link_usage

        # Function to adjust arrow start and end points
        def adjust_arrow_points(start, end, gap_value):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist == 0:  # To avoid division by zero
                return start, end
            gap_scaled = gap_value * 1440  # Adjusting arrow_gap to coordinate system
            new_start = (start[0] + gap_scaled * dx / dist, start[1] + gap_scaled * dy / dist)
            new_end = (end[0] - gap_scaled * dx / dist, end[1] - gap_scaled * dy / dist)
            return new_start, new_end

        # Code for plotting edges with arrow gap
        if edges:
            if n is not None:
                fileName = outputPath + f"ISLs_map_{n}.png"
            else:
                fileName = outputPath + "ISLs_map.png"
            for plane in self.LEO:
                for sat in plane.sats:
                    orig_start_x = int((0.5 + math.degrees(sat.longitude) / 360) * 1440)
                    orig_start_y = int((0.5 - math.degrees(sat.latitude) / 180) * 720)

                    for connected_sat in sat.intraSats + sat.interSats:
                        orig_end_x = int((0.5 + math.degrees(connected_sat[1].longitude) / 360) * 1440)
                        orig_end_y = int((0.5 - math.degrees(connected_sat[1].latitude) / 180) * 720)

                        # Adjust arrow start and end points
                        adj_start, adj_end = adjust_arrow_points((orig_start_x, orig_start_y), (orig_end_x, orig_end_y), arrow_gap)

                        plt.arrow(adj_start[0], adj_start[1], adj_end[0] - adj_start[0], adj_end[1] - adj_start[1], 
                                shape='full', lw=0.5, length_includes_head=True, head_width=5)

            # Plot edges between gateways and satellites
            for GT in self.gateways:
                    if GT.linkedSat[1]:  # Check if there's a linked satellite
                        gt_x = GT.gridLocationX  # Use gridLocationX for gateway X coordinate
                        gt_y = GT.gridLocationY  # Use gridLocationY for gateway Y coordinate
                        sat_x = int((0.5 + math.degrees(GT.linkedSat[1].longitude) / 360) * 1440)  # Satellite longitude
                        sat_y = int((0.5 - math.degrees(GT.linkedSat[1].latitude) / 180) * 720)    # Satellite latitude

                        # Adjust only the endpoint for the arrow
                        _, adj_end = adjust_arrow_points((gt_x, gt_y), (sat_x, sat_y), arrow_gap)
                        
                        plt.arrow(gt_x, gt_y, adj_end[0] - gt_x, adj_end[1] - gt_y,
                                shape='full', lw=0.5, length_includes_head=True, head_width=5)
                        
        if plotSat:
            colors = cm.rainbow(np.linspace(0, 1, len(self.LEO)))

            for plane, c in zip(self.LEO, colors):
                for sat in plane.sats:
                    gridSatX = int((0.5 + math.degrees(sat.longitude) / 360) * 1440)
                    gridSatY = int((0.5 - math.degrees(sat.latitude) / 180) * 720) #GT.totalY)
                    scat2 = plt.scatter(gridSatX, gridSatY, marker='o', s=18, linewidth=0.5, edgecolors='black', color=c, label=sat.ID)
                    if plotSatID:
                        plt.text(gridSatX + 10, gridSatY - 10, sat.ID, fontsize=6, ha='left', va='center')    # ANCHOR plots the text of the ID of the satellites

        if plotGT:
            for GT in self.gateways:
                scat1 = plt.scatter(GT.gridLocationX, GT.gridLocationY, marker='x', c='r', s=28, linewidth=1.5, label = GT.name)

        # Print path if given
        if path:
            if bottleneck:
                xValues = [[], [], []]
                yValues = [[], [], []]
                minimum = np.amin(bottleneck[1])
                length = len(path)
                index = 0
                arr = 0
                minFound = False

                while index < length:
                    xValues[arr].append(int((0.5 + path[index][1] / 360) * 1440))  # longitude
                    yValues[arr].append(int((0.5 - path[index][2] / 180) * 720))  # latitude
                    if not minFound:
                        if bottleneck[1][index] == minimum:
                            arr+=1
                            xValues[arr].append(int((0.5 + path[index][1] / 360) * 1440))  # longitude
                            yValues[arr].append(int((0.5 - path[index][2] / 180) * 720))  # latitude
                            xValues[arr].append(int((0.5 + path[index+1][1] / 360) * 1440))  # longitude
                            yValues[arr].append(int((0.5 - path[index+1][2] / 180) * 720))  # latitude
                            arr+=1
                            minFound = True
                    index += 1

                scat3 = plt.plot(xValues[0], yValues[0], 'b')
                scat3 = plt.plot(xValues[1], yValues[1], 'r')
                scat3 = plt.plot(xValues[2], yValues[2], 'b')
            else:
                xValues = []
                yValues = []
                for hop in path:
                    xValues.append(int((0.5 + hop[1] / 360) * 1440))     # longitude
                    yValues.append(int((0.5 - hop[2] / 180) * 720))      # latitude
                scat3 = plt.plot(xValues, yValues)  # , marker='.', c='b', linewidth=0.5, label = hop[0])

        # Plot the map with the usage of all the links
        if paths is not None:
            link_usage = calculate_link_usage([block.QPath for block in paths]) if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning' else calculate_link_usage([block.path for block in paths])

            # After calculating max_usage in the plotting section
            try:
                max_usage = max(info['count'] for info in link_usage.values())
                min_usage = max_usage * 0.1  # Set minimum usage to 10% of the maximum
            except ValueError:
                print("Error: No data available for plotting congestion map.")
                print('Link usage values:\n')
                print(link_usage.values())  # FIXME why does this break when few values?
                return  -1 # If this is within a function, it will exit. If not, remove or adjust this line.

            # Find the most used link
            most_used_link = max(link_usage.items(), key=lambda x: x[1]['count'])
            print(f"Most used link: {most_used_link[0]}, Packets: {most_used_link[1]['count']}")

            norm = Normalize(vmin=usage_threshold, vmax=100)
            # cmap = cm.get_cmap('RdYlGn_r')  # Use a red-yellow-green reversed colormap
            # cmap = cm.get_cmap('inferno_r')  # Use a darker colormap
            cmap = cm.get_cmap('cool')  # Use a darker colormap

            for link_str, info in link_usage.items():
                usage = info['count']
                # Convert usage to a percentage of the maximum, with a floor of usage_threshold%
                usage_percentage = max(usage_threshold, (usage / max_usage) * 100)  # Ensure minimum of usage_threshold%
                # Adjust width based on usage_percentage instead of raw usage
                width = 0.5 + (usage_percentage / 100) * 2  # Use usage_percentage for scaling
                
                # Use usage_percentage for color scaling
                color = cmap(norm(usage_percentage))  # This line should use `usage_percentage` for color scaling

                coordinates = info['coordinates']

                # Get original start and end points for adjusting
                orig_start_x, orig_start_y = (0.5 + coordinates[0][0] / 360) * 1440, (0.5 - coordinates[0][1] / 180) * 720
                orig_end_x, orig_end_y = (0.5 + coordinates[1][0] / 360) * 1440, (0.5 - coordinates[1][1] / 180) * 720

                # Adjust start and end points using adjust_arrow_points
                (start_x, start_y), (end_x, end_y) = adjust_arrow_points((orig_start_x, orig_start_y), (orig_end_x, orig_end_y), arrow_gap)

                # Calculate control points for a slight curve, adjusted for the new start and end points
                mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
                ctrl_x, ctrl_y = mid_x + (end_y - start_y) / 10, mid_y - (end_x - start_x) / 5  # Adjust divisor for curve tightness

                # Create a Bezier curve for the directed link with adjusted start and end points
                verts = [(start_x, start_y), (ctrl_x, ctrl_y), (end_x, end_y)]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)

                # Ensure this color variable is used for the FancyArrowPatch
                patch = FancyArrowPatch(path=path, arrowstyle='-|>', color=color, linewidth=width, mutation_scale=5, zorder=0.5)
                plt.gca().add_patch(patch)

            # Add legend for congestion color coding
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            ticks = [10] + list(np.linspace(10, 100, num=5))  # Ticks from 10% to 100%
            plt.colorbar(sm, orientation='vertical', label='Relative Traffic Load (%)', fraction=0.02, pad=0.04, ticks=[int(tick) for tick in ticks]) 
            # plt.colorbar(sm, orientation='vertical', fraction=0.02, pad=0.04, ticks=[int(tick) for tick in ticks]) 
            # plt.colorbar(sm, orientation='vertical', label='Number of packets', fraction=0.02, pad=0.04)

            plt.xticks([])
            plt.yticks([])
            # outPath = outputPath + "/CongestionMapFigures/"
            # fileName = outPath + "/CongestionMap.png"
            # os.makedirs(outPath, exist_ok=True)


        if plotSat and plotGT:
            plt.legend([scat1, scat2], ['Gateways', 'Satellites'], loc=3, prop=legend_properties, markerscale=markerscale)
        elif plotSat:
            plt.legend([scat2], ['Satellites'], loc=3, prop=legend_properties, markerscale=markerscale)
        elif plotGT:
            plt.legend([scat1], ['Gateways'], loc=3, prop=legend_properties, markerscale=markerscale)

        plt.xticks([])
        plt.yticks([])

        if paths is None:
            cell_users = np.array(self.getCellUsers()).transpose()
            plt.imshow(cell_users, norm=LogNorm(), cmap='viridis')
        else:
            plt.gca().invert_yaxis()

        # plt.show()
        # plt.imshow(np.log10(np.array(self.getCellUsers()).transpose() + 1), )

        # Add title
        if time is not None and ID is not None:
            plt.title(f"Creation time: {time*1000:.0f}ms, block ID: {ID}")

        if save:
            plt.tight_layout()
            plt.savefig(fileName, dpi=1000, bbox_inches='tight', pad_inches=0)   
  
    def initializeQTables(self, NGT, hyperparams, g):
        '''
        QTables initialization at each satellite
        '''
        print('----------------------------------')

        # path = './Results/Q-Learning/qTablesImport/qTablesExport/' + str(NGT) + 'GTs/'
        path = tablesPath

        if importQVals:
            print('Importing Q-Tables from: ' + path)
        else:
            print('Initializing Q-tables...')
        
        i = 0
        for plane in self.LEO:
            for sat in plane.sats:
                i += 1
                if importQVals:
                    with open(path + sat.ID + '.npy', 'rb') as f:
                        qTable = np.load(f)
                    sat.QLearning = QLearning(NGT, hyperparams, self, g, sat, qTable=qTable)
                else:
                    sat.QLearning = QLearning(NGT, hyperparams, self, g, sat)

        if importQVals:
            print(str(i) + ' Q-Tables imported!')
        else:
            print(str(i) + ' Q-Tables created!')
        print('----------------------------------')

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs = []
        ys = []
        zs = []
        xG = []
        yG = []
        zG = []
        for con in self.LEO:
            for sat in con.sats:
                xs.append(sat.x)
                ys.append(sat.y)
                zs.append(sat.z)
        ax.scatter(xs, ys, zs, marker='o')
        for GT in self.gateways:
            xG.append(GT.x)
            yG.append(GT.y)
            zG.append(GT.z)
        ax.scatter(xG, yG, zG, marker='^')
        plt.show()

    def __repr__(self):
        return 'total divisions in x = {}\n total divisions in y = {}\n total cells = {}\n window of operation ' \
               '(longitudes) = {}\n window of operation (latitudes) = {}'.format(
                self.total_x,
                self.total_y,
                self.total_cells,
                self.windowx,
                self.windowy)


class hyperparam:
    def __init__(self, pathing):
        '''
        Hyperparameters of the Q-Learning model
        '''
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.ArriveR    = ArriveReward
        self.w1         = w1
        self.w2         = w2
        self.w4         = w4
        self.again      = againPenalty
        self.unav       = unavPenalty
        self.pathing    = pathing
        self.tau        = tau
        self.updateF    = updateF
        self.batchSize  = batchSize
        self.bufferSize = bufferSize
        self.hardUpdate = hardUpdate==1
        self.importQ    = importQVals
        self.MAX_EPSILON= MAX_EPSILON
        self.MIN_EPSILON= MIN_EPSILON
        self.LAMBDA     = LAMBDA
        self.plotPath  = plotPath
        self.coordGran  = coordGran
        self.ddqn       = ddqn
        self.latBias    = latBias
        self.lonBias    = lonBias
        self.diff       = diff
        self.explore    = explore
        self.reducedState= reducedState
        self.online     = onlinePhase
 
    def __repr__(self):
        return 'Hyperparameters:\nalpha: {}\ngamma: {}\nepsilon: {}\nw1: {}\nw2: {}\n'.format(
        self.alpha,
        self.gamma,
        self.epsilon,
        self.w1,
        self.w2)


# @profile
class QLearning:
    def __init__(self, NGT, hyperparams, earth, g, sat, qTable = None):
        '''
        Create a 6D numpy array to hold the current Q-values for each state and action pair: Q(s, a)
        The array contains 5 dimensions with the shape of the environment, as well as a 6th "action" dimension.
        The "action" dimension consists of 4 layers that will allow us to keep track of the Q-values for each possible action in each state
        The value of each (state, action) pair is initialized ranomly.
        '''
        satUp, satDown, satRight, satLeft = 3, 3, 3, 3
        linkedSats   = getLinkedSats(sat, g, earth)
        self.linkedSats =  {'U': linkedSats['U'],
                            'D': linkedSats['D'],
                            'R': linkedSats['R'],
                            'L': linkedSats['L']}

        self.actions         = ('U', 'D', 'R', 'L')     # Up, Down, Left, Right
        self.Destinations    = NGT

        self.nStates    = satUp*satDown*satRight*satLeft*NGT
        self.nActions   = len(self.actions)
                
        if qTable is None:  # initialize it randomly if we are not going to import it
            self.qTable = np.random.rand(satUp, satDown, satRight, satLeft, NGT, self.nActions)  # first 5 fields are states while 6th field is the action. 4050 values with 10 GTs

        else:
            self.qTable = qTable

        self.alpha  = hyperparams.alpha
        self.gamma  = hyperparams.gamma
        # self.epsilon= hyperparams.epsilon
        self.epsilon= []
        self.maxEps = hyperparams.MAX_EPSILON
        self.minEps = hyperparams.MIN_EPSILON
        self.w1     = hyperparams.w1
        self.w2     = hyperparams.w2

        self.oldState  = (0,0,0,0,0)
        self.oldAction = 0

    def makeAction(self, block, sat, g, earth, prevSat=None):
        '''
        This function will:
        1. Check if the destination is the linked gateway. In that case it will just return 0 and the block will be sent there.
        2. Observation of the environment in order to determine state space and get the linked satellites.
        3. Chooses an action. Random one (Exploration) or the most valuable one (Exploitation). If the direction of that action has no linked satellite, the QValue will be -inf
        4. Receive reward/penalty
            Penalties: If the block visits again the same satellite. Reward = -1
                       Another one directly proportional to the length of the destination queue.
            Reward: So far, it will be higher if it gets physically closer to the satellite
        5. Updates Q-Table of the previous hop (Agent) with the following information:
            1. Reward      : Time waited at satB Queue && slant range reduction.
            2. maxNewQValue: Max Q Value of all possible actions at the new agent.
            3. Old state-action taken at satA in order to know where to update the Q-Table. 
            Everytime satB receives a dataBlock from satA satB will send the information required to update satA QTable.
        '''

        # There is no 'Done' state, it will simply continue until the time stops
        # simplemente se va a recibir una recompensa positiva si el satelite al que envias el paquete es el linkado al destino de este

        # 1. check if the destination is the linked gateway. The value of this action becomes 10. # ANCHOR plots route of delivered package Q-Learning
        if sat.linkedGT and block.destination.name == sat.linkedGT.name:
            prevSat.QLearning.qTable[block.oldState][block.oldAction] = ArriveReward
            earth.rewards.append([ArriveReward, sat.env.now])
            if plotDeliver:
                if int(block.ID[len(block.ID)-1]) == 0: # Draws 1/10 arrivals
                    os.makedirs(earth.outputPath + '/pictures/', exist_ok=True) # drawing delivered
                    outputPath = earth.outputPath + '/pictures/' + block.ID + '_' + str(len(block.QPath)) + '_'
                    plotShortestPath(earth, block.QPath, outputPath, ID=block.ID, time = block.creationTime)
            
            return 0

        # 2. Observation of the environment
        newState = tuple(getState(block, sat, g, earth))
       
        # 3. Choose an action (the direction of the next hop)
        # randomly
        if explore and random.uniform(0, 1)<self.alignEpsilon(earth, sat):
            action = self.actions[random.randrange(len(self.actions))]
            while(self.linkedSats[action] == None): 
                action = self.actions[random.randrange(len(self.actions))]  # if that direction has no linked satellite
        
        # highest value
        else:
            qValues = self.qTable[newState]
            action  = self.actions[np.argmax(qValues)]                      # Most valuable action (The one that will give more reward) 
            while self.linkedSats[action] == None:
                self.qTable[newState][self.actions.index(action)] = -np.inf # change qTable if that action is not available
                action = self.actions[np.argmax(qValues)]

        destination = self.linkedSats[action]    # Action is the keyword of the chosen linked satellite, linkedSats is a dictionary with each satellite associated to its corresponding keyword

        # ACT -> [it is done outside, the next hop is added at sat.receiveBlock method to block.QPath]
        nextHop = [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)]

        # 4. Receive reward/penalty for the previous action
        if prevSat is not None:
            hop = [sat.ID, math.degrees(sat.longitude), math.degrees(sat.latitude)]
            # if the next hop was already visited before the reward will be againPenalty
            if hop in block.QPath[:len(block.QPath)-2]:
                reward = againPenalty
            else:
                distanceReward = getDistanceReward(prevSat, sat, block.destination, self.w2)
                try:
                    queueReward    = getQueueReward   (block.queueTime[len(block.queueTime)-1], self.w1)
                except IndexError:
                    queueReward = 0 # FIXME
                reward = distanceReward + queueReward
            
            earth.rewards.append([reward, sat.env.now])

        # 5. Updates Q-Table 
        # Update QTable of previous Node (Agent, satellite) if it was not a gateway     
            nextMax     = np.max(self.qTable[newState]) # max value of next state given oldAction
            oldQValue   = prevSat.QLearning.qTable[block.oldState][block.oldAction]
            newQvalue   = (1-self.alpha) * oldQValue + self.alpha * (reward+self.gamma*nextMax) 
            prevSat.QLearning.qTable[block.oldState][block.oldAction] = newQvalue
            
        else:
            # prev node was a gateway, no need to compute the reward
            reward = 0

        # this will be saved always, except when the next hop is the destination, where the process will have already returned
        block.oldState  = newState
        block.oldAction = self.actions.index(action)

        earth.step += 1

        return nextHop

    def alignEpsilon(self, earth, sat):
        global      CurrentGTnumber
        epsilon     = self.minEps + (self.maxEps - self.minEps) * math.exp(-LAMBDA * earth.step/(decayRate*(CurrentGTnumber**2)))
        earth        .epsilon.append([epsilon, sat.env.now])
        return epsilon

    def __repr__(self):
            return '\n Nº of destinations = {}\n Action Space = {}\n Nº of states = {}\n qTable: {}'.format(
            self.Destinations,
            self.actions,
            self.nStates,
            self.qTable)


# @profile
class DDQNAgent:
    def __init__(self, NGT, hyperparams, earth, sat_ID = None):   
        self.actions        = ('U', 'D', 'R', 'L')
        if not reducedState:
            self.states         = ['UpLinked Up', 'UpLinked Down','UpLinked Right','UpLinked Left',                        # Up Link
                            'Up Latitude', 'Up Longitude',                                                             # Up positions
                            'DownLinked Up', 'DownLinked Down','DownLinked Right','DownLinked Left',                   # Down Link
                            'Down Latitude', 'Down Longitude',                                                         # Down positions
                            'RightLinked Up', 'RightLinked Down','RightLinked Right','RightLinked Left',               # Right Link
                            'Right Latitude', 'Right Longitude',                                                       # Right positions
                            'LeftLinked Up', 'LeftLinked Down','LeftLinked Right','LeftLinked Left',                   # Left Link
                            'Left Latitude', 'Left Longitude',                                                         # Left positions

                            'Actual latitude', 'Actual longitude',                                                     # Actual Position
                            'Destination latitude', 'Destination longitude']                                           # Destination Position
        elif reducedState:
            self.states         = ('Up Latitude', 'Up Longitude',               # Up Link
                            'Down Latitude', 'Down Longitude',                  # Down Link
                            'Right Latitude', 'Right Longitude',                # Right Link
                            'Left Latitude', 'Left Longitude',                  # Left Link
                            'Actual latitude', 'Actual longitude',              # Current pos
                            'Destination latitude', 'Destination longitude')    # Destination pos
        if diff_lastHop: 
            self.states.insert(0, 'Last Hop')

        self.actionSize     = len(self.actions)
        self.stateSize      = len(self.states)
        self.destinations   = NGT
        self.earth          = earth

        if sat_ID is None:
            print(f'State Space:\n {self.states}\nState size: {self.stateSize} states')
            print(f'Action Space:\n {self.actions}')

        self.alpha  = hyperparams.alpha
        self.gamma  = hyperparams.gamma
        self.epsilon= []
        self.maxEps = hyperparams.MAX_EPSILON
        self.minEps = hyperparams.MIN_EPSILON
        self.w1     = hyperparams.w1
        self.w2     = hyperparams.w2
        self.w4     = hyperparams.w4
        self.tau    = hyperparams.tau
        self.updateF= hyperparams.updateF
        self.batchS = hyperparams.batchSize
        self.bufferS= hyperparams.bufferSize
        self.hardUpd= hyperparams.hardUpdate
        self.importQ= hyperparams.importQ
        self.online = hyperparams.online

        self.step   = 0
        self.i      = 0

        self.replayBuffer  = []
        self.experienceReplay = ExperienceReplay(self.bufferS)
        # self.optimizer        = Adam(learning_rate=self.alpha, clipnorm=Clipnorm)
        self.loss_function    = losses.Huber()

        if not self.importQ:
            '''
            The compile method is used to configure the learning process of qNetwork and it sets the optimizer and loss function that the model will use to learn during training.
            It only is done in the q network because in the DDQN algorithm, we train the qNetwork with the data from the environment and update qTarget periodically.

            In DDQN the qNetwork is updated with the learning process defined by the loss and optimizer, but the qTarget network used for evaluation and stability purpose is
            a frozen version of qNetwork, which is updated periodically and not during the learning process.
            '''
            # The first model makes the predictions for Q-values which are used to make a action
            self.qNetwork = self.createModel()
            if sat_ID is None:
                print('----------------------------------')
                print(f"Q-NETWORK created:")
                print('----------------------------------')
                self.qNetwork.summary()
            else:
                print(f'Satellite {sat_ID} Q-Network initialized')
            if ddqn:
                self.qTarget  = self.createModel()
                if sat_ID is None:
                    print('----------------------------------')
                    print("DDQN enabled, TARGET NETWORK created:")
                    print('----------------------------------')
                    self.qTarget.summary()
                else:
                    print(f'Satellite {sat_ID} Q-Target initialized')
        else:
            # if import models, it will import a trained model
            try:
                global nnpath
                self.qNetwork = keras.models.load_model(nnpath)
                if sat_ID is None:
                    print('----------------------------------')
                    print(f"Q-Network imported!!!")
                    print('----------------------------------')
                    self.qNetwork.summary()
                else:
                    print(f'Satellite {sat_ID} Q-Network imported!')
                
                if ddqn:
                    global nnpathTarget
                    # self.qTarget = self.qNetwork
                    self.qTarget = keras.models.load_model(nnpathTarget)
                    if sat_ID is None:
                        print('----------------------------------')
                        # print("DDQN enabled, TARGET NETWORK copied from Q-NETWORK:")
                        print(f"Q-Target imported!!!")
                        print('----------------------------------')
                    else:
                        # print(f'Satellite {sat_ID} Q-Target copied from Q-Network')
                        print(f'Satellite {sat_ID} Q-Target imported!')

            except FileNotFoundError:
                print('----------------------------------')
                print(f"Wrong Neural Network path")
                print('----------------------------------')
        
    def getNextHop(self, newState, linkedSats, sat, block):
        '''
        Given a new observed state and the linkied satellites, it will return the next hop
        '''
        # randomly (Exploration)
        if explore and random.uniform(0, 1)<self.alignEpsilon(self.step, sat):
            actIndex = random.randrange(self.actionSize)
            action   = self.actions[actIndex]
            while(linkedSats[action] == None):   # if that direction has no linked satellite
                self.experienceReplay.store(newState, actIndex, unavPenalty, newState, False) # stores experience, repeats randomly
                self.earth.rewards.append([unavPenalty, sat.env.now])
                action = self.actions[random.randrange(len(self.actions))]

        # highest value (Exploitation)
        else:
            if noPingPong: # No PING PONG: if one of the neighbours is the connected satellite then choose that one
                actIndex = -1
                if sat.upper == block.destination.linkedSat[1]:
                    actIndex = 0
                elif sat.lower == block.destination.linkedSat[1]:
                    actIndex = 1
                elif sat.right == block.destination.linkedSat[1]:
                    actIndex = 2
                elif sat.left == block.destination.linkedSat[1]:
                    actIndex = 3

                if actIndex>-1:
                    action      = self.actions[actIndex]
                    destination = linkedSats[action]
                    return [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)], actIndex
                
                # # Mapping from state indices to direction decisions
                # decision_map = {
                #     (4, 5): 0,    # Up
                #     (10, 11): 1,  # Down
                #     (16, 17): 2,  # Right
                #     (22, 23): 3   # Left
                # }
                #     # Current satellite's destination position
                # dest_lat = newState[0, 26]
                # dest_lon = newState[0, 27]

                # # Iterate through the decision map and compare
                # for (lat_idx, lon_idx), actIndex in decision_map.items():
                #     if np.isclose(dest_lat, newState[0, lat_idx]) and np.isclose(dest_lon, newState[0, lon_idx]):
                #         action      = self.actions[actIndex]
                #         destination = linkedSats[action]
                #         return [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)], actIndex

            # Predict 
            qValues = self.qNetwork(newState).numpy()               # NOTE NN. Gets next hop. state structure in debugging
            actIndex = np.argmax(qValues)
            action   = self.actions[actIndex]
            while(linkedSats[action] == None):              # the chosen action has no linked satellite. NEGATIVE REWARD and store it, motherfucker.
            
            # while (linkedSats[action] is None or        # the chosen action has no linked satellite or the chosen satellite has been visited twice.
            # sum(linkedSats[action].ID == path[0] for path in block.QPath[:-1]) > 1):    

                self.experienceReplay.store(newState, actIndex, unavPenalty, newState, False) # from state to the same state, reward -1, not terminated
                self.earth.rewards.append([unavPenalty, sat.env.now])
                qValues[0][actIndex] = -np.inf              # it will not be chosen again (as the model has still not trained with that)
            
            #     if np.all(qValues == -np.inf):              # all the neighbors have been visited twice
            #         print(f'WARNING: All neighbors have been visited at least twice. A loop is going on in {sat.ID} with block: {block.ID}')
            #         while (linkedSats[action] is None): # if all options were either not available or visited twice, then choose randomly an action that is available
            #             np.random.randint(4)
            #             actIndex = np.argmax(qValues)               # find again for the highest value
            #             action   = self.actions[actIndex]  
            #         break
                actIndex = np.argmax(qValues)               # find again for the highest value
                action   = self.actions[actIndex]  

        destination = linkedSats[action]    # Action is the keyword of the chosen linked satellite, linkedSats is a dictionary with 
                                            # each satellite associated to its corresponding keyword
        
        # ACT -> [it is done outside, the next hop is added at sat.receiveBlock method to block.QPath]
        try:
            return [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)], actIndex
        except:
            return -1

    def makeDeepAction(self, block, sat, g, earth, prevSat=None):
        '''
        There is no 'Done' state, it will simply continue until the time stops.
        This function will:
        1. Observation of the environment in order to determine state space and get the linked satellites to the one making the action.
        2. Check if the destination is the linked gateway. 
            If the satellite sent the block to the satellite linked to the destination GW, it will receive a reward of 10.
            The previous satellite will match the destination of the block to the linked gateway of the next state (I hope and I guess)
            In that case it will just return 0 and the block will be sent there.
        3. Chooses an action.
            Random one (Exploration)
            The most valuable one (Exploitation).
            If the direction of that action has no linked satellite, that action will not be available.
       4. Receive reward/penalty
            Penalties: If the block visits again the same satellite. Reward = -1
                       If it tries to send the block to a direction where there is no linked satellite.
                       Another one directly proportional to the length of the destination queue.
            Reward: One proportional to the slant range reduction, meaning that it will be higher if it gets physically closer to the satellite.
                    Another one when it reaches the destination
        5. Store experience from the previous hop (Agent) with the following information:
            1. Reward      : Time waited at satB Queue && slant range reduction.
            2. maxNewQValue: Max Q Value of all possible actions at the new agent.
            3. Old state-action taken at satA in order to know where to update the NNs. 
            Everytime satB receives a dataBlock from satA satB will send the information required to update the NNs.        
            Unlike in regular Q-Learning, in this step we just have to store the experience into the experience replay buffer.
            It will be updated automatically taking a random batch from the buffer every n iterations.
            We will store the old state of the block, the action index taken there, the reward received and the new state it moved into.
        6. Update the qTarget every n iterations.
        '''
        # 1. Observe the state and search for the satellites linked to the one making the action
        linkedSats  = getDeepLinkedSats(sat, g, earth)
        if reducedState:
            newState    = getDeepStateReduced(block, sat, linkedSats)
        elif diff and not diff_lastHop:
            newState    = getDeepStateDiff(block, sat, linkedSats) # This is the one being used by default
        elif diff_lastHop:
            newState    = getDeepStateDiffLastHop(block, sat, linkedSats)
        else:
            newState    = getDeepState(block, sat, linkedSats)

        if newState is None: 
            earth.lostBlocks+=1
            return 0
        self.step   += 1

        # 2. Check if the destination is the linked gateway. The reward is ArriveReward here and goes to the previous satellite. # ANCHOR plot delivered deep NN
        if sat.linkedGT and (block.destination.ID == sat.linkedGT.ID):    # Compare IDs
            if distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward  = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
                # distanceReward  = getDistanceRewardV4(prevSat, sat, block.destination, self.w2, self.w4)
                queueReward     = getQueueReward   (block.queueTime[len(block.queueTime)-1], self.w1)
                reward          = distanceReward + queueReward + ArriveReward
                self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, True)
                self.earth.rewards.append([reward, sat.env.now])
                # self.experienceReplay.store(block.oldState, block.oldAction, ArriveReward, newState, True)
            elif distanceRew == 5:
                distanceReward  = getDistanceRewardV5(prevSat, sat, self.w2)
                reward          = distanceReward + ArriveReward
                self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, True)
                self.earth.rewards.append([reward, sat.env.now])
            else:
                self.experienceReplay.store(block.oldState, block.oldAction, ArriveReward, newState, True)
                self.earth.rewards.append([ArriveReward, sat.env.now])

            if TrainThis: self.train(sat, earth) # FIXME why here a train?? should not be here. Make a test without this when the model is stable
            if plotDeliver:
                if int(block.ID[len(block.ID)-1]) == 0: # Draws 1/10 arrivals
                    os.makedirs(earth.outputPath + '/pictures/', exist_ok=True) # drawing delivered
                    outputPath = earth.outputPath + '/pictures/' + block.ID + '_' + str(len(block.QPath)) + '_'
                    plotShortestPath(earth, block.QPath, outputPath, ID=block.ID, time = block.creationTime)
            return 0

        # 3. Choose an action (the direction of the next hop)
        nextHop, actIndex = self.getNextHop(newState, linkedSats, sat, block)
        
        # 4. Computes reward/penalty for the previous action
        if prevSat is not None:
            hop = [sat.ID, math.degrees(sat.longitude), math.degrees(sat.latitude)]
            # if the next hop was already visited before the reward will be -1
            if hop in block.QPath[:len(block.QPath)-2]:
                again = againPenalty
            else:
                again = 0

            if distanceRew == 1:
                distanceReward  = getDistanceReward(prevSat, sat, block.destination, self.w2)
            elif distanceRew == 2:
                prevLinkedSats  = getDeepLinkedSats(prevSat, g, earth)
                distanceReward  = getDistanceRewardV2(prevSat, sat, prevLinkedSats['U'], prevLinkedSats['D'], prevLinkedSats['R'], prevLinkedSats['L'], block.destination, self.w2)
            elif distanceRew == 3:
                prevLinkedSats  = getDeepLinkedSats(prevSat, g, earth)
                distanceReward  = getDistanceRewardV3(prevSat, sat, prevLinkedSats['U'], prevLinkedSats['D'], prevLinkedSats['R'], prevLinkedSats['L'], block.destination, self.w2)
            elif distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward  = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
                # distanceReward  = getDistanceRewardV4(prevSat, sat, block.destination, self.w2, self.w4)
            elif distanceRew == 5:
                distanceReward  = getDistanceRewardV5(prevSat, sat, self.w2)

            try:
                queueReward     = getQueueReward   (block.queueTime[len(block.queueTime)-1], self.w1)
            except IndexError:
                queueReward = 0 # FIXME In some hop the queue time was not appended to block.queueTime, line 620
            reward          = distanceReward + again + queueReward

        # 5. Store the experience of previous Node (Agent, satellite) if it was not a gateway  
            self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, False) # action index
            self.earth.rewards.append([reward, sat.env.now])

        # 6. Learning, train the Q-Network at every time we store experience
            if TrainThis and self.step % nTrain == 0:
                self.train(sat, earth)

        else:
            # prev node was a gateway, no need to compute the reward
            reward = 0

        # 7. Align the Q-Target
        if ddqn:
            self.alignQTarget(hardUpdate)

        # this will be saved always, except when the next hop is the destination, where the process will have already returned
        block.oldState  = newState
        block.oldAction = actIndex
        
        return nextHop

    def alignEpsilon(self, step, sat): # the epsilon is reduced with time
        '''
        Updates epsilon value at each step
        0.01+0.99*e^(-0.0005*10000):
        0     -> 1
        1000  -> 0.61
        5000  -> 0.091
        10000 -> 0.01667
        '''
        global      CurrentGTnumber
        epsilon     = self.minEps + (self.maxEps - self.minEps) * math.exp(-LAMBDA * step/(decayRate*(CurrentGTnumber**2)))
        self        .epsilon.append([epsilon, sat.env.now])
        return epsilon

    def alignQTarget(self, hardUpdate = True): # Soft one is done every step
        '''
        This function is not used now since the q target only exists in double deep q learning and it is not implemented.
        Updates the qTarget NN with the weights of the qNetwork.

        The choice between using hard updates or soft updates for the target network depends on the specific requirements of your problem and the properties of your data.

        Hard updates, where the target network is updated with the latest weights of the Q-network, could be more beneficial when the data changes frequently and quickly.
        However, if the data is relatively stable and consistent, then hard updates may cause the target network to oscillate too much, destabilizing the training of the Q-network.

        Soft updates, where the target network's parameters are updated with a moving average of the Q-network's parameters, are more stable than hard updates and can help the
        Q-network converge more smoothly. This is because soft updates gradually propagate the changes in the Q-network's parameters to the target network, rather than suddenly 
        switching to the latest weights. This can be a better choice when the data is relatively stable and consistent, or when you're worried about potential stability issues in
        the training process.

        Ultimately, the best way to determine which method is more convenient is through experimentation with your specific problem and dataset.
        '''
        if hardUpdate:
            self.i += 1
            if self.i == self.updateF:
                self.qTarget.set_weights(self.qNetwork.get_weights()) # NOTE qTarget gets qNetrowk values
                # print(f"Q-Target network hard updated!!!")
                self.i = 0

        else:
            for t, e in zip(self.qTarget.trainable_variables, 
            self.qNetwork.trainable_variables): t.assign(t * (1 - self.tau) + e * self.tau)

    def createModel(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(self.stateSize,), kernel_initializer='random_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='random_uniform'))
        model.add(Dense(self.actionSize, activation='linear'))
        # optimizer = Adam(learning_rate=alpha_dnn)
        # model.compile(loss='mse', optimizer=optimizer)
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self, sat, earth):
        if self.experienceReplay.buffeSize < self.batchS*3:
            return -1

        # 1. Get a random batch from the experience
        miniBatch = self.experienceReplay.getBatch(self.batchS)
        states, actions, rewards, nextStates, Dones = self.experienceReplay.getArraysFromBatch(miniBatch)
        states      = states.reshape((self.batchS,self.stateSize))
        nextStates  = nextStates.reshape((self.batchS,self.stateSize))
         
        # 2. Compute expected reward
        if ddqn:
            futureRewards = self.qTarget(nextStates)           # NOTE NN. Gets future rewards
        else:
            futureRewards = self.qNetwork(nextStates)          # NOTE NN. Gets future rewards
        expectedRewards = rewards + self.gamma*np.max(futureRewards, axis=1)

        # 3. Mask for the actions
        acts = np.eye(self.actionSize)[actions]

        # 4. Stop Loss
        if stopLoss and len(sat.orbPlane.earth.loss)>nLosses:
            savedLoss = sat.orbPlane.earth.loss
            last_n_losses = [sample[0] for sample in savedLoss[-nLosses:]]
            average = sum(last_n_losses) / nLosses 
            sat.orbPlane.earth.lossAv.append(average)
            if average < lThreshold:
                global TrainThis
                TrainThis = False
                print('----------------------------------')
                print(f"STOP LOSS ACTIVATED")
                print(f'Last {nLosses} losses: {last_n_losses}')
                print(f'Simulation time: {sat.env.now}')
                print('----------------------------------')
                return 0

        # 5. fit the model and save the loss
        loss = self.qNetwork.fit(states, acts * expectedRewards[:, None], batch_size=self.batchS, epochs=1, verbose=0) # NOTE qNetwork fit
        sat.orbPlane.earth.loss.append([loss.history['loss'][0], sat.env.now])
        earth.trains.append([sat.env.now]) # counts the number of trainings
        

# @profile
class ExperienceReplay:
    def __init__(self, maxlen = 100):
        '''
        This is a buffer that holds information that are used during training process.

        Deque (Doubly Ended Queue). Deque is preferred over a list in the cases where we need quicker append and pop operations
        from both the ends of the container, as deque provides an O(1) time complexity for append and pop operations as compared
        to a list that provides O(n) time complexity
        '''
        self.buffer = deque(maxlen=maxlen)

    def store(self, state, action, reward, nextState, terminated):
        '''
        appends a set of (state, action, reward, next state, terminated) to the experience replay buffer
        '''
        # if the buffer is full, it behave as a FIFO
        self.buffer.append((state, action, reward, nextState, terminated))

    def getBatch(self, batchSize):
        '''
        gets a random batch of samples from all the samples
        '''
        return random.sample(self.buffer, batchSize)

    def getArraysFromBatch(self, batch):
        '''
        gets the batch data divided into fields
        '''
        states  = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_st = np.array([x[3] for x in batch])
        dones   = np.array([x[4] for x in batch])
        
        return states, actions, rewards, next_st, dones

    @property
    def buffeSize(self):
        '''
        a pythonic way to use getters and setters in object-oriented programming
        this decorator is a built-in function that allows us to define methods that can be accessed like an attribute
        '''
        return len(self.buffer)
        

###############################################################################
############################   Functions    ###################################
###############################################################################


# @profile
def initialize(env, popMapLocation, GTLocation, distance, inputParams, movementTime, totalLocations, outputPath, matching='Greedy'):
    """
    Initializes an instance of the earth with cells from a population map and gateways from a csv file.
    During initialisation, several steps are performed to prepare for simulation:
        - GTs find the cells that within their ground coverage areas and "link" to them.
        - A certain LEO Constellation with a given architecture is created.
        - Satellites are distributed out to GTs so each GT connects to one satellite (if possible) and each satellite
        only has one connected GT.
        - A graph is created from all the GSLs and ISLs
        - Paths are created from each GT to all other GTs
        - Buffers and processes are created on all GTs and satellites used for sending the blocks throughout the network
    """
    print = builtins.print # Idk why but print breaks here so I had to rebuilt it
    # print(type(print))

    constellationType = inputParams['Constellation'][0]
    fraction = inputParams['Fraction'][0]
    testType = inputParams['Test type'][0]
    print(f'Fraction of traffic generated: {fraction}, test type: {testType}')
    # pathing  = inputParams['Pathing'][0]

    if testType == "Rates":
        getRates = True
    else:
        getRates = False

    # Load earth and gateways
    earth = Earth(env, popMapLocation, GTLocation, constellationType, inputParams, movementTime, totalLocations, getRates, outputPath=outputPath)

    print(earth)
    print()

    earth.linkCells2GTs(distance)
    earth.linkSats2GTs("Optimize")
    graph = createGraph(earth, matching=matching)
    earth.graph = graph

    for gt in earth.gateways:
        gt.graph = graph


    paths = []
    # make paths for all source destination pairs
    for GT in earth.gateways:
        for destination in earth.gateways:
            if GT != destination:
                if destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                    path = getShortestPath(GT.name, destination.name, earth.pathParam, GT.graph)
                    GT.paths[destination.name] = path
                    paths.append(path)

    # add ISL references to all satellites and adjust data rate to GTs
    sats = []
    for plane in earth.LEO:
        for sat in plane.sats:
            sats.append(sat)
            # Catalogues the inter-plane ISL as east or west (Right or left)
            sat.findInterNeighbours(earth)

    fiveNeighbors = ([0],[])
    pathNames = [name[0] for name in path]
    for plane in earth.LEO:
        for sat in plane.sats:
            if sat.linkedGT is not None:
                sat.adjustDownRate()
                # make a process for the GSL from sat to GT
                sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))
            neighbors = list(nx.neighbors(graph, sat.ID))
            if len(neighbors) == 5:
                fiveNeighbors[0][0] += 1
                fiveNeighbors[1].append(neighbors)
            itt = 0
            for sat2 in sats:
                if sat2.ID in neighbors:
                    dataRate = nx.path_weight(graph,[sat2.ID, sat.ID], "dataRateOG")
                    distance = nx.path_weight(graph,[sat2.ID, sat.ID], "slant_range")
                    # check if satellite is inter- or intra-plane
                    if sat2.in_plane == sat.in_plane:
                        sat.intraSats.append((distance, sat2, dataRate))
                        # make a send buffer for intra ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                        sat.sendBufferSatsIntra.append(([sat.env.event()], [], sat2.ID))
                        # make a process for intra ISL
                        sat.sendBlocksSatsIntra.append(sat.env.process(sat.sendBlock((distance, sat2, dataRate), True, True)))
                    else:
                        sat.interSats.append((distance, sat2, dataRate))
                        # make a send buffer for inter ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                        sat.sendBufferSatsInter.append(([sat.env.event()], [], sat2.ID))
                        # make a process for inter ISL
                        sat.sendBlocksSatsInter.append(sat.env.process(sat.sendBlock((distance, sat2, dataRate), True, False)))
                    itt += 1
                    if itt == len(neighbors):
                        break

    bottleneck2, minimum2 = findBottleneck(paths[1], earth, False)
    bottleneck1, minimum1 = findBottleneck(paths[0], earth, False, minimum2)

    print('Traffic generated per GT (totalAvgFlow per Milliard):')
    print('----------------------------------')
    for GT in earth.gateways:
        mins = []
        if GT.linkedSat[0] is not None:

            for pathKey in GT.paths:
                _, minimum = findBottleneck(GT.paths[pathKey], earth)
                mins.append(minimum)
            if GT.dataRate < GT.linkedSat[1].downRate:
                GT.getTotalFlow(1, "Step", 1, GT.dataRate, fraction)  # using data rate of the GSL uplink
            else:
                GT.getTotalFlow(1, "Step", 1, GT.linkedSat[1].downRate, fraction)  # using data rate of the GSL downlink
    print('----------------------------------')

    # In case we want to train the constellation we initialize the Q-Tables
    if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning':
        hyperparams = hyperparam(pathing)
    if pathing == 'Deep Q-Learning':
        if not onlinePhase:
            # Initialize global agent
            earth.DDQNA = DDQNAgent(len(earth.gateways), hyperparams, earth)
        else:
            print('----------------------------------')
            print('Creating satellites agents...')
            if importQVals:
                print:(f'Importing the Neural networks from: \n{nnpath}\n{nnpathTarget}')
            for plane in earth.LEO:
                for sat in plane.sats:
                    sat.DDQNA = DDQNAgent(len(earth.gateways), hyperparams, earth, sat.ID)
            print('----------------------------------')

    # save hyperparams
    if pathing == 'Q-Learning' or pathing == "Deep Q-Learning":
        saveHyperparams(earth.outputPath, inputParams, hyperparams)

    if pathing == 'Q-Learning':
        '''
        Q-Agents are initialized here
        '''
        earth.initializeQTables(len(earth.gateways), hyperparams, graph)

    return earth, graph, bottleneck1, bottleneck2


# @profile
def findBottleneck(path, earth, plot = False, minimum = None):
    # Find the bottleneck of a route.
    bottleneck = [[], [], [], []]
    for GT in earth.gateways:
        if GT.name == path[0][0]:
            bottleneck[0].append(str(path[0][0].split(",")[0]) + "," + str(path[1][0]))
            bottleneck[1].append(GT.dataRate)
            bottleneck[2].append(GT.latitude)
            if minimum:
                bottleneck[3].append(minimum/GT.dataRate)

    for i, step in enumerate(path[1:], 1):
        for orbit in earth.LEO:
            for satellite in orbit.sats:
                if satellite.ID == step[0]:

                    for sat in satellite.interSats:
                        if sat[1].ID == path[i + 1][0]:
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])
                    for sat in satellite.intraSats:
                        if sat[1].ID == path[i + 1][0]:
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])
    for GT in earth.gateways:
        if GT.name == path[-1][0]:
            bottleneck[0].append(str(path[-2][0]) + "," + str(path[-1][0].split(",")[0]))
            bottleneck[1].append(GT.linkedSat[1].downRate)
            bottleneck[2].append(GT.latitude)
            if minimum:
                bottleneck[3].append(minimum/GT.dataRate)

    if plot:
        earth.plotMap(True,True,path, bottleneck)
        plt.show()
        plt.close()

    minimum = np.amin(bottleneck[1])
    return bottleneck, minimum


# @profile
def create_Constellation(specific_constellation, env, earth):

    if specific_constellation == "small":               # Small Walker star constellation for tests.
        print("Using small walker Star constellation")
        P = 4					# Number of orbital planes
        N_p = 8 				# Number of satellites per orbital plane
        N = N_p*P				# Total number of satellites
        height = 1000e3			# Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 53	# Inclination angle for the orbital planes, set to 90 for Polar
        Walker_star = True		# Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30

    elif specific_constellation =="Kepler":
        print("Using Kepler constellation design")
        P = 7
        N_p = 20
        N = N_p*P
        height = 600e3
        inclination_angle = 98.6
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="Iridium_NEXT":
        print("Using Iridium NEXT constellation design")
        P = 6
        N_p = 11
        N = N_p*P
        height = 780e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="OneWeb":
        print("Using OneWeb constellation design")
        P = 18
        N = 648
        N_p = int(N/P)
        height = 1200e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="Starlink":			# Phase 1 550 km altitude orbit shell
        print("Using Starlink constellation design")
        P = 72
        N = 1584
        N_p = int(N/P)
        height = 550e3
        inclination_angle = 53
        Walker_star = False
        min_elevation_angle = 25

    elif specific_constellation == "Test":
        print("Using a test constellation design")
        P = 30                     # Number of orbital planes
        N = 1200                   # Total number of satellites
        N_p = int(N/P)             # Number of satellites per orbital plane
        height = 600e3             # Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 86.4   # Inclination angle for the orbital planes, set to 90 for Polar
        Walker_star = True         # Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30
    else:
        print("Not valid Constellation Name")
        P = np.NaN
        N_p = np.NaN
        N = np.NaN
        height = np.NaN
        inclination_angle = np.NaN
        Walker_star = False
        exit()

    distribution_angle = 2*math.pi  # Angle in which the orbital planes are distributed in

    if Walker_star:
        distribution_angle /= 2
    orbital_planes = []

    # Add orbital planes and satellites
    # Orbital_planes.append(orbital_plane(0, height, 0, math.radians(inclination_angle), N_p, min_elevation_angle, 0))
    for i in range(0, P):
        orbital_planes.append(OrbitalPlane(str(i), height, i*distribution_angle/P, math.radians(inclination_angle), N_p,
                                           min_elevation_angle, str(i) + '_', env, earth))

    return orbital_planes


###############################################################################
###############################  Create Graph   ###############################
###############################################################################


def get_direction(Satellites):
    '''
    Gets the direction of the satellites so each transceiver antenna can be set to one direction.
    '''
    N = len(Satellites)
    direction = np.zeros((N,N), dtype=np.int8)
    for i in range(N):
        epsilon = -Satellites[i].inclination    # orbital plane inclination
        for j in range(N):
            direction[i,j] = np.sign(Satellites[i].y*math.sin(epsilon)+
                                    Satellites[i].z*math.cos(epsilon)-Satellites[j].y*math.sin(epsilon)-
                                    Satellites[j].z*math.cos(epsilon))
    return direction


def get_pos_vectors_omni(Satellites):
    '''
    Given a list of satellites returns a list with x, y, z coordinates and the plane where they are (meta)
    '''
    N = len(Satellites)
    Positions = np.zeros((N,3))
    meta = np.zeros(N, dtype=np.int_)
    for n in range(N):
        Positions[n,:] = [Satellites[n].x, Satellites[n].y, Satellites[n].z]
        meta[n] = Satellites[n].in_plane

    return Positions, meta


def get_slant_range(edge):
        return(edge.slant_range)


# @numba.jit  # Using this decorator you can mark a function for optimization by Numba's JIT compiler
def get_slant_range_optimized(Positions, N):
    '''
    returns a matrix with the all the distances between the satellites (optimized)
    '''
    slant_range = np.zeros((N,N))
    for i in range(N):
        slant_range[i,i] = math.inf
        for j in range(i+1,N):
            slant_range[i,j] = np.linalg.norm(Positions[i,:] - Positions[j,:])
    slant_range += np.transpose(slant_range)
    return slant_range


@numba.jit  # Using this decorator you can mark a function for optimization by Numba's JIT compiler
def los_slant_range(_slant_range, _meta, _max, _Positions):
    '''
    line of sight slant range
    '''
    _slant_range_new = np.copy(_slant_range)
    _N = len(_slant_range)
    for i in range(_N):
        for j in range(_N):
            if _slant_range_new[i,j] > _max[_meta[i], _meta[j]]:
                _slant_range_new[i,j] = math.inf
    return _slant_range_new


def get_data_rate(_slant_range_los, interISL):
    """
    Given a matrix of slant ranges returns a matrix with all the shannon dataRates possibles between all the satellites.
    """
    speff_thresholds = np.array(
        [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
         1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
         2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
         3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
         5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
    lin_thresholds = np.array(
        [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
         1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
         3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
         8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
         16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
         45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])

    pathLoss = 10*np.log10((4 * math.pi * _slant_range_los * interISL.f / Vc)**2)   # Free-space pathloss in dB
    snr = 10**((interISL.maxPtx_db + interISL.G - pathLoss - interISL.No)/10)       # SNR in times
    shannonRate = interISL.B*np.log2(1+snr)                                         # data rates matrix in bits per second

    speffs = np.zeros((len(_slant_range_los),len(_slant_range_los)))

    for n in range(len(_slant_range_los)):
        for m in range(len(_slant_range_los)):
            feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr[n,m])]
            if feasible_speffs.size == 0:
                speffs[n, m] = 0
            else:
                speffs[n,m] = interISL.B * feasible_speffs[-1]

    return speffs


def markovianMatchingTwo(earth):
    '''
    Returns a list of edge class elements. Each edge stands for a connection between two satellites. On that class
    the slant range and the data rate between both satellites are stored as attributes.
    This function is for satellites with two transceivers antennas that will enable two inter-plane ISL each one
    in a different direction.
    Intra-plane ISL are also computed and returned in _A_Markovian list

    It is not the optimal solution, but it is from 10 to 1000x faster.
    Minimizes the total cost of the constellation matching problem.
    '''

    _A_Markovian    = []    # list with all the
    Satellites      = []    # list with all the satellites
    W_M             = []    # list with the distances of every possible link between sats
    covered         = set() # Set with the connections already covered

    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    N = len(Satellites)

    interISL = RFlink(
        frequency=26e9,
        bandwidth=500e6,
        maxPtx=10,
        aDiameterTx=0.26,
        aDiameterRx=0.26,
        pointingLoss=0.3,
        noiseFigure=2,
        noiseTemperature=290,
        min_rate=10e3
    )

    # max slant range for each orbit
    ###########################################################
    M = len(earth.LEO)              # Number of planes in LEO
    Max_slnt_rng = np.zeros((M,M))  # All ISL slant ranges must me lowe than 'Max_slnt_rng[i, j]'

    Orb_heights  = []
    for plane in earth.LEO:
        Orb_heights.append(plane.h)
        maxSlantRange = plane.sats[0].maxSlantRange

    for _i in range(M):
        for _j in range(M):
            Max_slnt_rng[_i,_j] = (np.sqrt( (Orb_heights[_i] + Re)**2 - Re**2 ) +
                                np.sqrt( (Orb_heights[_j] + Re)**2 - Re**2 ) )


    # Get data rate old method
    ###########################################################
    direction       = get_direction(Satellites)             # get both directions of the satellites to use the two transceivers
    Positions, meta = get_pos_vectors_omni(Satellites)      # position and plane of all the satellites
    slant_range     = get_slant_range_optimized(Positions, N)                       # matrix with all the distances between satellties
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng, Positions)   # distance matrix but if d>dMax, d=infinite
    shannonRate     = get_data_rate(slant_range_los, interISL)                      # max dataRate

    '''
    Compute all possible edges between different plane satellites whose transceiver antennas are free.
    if slant range > max slant range then that edge is not added
    '''
    ###########################################################
    for i in range(N):
        for j in range(i+1,N):
            if Satellites[i].in_plane != Satellites[j].in_plane and ((i,direction[i,j]) not in covered) and ((j,direction[j,i]) not in covered):
                if slant_range_los[i,j] < 6000e3: # math.inf:
                    W_M.append(edge(Satellites[i].ID,Satellites[j].ID,slant_range_los[i,j],direction[i,j], direction[j,i], shannonRate[i,j]))

    W_sorted=sorted(W_M,key=get_slant_range) # NOTE we could choose shannonRate instead

    # from all the possible links adds only the uncovered with the best weight possible
    ###########################################################
    while W_sorted:
        if  ((W_sorted[0].i,W_sorted[0].dji) not in covered) and ((W_sorted[0].j,W_sorted[0].dij) not in covered):
            _A_Markovian.append(W_sorted[0])
            covered.add((W_sorted[0].i,W_sorted[0].dji))
            covered.add((W_sorted[0].j,W_sorted[0].dij))
        W_sorted.pop(0)

    # add intra-ISL edges
    ###########################################################
    for plane in earth.LEO:
        nPerPlane = len(plane.sats)
        for sat in plane.sats:
            sat.findIntraNeighbours(earth)

            # upper neighbour
            i = sat.in_plane        *nPerPlane    +sat.i_in_plane
            j = sat.upper.in_plane  *nPerPlane    +sat.upper.i_in_plane

            _A_Markovian.append(edge(sat.ID, sat.upper.ID,  # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            direction[i,j], direction[j,i],                 # directions
            shannonRate[i,j]))                              # Max dataRate

            # lower neighbour
            j = sat.lower.in_plane  *nPerPlane    +sat.lower.i_in_plane

            _A_Markovian.append(edge(sat.ID, sat.lower.ID,  # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            direction[i,j], direction[j,i],                 # directions
            shannonRate[i,j]))                              # Max dataRate

    return _A_Markovian


def greedyMatching(earth):
    '''
    Returns a list of edge class elements based on a greedy algorithm.
    Each satellite is connected to its upper and lower satellite in the same orbital plane (intra-plane),
    and the nearest satellites to the east and west in different planes (inter-plane).
    The slant range and the data rate between satellites are stored as attributes in the edge class.
    '''

    _A_Greedy = []  # list to store edges
    Satellites = []  # list of all satellites

    # Collect all satellites from each plane
    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    N = len(Satellites)

    # inter-plane ISL 
    ##############################################################
    # link parameters
    interISL = RFlink(
        frequency=f,
        bandwidth=B,
        maxPtx=maxPtx,
        aDiameterTx=Adtx,
        aDiameterRx=Adrx,
        pointingLoss=pL,
        noiseFigure=Nf,
        noiseTemperature=Tn,
        min_rate=min_rate
    )

   # max slant range for each orbit
    ###########################################################
    M = len(earth.LEO)              # Number of planes in LEO
    Max_slnt_rng = np.zeros((M,M))  # All ISL slant ranges must be lowe than 'Max_slnt_rng[i, j]'

    Orb_heights  = []
    for plane in earth.LEO:
        Orb_heights.append(plane.h)
        maxSlantRange = plane.sats[0].maxSlantRange

    for _i in range(M):
        for _j in range(M):
            Max_slnt_rng[_i,_j] = (np.sqrt( (Orb_heights[_i] + Re)**2 - Re**2 ) +
                                np.sqrt( (Orb_heights[_j] + Re)**2 - Re**2 ) )
            
    # Compute positions and slant ranges
    ##############################################################
    direction       = get_direction(Satellites)             # get both directions of the satellites to use the two transceivers
    Positions, meta = get_pos_vectors_omni(Satellites)      # position and plane of all the satellites
    slant_range     = get_slant_range_optimized(Positions, N)                       # matrix with all the distances between satellties
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng, Positions)   # distance matrix but if d>dMax, d=infinite
    shannonRate     = get_data_rate(slant_range_los, interISL)                      # max dataRate

    # Create edges for inter-plane links (closest east and west satellites)
    for i, sat in enumerate(Satellites):
        closest_east, closest_west = None, None
        min_east_distance, min_west_distance = float('inf'), float('inf')

        for j, other_sat in enumerate(Satellites):
            if sat.in_plane != other_sat.in_plane:
                if slant_range_los[i, j] < min_east_distance and Positions[j, 0] > Positions[i, 0]:  # East satellite
                    closest_east, min_east_distance = other_sat, slant_range_los[i, j]
                elif slant_range_los[i, j] < min_west_distance and Positions[j, 0] < Positions[i, 0]:  # West satellite
                    closest_west, min_west_distance = other_sat, slant_range_los[i, j]

        # Add edges for closest east and west satellites
        if closest_east:
            _A_Greedy.append(edge(sat.ID, closest_east.ID, min_east_distance, None, None, shannonRate[i, Satellites.index(closest_east)]))
        if closest_west:
            _A_Greedy.append(edge(sat.ID, closest_west.ID, min_west_distance, None, None, shannonRate[i, Satellites.index(closest_west)]))
        
    # intra-plane ISL links (upper and lower neighbors)
    ##############################################################
    for plane in earth.LEO:
        nPerPlane = len(plane.sats)
        for sat in plane.sats:
            sat.findIntraNeighbours(earth)

            # upper neighbour
            i = sat.in_plane        *nPerPlane    +sat.i_in_plane
            j = sat.upper.in_plane  *nPerPlane    +sat.upper.i_in_plane

            _A_Greedy.append(edge(sat.ID, sat.upper.ID,     # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            None, None,                                     # directions
            shannonRate[i,j]))                              # Max dataRate

            # lower neighbour
            j = sat.lower.in_plane  *nPerPlane    +sat.lower.i_in_plane

            _A_Greedy.append(edge(sat.ID, sat.lower.ID,     # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            None, None,                                     # directions
            shannonRate[i,j]))                              # Max dataRate

    return _A_Greedy


def deleteDuplicatedLinks(satA, g, earth):
    '''
    Given a satellite, searches for its east and west neighbour. If the east or west link is duplicated,
    it will remove the link with a higher latitude difference, keeping the horizontal links
    '''

    def getMostHorizontal(currentSat, satA, satB):
        '''
        Chooses the dat with the closest latitude to currentSat
        '''
        return (satA, satB) if abs(satA.latitude-currentSat.latitude)<abs(satB.latitude-currentSat.latitude) else (satB, satA)

    linkedSats = {'U':None, 'D':None, 'R':None, 'L':None}
    for edge in list(g.edges(satA.ID)):
        if edge[1][0].isdigit():
            satB = findByID(earth, edge[1])
            dir = getDirection(satA, satB)

            if(dir == 3):                                         # Found Satellite at East
                if linkedSats['R'] is not None:
                    # print(f"{satA.ID} east satellite duplicated: {linkedSats['R'].ID}, {satB.ID}")
                    most_horizontal, less_horizontal = getMostHorizontal(satA, linkedSats['R'], satB)
                    # print(f'Keeping most horizontal link: {most_horizontal.ID}')
                    linkedSats['R']  = most_horizontal
                    # remove pair from G
                    g.remove_edge(satA.ID, less_horizontal.ID)
                else:
                    linkedSats['R']  = satB

            elif(dir == 4):                                         # Found Satellite at West
                if linkedSats['L'] is not None:
                    # print(f"{satA.ID} West satellite duplicated: {linkedSats['L'].ID}, {satB.ID}")
                    most_horizontal, less_horizontal = getMostHorizontal(satA, linkedSats['L'], satB)
                    # print(f'Keeping most horizontal link: {most_horizontal.ID}')
                    linkedSats['L']  = most_horizontal
                    # remove pair from G
                    g.remove_edge(satA.ID, less_horizontal.ID)
                else:
                    linkedSats['L']  = satB


def establishRemainingISLs(earth, g):
    Satellites = []

    # Collect all satellites from each plane
    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    # Gather positions and other parameters
    Positions, meta = get_pos_vectors_omni(Satellites)
    direction = get_direction(Satellites)
    slant_range = get_slant_range_optimized(Positions, len(Satellites))

    # Prepare link parameters
    interISL = RFlink(
        frequency=f,
        bandwidth=B,
        maxPtx=maxPtx,
        aDiameterTx=Adtx,
        aDiameterRx=Adrx,
        pointingLoss=pL,
        noiseFigure=Nf,
        noiseTemperature=Tn,
        min_rate=min_rate
    )

    # Calculate maximum slant range
    Max_slnt_rng = np.zeros((len(earth.LEO), len(earth.LEO)))
    Orb_heights = [plane.h for plane in earth.LEO]
    for i in range(len(earth.LEO)):
        for j in range(len(earth.LEO)):
            Max_slnt_rng[i, j] = (np.sqrt((Orb_heights[i] + Re)**2 - Re**2) +
                                  np.sqrt((Orb_heights[j] + Re)**2 - Re**2))

    # Define slant range and data rate matrices
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng, Positions)
    shannonRate = get_data_rate(slant_range_los, interISL)

    # Identify satellites with specific missing neighbors
    satellites_with_no_right = {sat: Positions[idx] for idx, sat in enumerate(Satellites) if sat.right is None}
    satellites_with_no_left = {sat: Positions[idx] for idx, sat in enumerate(Satellites) if sat.left is None}

    # Calculate potential matches sorted by horizontal alignment
    potential_links = []
    for sat_r in satellites_with_no_right:
        for sat_l in satellites_with_no_left:
            if sat_r.in_plane != sat_l.in_plane:
                idx_r = Satellites.index(sat_r)
                idx_l = Satellites.index(sat_l)
                if slant_range_los[idx_r, idx_l] < math.inf:
                    # Handle longitude wrapping correctly
                    longitude_difference = (satellites_with_no_left[sat_l][0] - satellites_with_no_right[sat_r][0] + 360) % 360
                    if longitude_difference > 0 and longitude_difference < 180:
                        # lat_diff = abs(satellites_with_no_right[sat_r][1] - satellites_with_no_left[sat_l][1])
                        lat_diff = abs(sat_r.latitude-sat_l.latitude)
                        potential_links.append((lat_diff, sat_r, sat_l, slant_range_los[idx_r, idx_l]))

    # Sort by latitude difference to prioritize horizontal links
    # potential_links.sort()
    potential_links.sort(key=lambda x: x[0])  # Uses latitude difference as sort key


    # Establish links from closest to farthest in terms of horizontal alignment
    for lat_diff, sat_r, sat_l, distance in potential_links:
        if sat_r.right is None and sat_l.left is None:
            g.add_edge(sat_r.ID, sat_l.ID, slant_range=distance,
                       dataRate=1/shannonRate[Satellites.index(sat_r), Satellites.index(sat_l)],
                       dataRateOG=shannonRate[Satellites.index(sat_r), Satellites.index(sat_l)], hop=1)
            sat_r.right = sat_l
            sat_l.left = sat_r
            # print(f"Established horizontal link between {sat_r.ID} (right) and {sat_l.ID} (left) with latitude difference {lat_diff:.2f} deg and distance: {distance/1000:.2F} km.")

    return g


def createGraph(earth, matching='Greedy'):
    '''
    Each satellite has two transceiver antennas that are connected to the closest satellite in east and west direction to a satellite
    from another plane (inter-ISL). Each satellite also has anoteher two transceiver antennas connected to the previous and to the
    following satellite at their orbital plane (intra-ISL).
    A graph is created where each satellite is a node and each connection is an edge with a specific weight based either on the
    inverse of the maximum data rate achievable, total distance or number of hops.
    '''
    g = nx.Graph()

    # add LEO constellation
    ###############################
    for plane in earth.LEO:
        for sat in plane.sats:
            g.add_node(sat.ID, sat=sat)

    # add gateways and GSL edges
    ###############################
    for GT in earth.gateways:
        if GT.linkedSat[1]:
            g.add_node(GT.name, GT = GT)            # add GT as node
            g.add_edge(GT.name, GT.linkedSat[1].ID, # add GT linked sat as edge
            slant_range = GT.linkedSat[0],          # slant range
            invDataRate = 1/GT.dataRate,            # Inverse of dataRate
            dataRateOG = GT.dataRate,               # original shannon dataRate
            hop = 1)                                # in case we just want to count hops

    # add inter-ISL and intra-ISL edges
    ###############################
    if matching=='Markovian':
        markovEdges = markovianMatchingTwo(earth)
    elif matching=='Greedy':
        markovEdges = greedyMatching(earth)
    print(f'Matching: {matching}')
    # print('----------------------------------')

    global biggestDist
    global firstMove
    # biggestDist = -1
    for markovEdge in markovEdges:
        g.add_edge(markovEdge.i, markovEdge.j,  # source and destination IDs
        slant_range = markovEdge.slant_range,   # slant range
        dataRate = 1/markovEdge.shannonRate,    # Inverse of dataRate # FIXME sometimes markovEdge.shannonRate is 0
        dataRateOG = markovEdge.shannonRate,    # Original shannon datRate
        hop = 1,                                # in case we just want to count hops
        dij = markovEdge.dij,
        dji = markovEdge.dji)
        if firstMove and markovEdge.slant_range > biggestDist:  # keep the biggest possible distance for the normalization of the rewards
            biggestDist = markovEdge.slant_range

    # remove duplicated links and keep the most horizontal ones
    print('Removing duplicated links...')
    for plane in earth.LEO:
        for sat in plane.sats:
            deleteDuplicatedLinks(sat, g, earth)
        
    earth.graph = g
    
    # update the neighbors
    for plane in earth.LEO:
        for sat in plane.sats:
            sat.findIntraNeighbours(earth)
            sat.findInterNeighbours(earth)

    print('Establishing remaining edges...')
    g = establishRemainingISLs(earth, g)


    if firstMove:
        print(f'Biggest slant range between satellites: {biggestDist/1000:.2f} km')
        firstMove = False
    print('----------------------------------')

    return g


def getShortestPath(source, destination, weight, g):
    '''
    Gives you the shortest path between a source and a destination and plots it if desired.
    Uses the 'dijkstra' algorithm to compute the sortest path, where the total weight of the path can be either the sum of inverse
    of the maximumm dataRate achevable, the total slant range or the number of hops taken between source and destination.

    returns a list where each element is a sublist with the name of the node, its longitude and its latitude.
    '''

    path = []
    try:
        shortest = nx.shortest_path(g, source, destination, weight = weight)    # computes the shortest path [dataRate, slant_range, hops]
        for hop in shortest:                                                    # pre process the data so it can be used in the future
            key = list(g.nodes[hop])[0]
            if shortest.index(hop) == 0 or shortest.index(hop) == len(shortest)-1:
                path.append([hop, g.nodes[hop][key].longitude, g.nodes[hop][key].latitude])
            else:
                path.append([hop, math.degrees(g.nodes[hop][key].longitude), math.degrees(g.nodes[hop][key].latitude)])
    except Exception as e:
        print(f"getShortestPath Caught an exception: {e}")
        print('No path between ' + source + ' and ' + destination + ', check the graph to see more details.')
        return -1
    return path


def plotShortestPath(earth, path, outputPath, ID=None, time=None):
    earth.plotMap(True, True, path=path, ID=ID,time=time)
    plt.savefig(outputPath + 'popMap_' + path[0][0] + '_to_' + path[len(path)-1][0] + '.png', dpi = 500)
    # plt.show()
    plt.close()


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


###############################################################################
#########################    Q-Tables - StateSpace    #########################
###############################################################################


def watchScores(earth, g):
    '''
    This function will print the scores of each satellite at that moment
    The satellites with any missing queue are those who does not have the 4 linked satellites. All of them are inter-plane ISL
    '''
    print('----------------------------------')
    print("SCORES:\n")
    print('----------------------------------')
    for plane in earth.LEO:
        for sat in plane.sats:
            print('-----------------')
            print(sat.ID + ": ")
            print('-----------------')
            for edge in list(g.edges(sat.ID)):
                if edge[1][0].isdigit():    # pos 1 regarding to the linked node and position 0 regarding to the first character of the linked node
                    print('Score between ' + str(edge) + ': ' + str(getSatScore(findByID(earth, edge[0]), findByID(earth, edge[1]), g)))
                else:
                    print('Gateway linked: ' + str(edge))


def findByID(earth, satID):
    '''
    given the ID of a satellite, this function will return the corresponding satellite object
    '''
    for plane in earth.LEO:
        for sat in plane.sats:
            if (sat.ID == satID):
                return sat


def computeOutliers(g):
    '''
    Given a graph, will return the throughput and slant range thresholds that will be used to find the outliers
    (Devices with bad conditions)
    '''
    # define outliers
    slantRanges = []
    dataRates   = []

    for edge in list(g.edges()):
        slantRanges.append(g.edges[edge]['slant_range'])
        dataRates  .append(g.edges[edge]['dataRateOG'])

    # Slant Range Outliers
    slantRanges = pd.Series(slantRanges)
    Q3 = slantRanges.describe()['75%']
    Q1 = slantRanges.describe()['25%']
    IQR = Q3 - Q1
    upperFence = Q3 + (1.5*IQR)

    # Data Rate Outliers
    dataRates = pd.Series(dataRates)
    Q3 = dataRates.describe()['75%']
    Q1 = dataRates.describe()['25%']
    IQR = Q3 - Q1
    lowerFence = Q1 - (1.5*IQR)

    return lowerFence, upperFence


def getQueues(sat, threshold=None, DDQN = False):
    '''
    When !DDQN, this function will return True if one of the satellite queues has a length over a limit or they are
    missing one link

    Each satellite has a queue for each link which includes both ISL and GSL (sat 2 GT). The Queues are implemented as
    tuples that contain a list of simpy events, a list of the data blocks, and the ID of the satellite for the link
    (there is no ID for the GT queues). The structure is tuple[list[Simpy.event], list[DataBlock], ID].
    The list of events will always have at least one event present which will be non-triggered when there are no blocks
    in the queue. When blocks are present, there will be as many triggered events as there are blocks.

    On the GTs, there is one queue which has the same structure as the queues for the GSLs on the satellites:
    tuple[list[Simpy.event], list[DataBlock]]

    ISLs Queues: sendBufferSats where each entry is a separate queue.
    GSLs Queues: sendBufferGT. While there will never be more than one queue in this list.
    GTs  Queues: sendBuffer which is just the tuple itself

    In our case we will just choose the highest queue of all the ISLs and compare it to a threshold

    The try excepts are for those cases where the linked satellite does not have the 4 linked satllites queues.
    IF THE SATELLITE DOES NOT HAVE 4 LINEKD SATELLITES IT WILL BE CONSIDERED AS HIGH QUEUE
    '''
    queuesLen = []
    infQueue  = False
    queuesDic = {'U': np.inf,
                 'D': np.inf,
                 'R': np.inf,
                 'L': np.inf}
    try:
       queuesLen.append(len(sat.sendBufferSatsIntra[0][1]))
       queuesDic['U'] = len(sat.sendBufferSatsIntra[0][1])
    except (IndexError, AttributeError):
        infQueue = True
    try:
       queuesLen.append(len(sat.sendBufferSatsIntra[1][1]))
       queuesDic['D'] = len(sat.sendBufferSatsIntra[1][1])

    except (IndexError, AttributeError):
        infQueue = True
    try:
        queuesLen.append(len(sat.sendBufferSatsInter[0][1]))
        queuesDic['R'] = len(sat.sendBufferSatsInter[0][1])
    except (IndexError, AttributeError):
        infQueue = True
    try:
        queuesLen.append(len(sat.sendBufferSatsInter[1][1]))
        queuesDic['L'] = len(sat.sendBufferSatsInter[1][1])
    except (IndexError, AttributeError):
        infQueue = True

    if not DDQN:
        return max(queuesLen) > threshold or infQueue
    else:
        return queuesDic


def hasBadConnection(satA, satB, thresholdSL, thresholdTHR, g):
    '''
    This function will return true if the satellites distance between them > trheshold or if their throughpuyt < trheshold
    They are far away or the link is weak
    '''
    slantRange     = g.edges[satA.ID, satB.ID]['slant_range']
    throughputSats = g.edges[satA.ID, satB.ID]['dataRateOG']

    return (slantRange > thresholdSL or throughputSats < thresholdTHR)


def getSatScore(satA, satB, g):
    '''
    This function will compute the score of sending the package from satA to satB
    0: (Low  slant range || high throughput) && low queue
    1:  High slant range && low  throughput  && low queue
    2:  High queue

    Queue threshold:
    As high queue threshold we have set 125 packets, which is the 92 percentile of all the queues when we have 13 GTs
    (The moment when we start having congestion with slant range policy). The waiting time of a queue with 125 blocks
    is 9 msg (Each packet in the queue lasts ~0.072ms)
    '''
    thresholdQueue = 125
    thresholdTHR, thresholdSL = computeOutliers(g)

    if satB is None or getQueues(satB, thresholdQueue):
        return 2
    elif hasBadConnection(satA, satB, thresholdSL, thresholdTHR, g):
        return 1
    else:
        return 0


# @profile
def getDeepSatScore(queueLength):
    # return 1 if queueLength > infQueue else (int(np.floor(queueVals*np.log10(queueLength + 1)/np.log10(infQueue))))/queueVals
    return queueVals if queueLength > infQueue else int(np.floor(queueVals*np.log10(queueLength + 1)/np.log10(infQueue)))


def getDirection_deprecated(satA, satB):
    '''
    Returns the direction of going from satA to satB.
    If the satellites are very far away (More than half of the radious of the Earth, pi) the East-West logic is reversed

    If the node is not previous to the linked one we will treat it as a reversed way.

    Dir 1 (Go Upper): lower  -> higher latitude
    Dir 2 (Go Lower): higher -> lower  latitude
    Dir 3 (Go Right): lower  -> higher longitude
    Dir 4 (Go left) : higher -> lower  longitude
    '''

    planei = int(satA.in_plane)
    planej = int(satB.in_plane)

    if planei == planej:
        if satA.latitude < satB.latitude:
            return 1
        else:
            return 2
    if(abs(abs(satA.longitude) - abs(satB.longitude)) < math.pi):         # they are not too far away
        if satA.longitude < satB.longitude:
            return 3
        else:
            return 4
    else:                                                       # they are very far away
        if satA.longitude > satB.longitude:
            return 3
        else:
            return 4


def getDirection(satA, satB):
    '''
    Returns the direction from satA to satB, considering the Earth's wrap-around for longitude.
    '''

    def normalize_longitude(lon):
        # Normalize longitude to the range [-math.pi, math.pi]
        return ((lon + math.pi) % (2 * math.pi)) - math.pi

    planei = int(satA.in_plane)
    planej = int(satB.in_plane)

    if planei == planej:
        if satA.latitude < satB.latitude:
            return 1  # Go Upper
        else:
            return 2  # Go Lower

    # Normalize the longitudes
    norm_lonA = normalize_longitude(satA.longitude)
    norm_lonB = normalize_longitude(satB.longitude)

    # Calculate the normalized longitude difference
    lon_diff = normalize_longitude(norm_lonB - norm_lonA)

    # Decide direction based on normalized difference
    if lon_diff > 0:
        return 3  # Go Right
    else:
        return 4  # Go Left


def linkedSatsList(g):
    '''
    This funtion retunrs a dictionary (Gateway: linekdSatellite)
    '''
    linkedSats = []
    for node in g.nodes:
        if not node[0].isdigit():
            linkedSats.append(list(g.edges(node))[0])
    return pd.DataFrame(linkedSats)


def getDestination(Block, g, sat = None):
    '''
    Returns:
    blockDestination: Position of the satellite linked to the block destination Gateway among a list of all the
                      satellites linked to Gateways
    linkedGateway:    If the satellite provided is linked to a gateway, it will return the position of the satellite in
                      the mentioned list. Otherwise it will return -1.
    '''
    destination = list(g.edges(Block.destination.name))[0][1]    # ID of the Satellite linked to the block destination GT
    blockDestination = (linkedSatsList(g)[1] == destination).argmax()

    if sat is None:
        return blockDestination
    else:
        pass
        # satDest = Block.destination.linkedSat[1]
        # return getGridPosition(GridSize, [tuple([math.degrees(satDest.latitude), math.degrees(satDest.longitude), satDest.ID])], False, False)[0]


def getLinkedSats(satA, g, earth):
    '''
    Given a satellite the function will return a list with the linked satellite at each direction.
    If that direction has no linked satellite, it will be None
    At the graph each edge is a satA, satB pair with properties like dirij or dirji, i will always
    be the satellite of the lowest plane and 1 will be righ direction (East).

    SAT UP:      northest linked satellite
    SAT DOWN:    southest linked satellite
    SAT LEFT:    linked satellite with lower  plane ID
    SAT RIGHT:   linked satellite with higher plane ID
    '''
    linkedSats = {'U':None, 'D':None, 'R':None, 'L':None}
    for edge in list(g.edges(satA.ID)):
        if edge[1][0].isdigit():
            satB = findByID(earth, edge[1])
            dir = getDirection(satA, satB)

            if(dir == 1 and linkedSats['U'] is None):               # Found a satellite at north
                linkedSats['U']  = satB
            elif(dir == 1):                                         # Found second North, this sat is on South Pole
                if satB.latitude > linkedSats['U'].latitude:
                    # the satellite seen is more at north than Up one, so is set as new Up
                    linkedSats['D'] = linkedSats['U']
                    linkedSats['U'] = satB
                else:
                    # the satellite seen is less at north than Up one, so is set as Down
                    linkedSats['D'] = satB

            elif(dir == 2 and linkedSats['D'] is None):             # Found satellite at South
                linkedSats['D']  = satB   
            elif(dir == 2):                                         # Found second Down, this sat is on North Pole
                if satB.latitude < linkedSats['D'].latitude:        
                    linkedSats['U'] = linkedSats['D']
                    linkedSats['D'] = satB
                else:
                    linkedSats['U'] = satB

            elif(dir == 3):                                         # Found Satellite at East
                # if linkedSats['R'] is not None:
                #     print(f"{satA.ID} east satellite duplicated! Replacing {linkedSats['R'].ID} with {satB.ID}")
                linkedSats['R']  = satB

            elif(dir == 4):                                         # Found Satellite at West
                # if linkedSats['L'] is not None:
                #     print(f"{satA.ID} west satellite duplicated! Replacing {linkedSats['L'].ID} with {satB.ID}")
                linkedSats['L']  = satB

        else:
            pass
    return linkedSats


def getDeepLinkedSats(satA, g, earth):
    '''
    Given a satellite, this function will return a dictionary with the linked satellite
    at each direction based on the new definition of upper and lower satellites.
    Satellite at the right and left are determined based on inter-plane links.
    '''
    linkedSats = {'U':None, 'D':None, 'R':None, 'L':None}

    # Use the provided logic to find intra-plane neighbours (upper and lower)
    # satA.findIntraNeighbours(earth)
    linkedSats['U'] = satA.upper
    linkedSats['D'] = satA.lower
    linkedSats['R'] = satA.right
    linkedSats['L'] = satA.left

    # # Find inter-plane neighbours (right and left)
    # for edge in list(g.edges(satA.ID)):
    #     if edge[1][0].isdigit():
    #         satB = findByID(earth, edge[1])
    #         dir = getDirection(satA, satB)
    #         if(dir == 3):                                         # Found Satellite at East
    #             if linkedSats['R'] is not None:
    #                 print(f"{satA.ID} east satellite duplicated! Replacing {linkedSats['R'].ID} with {satB.ID}")
    #             linkedSats['R']  = satB

    #         elif(dir == 4):                                       # Found Satellite at West
    #             if linkedSats['L'] is not None:
    #                 print(f"{satA.ID} west satellite duplicated! Replacing {linkedSats['L'].ID} with {satB.ID}")
    #             linkedSats['L']  = satB
    #     else:
    #         pass

    return linkedSats


def getState(Block, satA, g, earth):
    '''
    Given a dataBlock and the current satellite this function will return a list with the 
    values of the 5 fields of the state space.
    Destination: linked satellite to the destination gateway index.

    we initialize the score of the satellites in 2 (worst case) because we do not know if they 
    will actually have a linked satellite in that direction.
    If they have it the satellite score will replace the initialization score (2) but if they dont 
    have it, as we need a score in order to set the state space we will give the worst score and
    send a None in the destinations dict. That action will be initialized with -infinite in the QTable
    '''
    destination  = getDestination(Block, g)
    state        = [2, 2, 2, 2, destination]   

    state[0] = getSatScore(satA, satA.QLearning.linkedSats['U'], g)
    state[1] = getSatScore(satA, satA.QLearning.linkedSats['D'], g)
    state[2] = getSatScore(satA, satA.QLearning.linkedSats['R'], g)
    state[3] = getSatScore(satA, satA.QLearning.linkedSats['L'], g)

    return state


def getBiasedLatitude(sat):
    try:
        return (int(math.degrees(sat.latitude))+latBias)/coordGran
    except AttributeError as e:
        # print(f"getBiasedLatitude Caught an exception: {e}")
        return notAvail


def getBiasedLongitude(sat):
    try:
        return (int(math.degrees(sat.longitude))+lonBias)/coordGran
    except AttributeError as e:
        # print(f"getBiasedLongitude Caught an exception: {e}")
        return notAvail


def getDeepStateReduced(block, sat, linkedSats):
    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None
    return np.array([getBiasedLatitude(linkedSats['U']),                        # Up link Positions
                    getBiasedLongitude(linkedSats['U']),
                    getBiasedLatitude(linkedSats['D']),                         # Down link Positions
                    getBiasedLongitude(linkedSats['D']),
                    getBiasedLatitude(linkedSats['R']),                         # Right link Positions
                    getBiasedLongitude(linkedSats['R']),
                    getBiasedLatitude(linkedSats['L']),                         # Left link Positions
                    getBiasedLongitude(linkedSats['L']),
                    getBiasedLatitude(sat),                                     # Actual Latitude
                    getBiasedLongitude(sat),                                    # Actual Longitude
                    getBiasedLatitude(satDest),                                 # Destination Latitude
                    getBiasedLongitude(satDest)]).reshape(1,-1)                 # Destination Longitude


def getDeepStateDiff(block, sat, linkedSats):
    def normalize_angle_diff(angle_diff):
        # Ensure the angle difference is within [-180, 180]
        return (angle_diff + 180) % 360 - 180

    def get_relative_position(neighbor_sat, current_coord, is_lat=True):
        # Convert and calculate relative position, considering the 180-degree discontinuity
        try:
            neighbor_coord = math.degrees(neighbor_sat.latitude if is_lat else neighbor_sat.longitude)
            current_coord = math.degrees(current_coord)
            diff = normalize_angle_diff(neighbor_coord - current_coord)
            return diff / coordGran
        except AttributeError:
            return notAvail
        
    def get_absolute_position(coord, bias, gran):
        # Convert absolute position to a normalized value within the specified range
        return (math.degrees(coord) + bias) / gran

    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None

    # Current coordinates
    current_lat = sat.latitude
    current_lon = sat.longitude

    # Queues
    queuesU = getQueues(linkedSats['U'], DDQN=True)
    queuesD = getQueues(linkedSats['D'], DDQN=True)
    queuesR = getQueues(linkedSats['R'], DDQN=True)
    queuesL = getQueues(linkedSats['L'], DDQN=True)

    state = [
        # Up link scores and positions
        getDeepSatScore(queuesU['U']),
        getDeepSatScore(queuesU['D']),
        getDeepSatScore(queuesU['R']),
        getDeepSatScore(queuesU['L']),
        get_relative_position(linkedSats['U'], current_lat, is_lat=True),
        get_relative_position(linkedSats['U'], current_lon, is_lat=False),

        # Down link scores and positions
        getDeepSatScore(queuesD['U']),
        getDeepSatScore(queuesD['D']),
        getDeepSatScore(queuesD['R']),
        getDeepSatScore(queuesD['L']),
        get_relative_position(linkedSats['D'], current_lat, is_lat=True),
        get_relative_position(linkedSats['D'], current_lon, is_lat=False),

        # Right link scores and positions
        getDeepSatScore(queuesR['U']),
        getDeepSatScore(queuesR['D']),
        getDeepSatScore(queuesR['R']),
        getDeepSatScore(queuesR['L']),
        get_relative_position(linkedSats['R'], current_lat, is_lat=True),
        get_relative_position(linkedSats['R'], current_lon, is_lat=False),

        # Left link scores and positions
        getDeepSatScore(queuesL['U']),
        getDeepSatScore(queuesL['D']),
        getDeepSatScore(queuesL['R']),
        getDeepSatScore(queuesL['L']),
        get_relative_position(linkedSats['L'], current_lat, is_lat=True),
        get_relative_position(linkedSats['L'], current_lon, is_lat=False),

        # Absolute current satellite's coordinates
        get_absolute_position(current_lat, latBias, coordGran),
        get_absolute_position(current_lon, lonBias, coordGran),

        # Destination's differential coordinates
        get_relative_position(satDest, current_lat, is_lat=True),
        get_relative_position(satDest, current_lon, is_lat=False)
    ]

    return np.array(state).reshape(1, -1)


def getDeepStateDiffLastHop(block, sat, linkedSats):
    def normalize_angle_diff(angle_diff):
        # Ensure the angle difference is within [-180, 180]
        return (angle_diff + 180) % 360 - 180

    def get_relative_position(neighbor_sat, current_coord, is_lat=True):
        # Convert and calculate relative position, considering the 180-degree discontinuity
        try:
            neighbor_coord = math.degrees(neighbor_sat.latitude if is_lat else neighbor_sat.longitude)
            current_coord = math.degrees(current_coord)
            diff = normalize_angle_diff(neighbor_coord - current_coord)
            return diff / coordGran
        except AttributeError:
            return notAvail
        
    def get_absolute_position(coord, bias, gran):
        # Convert absolute position to a normalized value within the specified range
        return (math.degrees(coord) + bias) / gran
    
    def get_last_satellite(block, sat): # REVIEW if index here are the same as decision index
        '''This will return information about the last block hop in relation to the current satellite:
        -1: Constellation moved and the last block's satellite is not connected to current satellite
        0: Upper neighbour
        1: Lower neighbour
        2: Right Neighbour
        3: Left  Neighbour'''
        actIndex = -1
        try:
            if len(block.QPath) > 2:
                if sat.upper and sat.upper.ID == block.QPath[-2][0]:
                    actIndex = 0
                elif sat.lower and sat.lower.ID == block.QPath[-2][0]:
                    actIndex = 1
                elif sat.right and sat.right.ID == block.QPath[-2][0]:
                    actIndex = 2
                elif sat.left and sat.left.ID == block.QPath[-2][0]:
                    actIndex = 3
            return actIndex
        except AttributeError as e:
            print(f'An error occurred when checking if {block.QPath[-2][0]} is a neighbour satellite of {sat.ID}')
            return actIndex

    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None

    # Current coordinates
    current_lat = sat.latitude
    current_lon = sat.longitude

    # Queues
    queuesU = getQueues(linkedSats['U'], DDQN=True)
    queuesD = getQueues(linkedSats['D'], DDQN=True)
    queuesR = getQueues(linkedSats['R'], DDQN=True)
    queuesL = getQueues(linkedSats['L'], DDQN=True)

    state = [
        # Previous satellite information
        get_last_satellite(block, sat),
        # Up link scores and positions
        getDeepSatScore(queuesU['U']),
        getDeepSatScore(queuesU['D']),
        getDeepSatScore(queuesU['R']),
        getDeepSatScore(queuesU['L']),
        get_relative_position(linkedSats['U'], current_lat, is_lat=True),
        get_relative_position(linkedSats['U'], current_lon, is_lat=False),

        # Down link scores and positions
        getDeepSatScore(queuesD['U']),
        getDeepSatScore(queuesD['D']),
        getDeepSatScore(queuesD['R']),
        getDeepSatScore(queuesD['L']),
        get_relative_position(linkedSats['D'], current_lat, is_lat=True),
        get_relative_position(linkedSats['D'], current_lon, is_lat=False),

        # Right link scores and positions
        getDeepSatScore(queuesR['U']),
        getDeepSatScore(queuesR['D']),
        getDeepSatScore(queuesR['R']),
        getDeepSatScore(queuesR['L']),
        get_relative_position(linkedSats['R'], current_lat, is_lat=True),
        get_relative_position(linkedSats['R'], current_lon, is_lat=False),

        # Left link scores and positions
        getDeepSatScore(queuesL['U']),
        getDeepSatScore(queuesL['D']),
        getDeepSatScore(queuesL['R']),
        getDeepSatScore(queuesL['L']),
        get_relative_position(linkedSats['L'], current_lat, is_lat=True),
        get_relative_position(linkedSats['L'], current_lon, is_lat=False),

        # Absolute current satellite's coordinates
        get_absolute_position(current_lat, latBias, coordGran),
        get_absolute_position(current_lon, lonBias, coordGran),

        # Destination's differential coordinates
        get_relative_position(satDest, current_lat, is_lat=True),
        get_relative_position(satDest, current_lon, is_lat=False)
    ]

    return np.array(state).reshape(1, -1)


def getDeepState(block, sat, linkedSats):
    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None

    queuesU = getQueues(linkedSats['U'], DDQN = True)
    queuesD = getQueues(linkedSats['D'], DDQN = True)
    queuesR = getQueues(linkedSats['R'], DDQN = True)
    queuesL = getQueues(linkedSats['L'], DDQN = True)
    return np.array([getDeepSatScore(queuesU['U']),                             # Up link scores
                    getDeepSatScore(queuesU['D']),
                    getDeepSatScore(queuesU['R']),
                    getDeepSatScore(queuesU['L']),
                    getBiasedLatitude(linkedSats['U']),                         # Up link Positions
                    getBiasedLongitude(linkedSats['U']),
                    getDeepSatScore(queuesD['U']),                              # Down link scores
                    getDeepSatScore(queuesD['D']),
                    getDeepSatScore(queuesD['R']),
                    getDeepSatScore(queuesD['L']),
                    getBiasedLatitude(linkedSats['D']),                         # Down link Positions
                    getBiasedLongitude(linkedSats['D']),
                    getDeepSatScore(queuesR['U']),                              # Right link scores
                    getDeepSatScore(queuesR['D']),
                    getDeepSatScore(queuesR['R']),
                    getDeepSatScore(queuesR['L']),
                    getBiasedLatitude(linkedSats['R']),                         # Right link Positions
                    getBiasedLongitude(linkedSats['R']),
                    getDeepSatScore(queuesL['U']),                              # Left link scores
                    getDeepSatScore(queuesL['D']),
                    getDeepSatScore(queuesL['R']),
                    getDeepSatScore(queuesL['L']),
                    getBiasedLatitude(linkedSats['L']),                         # Left link Positions
                    getBiasedLongitude(linkedSats['L']),

                    # int(math.degrees(sat.latitude))+latBias,                    # Actual Latitude
                    # int(math.degrees(sat.longitude))+lonBias,                   # Actual Longitude
                    # int(math.degrees(satDest.latitude))+latBias,                # Destination Latitude
                    # int(math.degrees(satDest.longitude))+lonBias]).reshape(1,-1)# Destination Longitude

                    getBiasedLatitude(sat),                                     # Actual Latitude
                    getBiasedLongitude(sat),                                    # Actual Longitude
                    getBiasedLatitude(satDest),                                 # Destination Latitude
                    getBiasedLongitude(satDest)]).reshape(1,-1)                 # Destination Longitude
    

def createQTable(NGT):
    '''
    Create a 6D numpy array to hold the current Q-values for each state and action pair: Q(s, a)
    The array contains 5 dimensions with the shape of the environment, as well as a 6th "action" dimension.
    The "action" dimension consists of 4 layers that will allow us to keep track of the Q-values for each possible action in each state
    The value of each (state, action) pair is initialized to 0.
    '''

    actions = ('N', 'S', 'E', 'W')
    satUp, satDown, satRight, satLeft = 3, 3, 3, 3
    Destination = NGT

    qValues = np.zeros((satUp, satDown, satRight, satLeft, Destination, len(actions)))  # first 5 fields are states while 6th field is the action. 4050 values with 10 GTs

    return qValues


###############################################################################
##########################   Q-Learning - Rewards    ##########################
###############################################################################


# @profile
def getSlantRange(satA, satB):
    '''
    given 2 satellites, it will return the slant range between them (With the method used at 'get_slant_range_optimized')
    '''
    return np.linalg.norm(np.array((satA.x, satA.y, satA.z)) - np.array((satB.x, satB.y, satB.z)))  # posA - posB


# @profile
def getQueueReward(queueTime, w1):
    '''
    Given the queue time in seconds, this function will return the queue reward.
    With 125 packets, 9ms Queue (The thershold that we take to consider a queue as high) the reward will be -0.04 (with w1 = 2)
    '''
    return w1*(1-10**queueTime)


# @profile
def getDistanceReward(satA, satB, destination, w2):
    '''
    This function will return the instant reward regarding to the slant range reduction from actual node to destination
    just after the agent takes an action (destination is the satellite linked to the destination Gateway)

    TSLa: Total slant range from sat A to destination
    TSLb: Total slant range from sat B to destination
    SLR : Slant Range reduction after taking the action (Going from satA to satB)

    Formula: w*(SLR + TSLa)/TSLa = w*(TSLa - TSLb + TSLa)/TSLa = w*(2*TSLa - TSLb)/TSLa
    '''
    balance   = -1      # centralizes the result in 0

    TSLa = getSlantRange(satA, destination)
    TSLb = getSlantRange(satB, destination)
    return w2*((2*TSLa-TSLb)/TSLa + balance)


def getDistanceRewardV2(sat, nextSat, satU, satD, satR, satL, destination, w2):
    '''
    Computes the reward by comparing how closer you get to the destination in terms of KM (SLr, Slant Range Reduction) with the
    average distance with all your neighbours (SLav, Slant Range average)
    If any of the linked satellites is not available, it is handled
    SLr/SLav + balance
    '''

    SLr = getSlantRange(sat, destination) - getSlantRange(nextSat, destination)
    SLU = SLD = SLR = SLL = 0
    count = 0

    # Calculate slant range for each satellite, if it is not None
    if satU is not None:
        SLU = getSlantRange(satU, sat)
        count += 1
    if satD is not None:
        SLD = getSlantRange(satD, sat)
        count += 1
    if satR is not None:
        SLR = getSlantRange(satR, sat)
        count += 1
    if satL is not None:
        SLL = getSlantRange(satL, sat)
        count += 1

    SLav = (SLU + SLD + SLR + SLL) / count if count > 0 else 0

    return w2 * (SLr / SLav) if SLav != 0 else 0


def getDistanceRewardV3(sat, nextSat, satU, satD, satR, satL, destination, w2):
    '''
    Returns the distance reward computed by comparing how closer you get to the destination in terms of KM (SLr, Slant Range Reduction) with
    how close you could get as maximum taking the other options going to any of the other neighbours (max(SLrs), max(Slant range reductions from all the neighbours))
    reward = SLr/max(SLs)
    '''
    SLr = getSlantRange(sat, destination) - getSlantRange(nextSat, destination)
    SLrs= []

    if satU is not None:
        SLrs.append(getSlantRange(sat, destination) - getSlantRange(satU, destination))
    if satD is not None:
        SLrs.append(getSlantRange(sat, destination) - getSlantRange(satD, destination))
    if satR is not None:
        SLrs.append(getSlantRange(sat, destination) - getSlantRange(satR, destination))
    if satL is not None:
        SLrs.append(getSlantRange(sat, destination) - getSlantRange(satL, destination))

    return w2*SLr/max(SLrs)
    

def getDistanceRewardV4(sat, nextSat, satDest, w2, w4):
    global biggestDist
    SLr = getSlantRange(sat, satDest) - getSlantRange(nextSat, satDest)
    TravelDistance = getSlantRange(sat, nextSat)
    if TravelDistance > biggestDist:
        # print(f'Very big distance: {sat.ID}, {nextSat.ID}')
        pass
    return w2*(SLr-TravelDistance/w4)/biggestDist
    # return w2*(SLr/biggestDist)
    # return w2*SLr/1000000


def getDistanceRewardV5(sat, nextSat, w2):
    SLr = getSlantRange(sat, nextSat)
    return w2*SLr/1000000


def saveHyperparams(outputPath, inputParams, hyperparams):
    print('Saving hyperparams at: ' + str(outputPath))
    hyperparams = ['Constellation: ' + str(inputParams['Constellation'][0]),
                'Import QTables: ' + str(hyperparams.importQ),
                'plotPath: ' + str(hyperparams.plotPath),
                'Test length: ' + str(inputParams['Test length'][0]),
                'Alphas: ' + str(hyperparams.alpha) + ', ' + str(alpha_dnn),
                'Gamma: ' + str(hyperparams.gamma),
                'Epsilon: ' + str(hyperparams.epsilon), 
                'Max epsilon: ' + str(hyperparams.MAX_EPSILON), 
                'Min epsilon: ' + str(hyperparams.MIN_EPSILON), 
                'Arrive Reward: ' + str(hyperparams.ArriveR), 
                'w1: ' + str(hyperparams.w1), 
                'w2: ' + str(hyperparams.w2),
                'w4: ' + str(hyperparams.w4),
                'againPenalty: ' + str(hyperparams.again),
                'unavPenalty: ' + str(hyperparams.unav),
                'Coords granularity: ' + str(hyperparams.coordGran),
                'Update freq: ' + str(hyperparams.updateF),
                'Batch Size: ' + str(hyperparams.batchSize),
                'Buffer Size: ' + str(hyperparams.bufferSize),
                'Hard Update: ' + str(hyperparams.hardUpdate),
                'Exploration: ' + str(hyperparams.explore),
                'DDQN: ' + str(hyperparams.ddqn),
                'Latitude bias: ' + str(hyperparams.latBias),
                'Longitude bias: ' + str(hyperparams.lonBias),
                'Diff: ' + str(hyperparams.diff),
                'Reduced State: ' + str(hyperparams.reducedState),
                'Online phase: ' + str(hyperparams.online)]

    # save hyperparams
    with open(outputPath + 'hyperparams.txt', 'w') as f:
        for param in hyperparams:
            f.write(param + '\n')


def saveQTables(outputPath, earth):
    print('Saving Q-Tables at: ' + outputPath)
    # create output path if it does not exist
    path = outputPath + 'qTablesExport_' + str(len(earth.gateways)) + 'GTs/'
    os.makedirs(path, exist_ok=True) 

    # save Q-Tables
    for plane in earth.LEO:
        for sat in plane.sats:
            qTable = sat.QLearning.qTable
            with open(path + sat.ID + '.npy', 'wb') as f:
                np.save(f, qTable)


def saveDeepNetworks(outputPath, earth):
    print('Saving Deep Neural networks at: ' + outputPath)
    os.makedirs(outputPath, exist_ok=True) 
    if not onlinePhase:
        earth.DDQNA.qNetwork.save(outputPath + 'qNetwork_'+ str(len(earth.gateways)) + 'GTs' + '.h5')
        if ddqn:
            earth.DDQNA.qTarget.save(outputPath + 'qTarget_'+ str(len(earth.gateways)) + 'GTs' + '.h5')
    else:
        for plane in earth.LEO:
            for sat in plane.sats:
                sat.DDQNA.qNetwork.save(outputPath + sat.ID + 'qNetwork_'+ str(len(earth.gateways)) + 'GTs' + '.h5')
                if ddqn:
                    sat.DDQNA.qTarget.save(outputPath + sat.ID + 'qTarget_'+ str(len(earth.gateways)) + 'GTs' + '.h5')


###############################################################################
#########################    Simulation && Results    #########################
###############################################################################


def plotLatenciesBars(percentages, outputPath):
    '''
    Bar plot where each bar is a scenario with a different nº of gateways and where each color represents one of the three latencies.
    '''
    # plot percent stacked barplot
    barWidth= 0.85
    r       = percentages['GTnumber']
    numbers = percentages['GTnumber']
    GTnumber= len(r)

    plt.bar(r, percentages['Propagation time'], color='#b5ffb9', edgecolor='white', width=barWidth, label="Propagation time")   # Propagation time
    plt.bar(r, percentages['Queue time'], bottom=percentages['Propagation time'], color='#f9bc86',                              # Queue time
             edgecolor='white', width=barWidth, label="Queue time")
    plt.bar(r, percentages['Transmission time'], bottom=[i+j for i,j in zip(percentages['Propagation time'],                    # Tx time
            percentages['Queue time'])], color='#a3acff', edgecolor='white', width=barWidth, label="Transmission time")

    # Custom x axis
    plt.xticks(numbers)
    plt.xlabel("Nº of gateways")
    plt.ylabel('Latency')

    # Add a legend
    plt.legend(loc='lower left')

    # Show and save graphic
    plt.savefig(outputPath + 'Percentages_{}_Gateways.png'.format(GTnumber+1))
    plt.close()
    # plt.show()


def plotQueues(queues, outputPath, GTnumber):
    '''
    Will plot the cumulative distribution function (CDF) and probability density function (PDF) of all the queues that each package has faced.
    ''' 
    os.makedirs(outputPath + '/pngQueues/', exist_ok=True) # create output path
    plt.hist(queues, bins=max(queues), cumulative=True, density = True, label='CDF DATA', histtype='step', alpha=0.55, color='blue')
    plt.xlabel('Queue length')
    plt.legend(loc = 'lower left')
    plt.savefig(outputPath + '/pngQueues/' + 'Queues_{}_Gateways.png'.format(GTnumber))
    plt.close()
    d = pd.DataFrame(queues)
    d.to_csv(outputPath + '/csv/' + "Queues_{}_Gateways.csv".format(GTnumber), index = False)


def extract_block_index(block_id):
    return int(block_id.split('_')[-1])


def save_plot_rewards(outputPath, reward, GTnumber, window_size=200):
    rewards = [x[0] for x in reward]
    times   = [x[1] for x in reward]
    data    = pd.DataFrame({'Rewards': rewards, 'Time': times})

    # Smoothed Rewards
    data['Smoothed Rewards'] = data['Rewards'].rolling(window=window_size).mean()

    # Top 10% and Bottom 10% Rewards
    data['Top 10% Avg Rewards'] = data['Rewards'].rolling(window=window_size).apply(lambda x: np.mean(np.partition(x, -int(len(x)*0.1))[-int(len(x)*0.1):]), raw=True)
    data['Bottom 10% Avg Rewards'] = data['Rewards'].rolling(window=window_size).apply(lambda x: np.mean(np.partition(x, int(len(x)*0.1))[:int(len(x)*0.1)]), raw=True)

    # Plotting
    plt.figure(figsize=(8, 4))
    line1, = plt.plot(data['Time'], data['Top 10% Avg Rewards'], color='skyblue', linewidth=2, label='Top 10% reward')
    line2, = plt.plot(data['Time'], data['Smoothed Rewards'], color='blue', linewidth=2, label='Average reward')
    line3, = plt.plot(data['Time'], data['Bottom 10% Avg Rewards'], color='navy', linewidth=2, label='Bottom 10% reward')

    plt.legend(handles=[line1, line2, line3], fontsize=15, loc='upper right')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Time [ms]", fontsize=15)
    plt.ylabel("Average rewards", fontsize=15)
    plt.grid(True)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

    # Save plot
    rewards_dir = os.path.join(outputPath, 'Rewards')
    plt.tight_layout()
    os.makedirs(rewards_dir, exist_ok=True)  # create output path
    plt.savefig(os.path.join(rewards_dir, "rewards_{}_gateways.png".format(GTnumber)))#, bbox_inches='tight')
    plt.close()

    # Save CSV
    csv_dir = os.path.join(outputPath, 'csv')
    os.makedirs(csv_dir, exist_ok=True)  # create output path
    data.to_csv(os.path.join(csv_dir, "rewards_{}_gateways.csv".format(GTnumber)), index=False)

    return data


def save_epsilons(outputPath, eps, GTnumber):
    epsilons = [x[0] for x in eps]
    times    = [x[1] for x in eps]
    plt.plot(times, epsilons)
    plt.title("Epsilon over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Epsilon")
    os.makedirs(outputPath + '/epsilons/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/epsilons/' + "epsilon_{}_gateways.png".format(GTnumber))
    plt.close()

    data = {'epsilon': [e for e in epsilons], 'time': [t for t in times]}
    df = pd.DataFrame(data)
    os.makedirs(outputPath + '/csv/' , exist_ok=True) # create output path
    df.to_csv(outputPath + '/csv/' + "epsilons_{}_gateways.csv".format(GTnumber), index=False)

    return df
    

def save_training_counts(outputPath, train_times, GTnumber):
    # Extract times
    times = [x[0]*1000 for x in train_times]

    # Calculate cumulative trainings over time
    training_counts = list(range(1, len(times) + 1))

    # Plotting the cumulative number of trainings
    plt.plot(times, training_counts)
    plt.title("Cumulative trainings over time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of Trainings")

    # Create output path and save the figure
    os.makedirs(outputPath + '/trainings/', exist_ok=True)
    plt.savefig(outputPath + '/trainings/' + "trainings_{}_gateways.png".format(GTnumber))
    plt.close()

    # Prepare data for saving
    data = {'time': times, 'trainings': training_counts}
    df = pd.DataFrame(data)

    # Create CSV output path and save data
    os.makedirs(outputPath + '/csv/', exist_ok=True)
    df.to_csv(outputPath + '/csv/' + "trainings_{}_gateways.csv".format(GTnumber), index=False)

    # return df


def save_losses(outputPath, earth1, GTnumber):
    losses = [x[0] for x in earth1.loss]
    times  = [x[1] for x in earth1.loss]
    plt.plot(times, losses)
    plt.xlabel("Time (s)")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    os.makedirs(outputPath + '/loss/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/loss/' + "loss_{}_gatewaysTime.png".format(GTnumber))
    plt.close()

    data = {'loss': [l for l in losses], 'time': [t for t in times]}
    df = pd.DataFrame(data)
    df.to_csv(outputPath + '/csv/' + "loss_{}_gateways.csv".format(GTnumber), index=False)
    os.makedirs(outputPath + '/loss/', exist_ok=True) # create output path

    xs = [l for l in range(len(losses))]
    plt.plot(xs, losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over Steps")
    plt.savefig(outputPath + '/loss/' + "loss_{}_gatewaysSteps.png".format(GTnumber))
    plt.close()

    # save losses average
    plt.plot(range(len(earth1.lossAv)), earth1.lossAv)
    plt.xlabel("Steps")
    plt.ylabel("Loss average")
    plt.title("Loss average over Steps")
    os.makedirs(outputPath + '/loss/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/loss/' + "loss_{}_gatewaysAverage.png".format(GTnumber))
    plt.close()


def plotSavePathLatencies(outputPath, GTnumber, pathBlocks):
    # figure of latencies between two first gateways
    latency = []
    arrival = []
    for item in pathBlocks[0]:
        latency.append(item[0])
        arrival.append(item[1])
    plt.scatter(arrival, latency, c='r')
    plt.xlabel("Time")
    plt.ylabel("Latency")
    os.makedirs(outputPath + '/pngLatencies/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/pngLatencies/' + '{}_gatewaysTime.png'.format(GTnumber))
    plt.close()

    # x axis is the number of the arrival, not the time
    xs = [l for l in range(len(latency))]
    plt.figure()
    plt.scatter(xs,latency, c='r')
    plt.xlabel("Arrival index")
    plt.ylabel('Latency')
    plt.savefig(outputPath + '/pngLatencies/' + '{}_gateways.png'.format(GTnumber))
    plt.close()

    # Save latencies
    os.makedirs(outputPath + '/csv/', exist_ok=True) # create output path
    data = {'Latency': [l for l in latency], 'Arrival Time': [t for t in arrival]}
    df = pd.DataFrame(data)
    df.to_csv(outputPath + '/csv/' + "pathLatencies_{}_gateways.csv".format(GTnumber), index=False)
    # os.makedirs(outputPath + '/loss/', exist_ok=True) # create output path


def plot_packet_latencies_and_uplink_downlink_throughput(data, outputPath, bins_num=30, save=False, plot_separately=True):
    """
    Generate either separate scatter plots of packet latencies for each path (source-destination),
    or a single plot combining all paths. Overlay line plots of uplink and downlink throughput on 
    a secondary y-axis, with a single legend for all items in the upper right.
    """

    save_dir = os.path.join(outputPath, 'Throughput')
    os.makedirs(save_dir, exist_ok=True)

    # Group blocks by (source, destination) paths
    paths_data = defaultdict(list)
    for block in data:
        src = block.path[0][0]        # Source
        dst = block.path[-1][0]       # Destination
        paths_data[(src, dst)].append(block)

    # Function to plot data for a single path or combined
    def plot_path_data(blocks, src=None, dst=None):
        fig, ax1 = plt.subplots(figsize=(8, 4))
        
        # Sort blocks by creation time
        blocks = sorted(blocks, key=lambda b: b.creationTime)
        
        # Extract times and latencies (converted to ms)
        creation_times = np.array([block.creationTime for block in blocks]) * 1000  # ms
        arrival_times = np.array([block.creationTime + block.totLatency for block in blocks]) * 1000  # ms
        latencies = np.array([block.totLatency * 1000 for block in blocks])  # ms

        # Scatter plot for packet arrival times vs latency
        arrival_scatter = ax1.scatter(arrival_times, latencies, color='#1E90FF', label='Packet Delivery', alpha=0.6, s=10)
        
        # Configure primary y-axis for latency
        ax1.set_xlabel('Time [ms]', fontsize=16)
        ax1.set_ylabel('Average E2E Latency [ms]', fontsize=16)
        
        # Create secondary y-axis for throughput
        ax2 = ax1.twinx()
        time_bins = np.linspace(min(creation_times), max(arrival_times), num=bins_num)
        
        # Calculate throughput
        uplink_counts, _ = np.histogram(creation_times, bins=time_bins)
        uplink_throughput = (uplink_counts * BLOCK_SIZE / 1e3) / np.diff(time_bins)  # Mbps
        downlink_counts, _ = np.histogram(arrival_times, bins=time_bins)
        downlink_throughput = (downlink_counts * BLOCK_SIZE / 1e3) / np.diff(time_bins)  # Mbps

        # Plot throughput on secondary y-axis
        uplink_line, = ax2.plot(time_bins[:-1], uplink_throughput, color='#00008B', lw=2, label='Uplink Throughput')
        downlink_line, = ax2.plot(time_bins[:-1], downlink_throughput, color='#1E90FF', lw=2, label='Downlink Throughput')
        
        # Configure secondary y-axis for throughput
        ax2.set_ylabel('Throughput [Mbps]', fontsize=16)
        
        # Combine legends
        handles = [arrival_scatter, uplink_line, downlink_line]
        labels = [handle.get_label() for handle in handles]
        ax1.legend(handles, labels, loc='upper center', fontsize=12)

        # Display grid and layout adjustments
        ax1.grid(True)
        ax2.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout()
        
        # Save or show plot
        if save:
            filename = f'{src}_{dst}_path_latency_throughput.png' if src and dst else 'combined_path_latency_throughput.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
        else:
            plt.show()
        plt.close()

    # Plot all paths together or separately based on flag
    if plot_separately:
        for (src, dst), blocks in paths_data.items():
            plot_path_data(blocks, src, dst)
    else:
        all_blocks = [block for blocks in paths_data.values() for block in blocks]
        plot_path_data(all_blocks)


def plot_throughput_cdf(data, outputPath, bins_num=100, save=False, plot_separately=True):
    """
    Generate and save a CDF plot of the throughput. Either plot each route separately or
    combine all routes into a single plot based on the `plot_separately` flag.
    """
    save_dir = os.path.join(outputPath, 'Throughput')
    os.makedirs(save_dir, exist_ok=True)

    # Group blocks by (source, destination) paths
    paths_data = defaultdict(list)
    for block in data:
        src = block.path[0][0]  # Source
        dst = block.path[-1][0]  # Destination
        paths_data[(src, dst)].append(block)

    # Helper function to plot CDF for a given set of blocks
    def plot_cdf_for_path(blocks, src=None, dst=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Sort blocks by creation time
        blocks = sorted(blocks, key=lambda b: b.creationTime)
        
        # Extract creation times and arrival times
        creation_times = np.array([block.creationTime for block in blocks])
        arrival_times = np.array([block.creationTime + block.totLatency for block in blocks])
        
        # Define time bins and calculate throughput
        time_bins = np.linspace(min(creation_times), max(arrival_times), num=bins_num)
        uplink_counts, _ = np.histogram(creation_times, bins=time_bins)
        uplink_throughput = (uplink_counts * BLOCK_SIZE / 1e6) / np.diff(time_bins)  # Mbps
        downlink_counts, _ = np.histogram(arrival_times, bins=time_bins)
        downlink_throughput = (downlink_counts * BLOCK_SIZE / 1e6) / np.diff(time_bins)  # Mbps
        
        # Sort and calculate CDF
        uplink_throughput_sorted = np.sort(uplink_throughput)
        downlink_throughput_sorted = np.sort(downlink_throughput)
        uplink_cdf = np.arange(1, len(uplink_throughput_sorted) + 1) / len(uplink_throughput_sorted)
        downlink_cdf = np.arange(1, len(downlink_throughput_sorted) + 1) / len(downlink_throughput_sorted)
        
        # Plot CDFs
        ax.plot(uplink_throughput_sorted, uplink_cdf, label='Uplink Throughput', color='#00008B', lw=2)
        ax.plot(downlink_throughput_sorted, downlink_cdf, label='Downlink Throughput', color='#1E90FF', lw=2)
        
        # Configure plot
        ax.set_xlabel('Throughput [Mbps]', fontsize=16)
        ax.set_ylabel('CDF', fontsize=16)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Adjust layout, save plot, and close
        plt.tight_layout()
        if save:
            filename = f'Throughput_CDF_{src}_to_{dst}.png' if src and dst else 'Throughput_CDF_All_Paths.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
        else:
            plt.show()
        plt.close()

    # Plot each path separately or all paths combined based on flag
    if plot_separately:
        for (src, dst), blocks in paths_data.items():
            plot_cdf_for_path(blocks, src, dst)
    else:
        all_blocks = [block for blocks in paths_data.values() for block in blocks]
        plot_cdf_for_path(all_blocks)


def plotSaveAllLatencies(outputPath, GTnumber, allLatencies, epsDF=None, annotate_min_latency=True):  
    # preprocess and setup
    GTnumber_Max = 4 # max number of gts for displaying the legend. If the number of GTs is bigger than this, then no legend is displayed
    sns.set(font_scale=1.5)
    window_size = winSize
    marker_size = markerSize
    df = pd.DataFrame(allLatencies, columns=['Creation Time', 'Latency', 'Arrival Time', 'Source', 
                                             'Destination', 'Block ID', 'QueueTime', 'TxTime', 'PropTime'])
    df['Block Index'] = df['Block ID'].apply(extract_block_index)
    df = df.sort_values(by=['Source', 'Destination', 'Block Index'])
    df.to_csv(outputPath + '/csv/' + "allLatencies_{}_gateways.csv".format(GTnumber))

    # Convert time values to milliseconds
    df['Creation Time'] *= 1000
    df['Arrival Time']  *= 1000
    df['Latency']       *= 1000
    if epsDF is not None:
        epsDF['time']   *= 1000

    # Calculate the rolling average for each unique path
    df['Path'] = df['Source'].astype(str) + ' -> ' + df['Destination'].astype(str)
    df['Latency_Rolling_Avg'] = df.groupby('Path')['Latency'].transform(lambda x: x.rolling(window=window_size).mean())
    
    # Metrics for x-axis
    metrics = ['Arrival Time', 'Creation Time']

    # Create subplots
    fig, axes = plt.subplots(len(metrics), 2, figsize=(18, 18))

    for i, metric in enumerate(metrics):
        # Line Plots on the left (column index 0)
        lineplot = sns.lineplot(x=metric, y='Latency_Rolling_Avg', hue='Path', ax=axes[i, 0], data=df)
        axes[i, 0].set_title(f'Latency Trends Over {metric} (Window Size = {window_size})')
        axes[i, 0].set_xlabel(metric + ' (ms)')
        axes[i, 0].set_ylabel('Average Latency (ms)')

        # Annotate minimum latency for Creation Time only
        if annotate_min_latency and metric == 'Creation Time':
            unique_paths = df['Path'].unique()
            for path in unique_paths:
                df_path = df[df['Path'] == path]
                min_latency = df_path['Latency_Rolling_Avg'].min()
                try:
                    min_pos = df_path[metric][df_path['Latency_Rolling_Avg'].idxmin()]
                    axes[i, 0].annotate(f'{min_latency:.0f} ms', xy=(min_pos, min_latency), 
                                        xytext=(-50, 30), textcoords='offset points', 
                                        arrowprops=dict(arrowstyle='->', color='black'))
                except KeyError as e:
                    print(f"Error annotating minimum latency for the path path {path}: {e}")


        # Scatter Plots on the right (column index 1)
        scatterplot = sns.scatterplot(x=metric, y='Latency', hue='Path', ax=axes[i, 1], data=df, marker='o', s=marker_size)
        axes[i, 1].set_title(f'Individual Latency Points Over {metric}')
        axes[i, 1].set_xlabel(metric)
        axes[i, 1].set_ylabel('Latency')

        # Create a twin y-axis for epsilon data if epsDF is not None
        if epsDF is not None:
            ax2 = axes[i, 0].twinx()
            line3 = sns.lineplot(x='time', y='epsilon', data=epsDF, color='purple', label='Epsilon', ax=ax2)

            # Merge legends
            handles1, labels1 = axes[i, 0].get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            axes[i, 0].legend(handles1 + handles2, labels1 + labels2, loc='upper right')
            # Check if ax2 has a legend before trying to remove it
            if ax2.get_legend():
                ax2.get_legend().remove()
        else:
            # Handle legend for the case when epsDF is None
            handles, labels = axes[i, 0].get_legend_handles_labels()
            axes[i, 0].legend(handles, labels, loc='upper right')

        if GTnumber > GTnumber_Max:
            # Disable legends on both subplots
            if axes[i, 0].get_legend():
                axes[i, 0].get_legend().set_visible(False)
            if axes[i, 1].get_legend():
                axes[i, 1].get_legend().set_visible(False)

        
    # Adjust the layout
    plt.tight_layout()
    os.makedirs(outputPath + '/pngAllLatencies/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/pngAllLatencies/' + '{}_gateways_All_Latencies_subplots.png'.format(GTnumber), dpi = 300)
    plt.close()
    sns.set()


def plotRatesFigures():
    values = [upGSLRates, downGSLRates, interRates, intraRate]

    plt.figure()
    plt.hist(np.asarray(interRates)/1e9, cumulative=1, histtype='step', density=True)
    plt.title('CDF - Inter plane ISL data rates')
    plt.ylabel('Empirical CDF')
    plt.xlabel('Data rate [Gbps]')
    plt.show()
    plt.close()

    plt.figure()
    plt.hist(np.asarray(upGSLRates)/1e9, cumulative=1, histtype='step', density=True)
    plt.title('CDF - Uplink data rates')
    plt.ylabel('Empirical CDF')
    plt.xlabel('Data rate [Gbps]')
    plt.show()
    plt.close()

    plt.figure()
    plt.hist(np.asarray(downGSLRates)/1e9, cumulative=1, histtype='step', density=True)
    plt.title('CDF - Downlink data rates')
    plt.ylabel('Empirical CDF')
    plt.xlabel('Data rate [Gbps]')
    plt.show()
    plt.close()


def plotCongestionMap(self, paths, outPath, GTnumber, plot_separately=True):
    def extract_gateways(path):
    # Assuming QPath's first and last elements contain gateway identifiers
        if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning':
            return path.QPath[0][0], path.QPath[-1][0]
        else:
            return path.path[0][0], path.path[-1][0]
        
    os.makedirs(outPath, exist_ok=True)

    # Identify unique routes and filter by packet threshold (100 packets)
    unique_routes = {}
    for block in paths:
        p = block.QPath if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning' else block.path
        if p:  # Ensure QPath or path is not empty
            gateways = extract_gateways(block)
            if gateways in unique_routes:
                unique_routes[gateways] += 1
            else:
                unique_routes[gateways] = 1

    filtered_routes = {route: count for route, count in unique_routes.items() if count > 100} # REVIEW Packet threshold for path visualization 500

    # Plot for all routes combined
    if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning':
        all_routes_paths = [block for block in paths if block.QPath and extract_gateways(block) in filtered_routes]
    else:
        all_routes_paths = [block for block in paths if block.path and extract_gateways(block) in filtered_routes]

    done = self.plotMap(plotGT=True, plotSat=True, edges=False, save=True, paths=np.asarray(all_routes_paths),
                 fileName=os.path.join(outPath, f"all_routes_CongestionMap_{GTnumber}GTs.png"))
    plt.close()
    if done == -1:
        print('Congestion map for all routes not available')

    # Plot for each unique route above the threshold
    if plot_separately:
        for route, count in filtered_routes.items():
            if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning':
                route_paths = [block for block in paths if extract_gateways(block) == route and block.QPath]
            else:
                route_paths = [block for block in paths if extract_gateways(block) == route and block.path]

            done = self.plotMap(plotGT=True, plotSat=True, edges=False, save=True, paths=np.asarray(route_paths),
                        fileName=os.path.join(outPath, f"CongestionMap_{route[0]}_to_{route[1]}_{GTnumber}GTs.png"))
            plt.close()
            if done == -1:
                print(f'Congestion map for {route} not available')


# @profile
def RunSimulation(GTs, inputPath, outputPath, populationData, radioKM):
    start_time = datetime.now()
    '''
    this is required for the bar plot at the end of the simulation
    percentages = {'Queue time': [],
                'Propagation time': [],
                'Transmission time': [],
                'GTnumber' : []}
    '''
    inputParams = pd.read_csv(inputPath + "inputRL.csv")

    locations = inputParams['Locations'].copy()
    print('Nº of Gateways: ' + str(len(locations)))

    # pathing     = inputParams['Pathing'][0]
    testType    = inputParams['Test type'][0]
    testLength  = inputParams['Test length'][0]
    # numberOfMovements = 0

    print('Routing metric: ' + pathing)

    simulationTimelimit = testLength if testType != "Rates" else movementTime * testLength + 10

    firstGT = True
    for GTnumber in GTs:
        global CurrentGTnumber
        global Train
        global TrainThis
        global nnpath
        if FL_Test:
            global CKA_Values
        if ddqn:
            global nnpathTarget
        TrainThis       = Train
        CurrentGTnumber = GTnumber
        
        if firstGT:
            # nnpath  = f'./pre_trained_NNs/qNetwork_1GTs.h5'   # Already set
            firstGT = False
        else:
            nnpath  = f'{outputPath}/NNs/qNetwork_{GTnumber-1}GTs.h5'
            if ddqn:
                nnpathTarget = f'{outputPath}/NNs/qTarget_{GTnumber-1}GTs.h5'

        if len(GTs)>1:
            start_time_GT = datetime.now()

        env = simpy.Environment()

        if mixLocs: # changes the selected GTs every iteration
            firstLocs = locations[:max(GTs)]
            random.shuffle(firstLocs)
            locations[:max(GTs)] = firstLocs
            # random.shuffle(locations)
        inputParams['Locations'] = locations[:GTnumber]
        print('----------------------------------')
        print('Time:')
        print(datetime.now().strftime("%H:%M:%S"))
        print('Locations:')
        print(inputParams['Locations'][:GTnumber])
        print(f'Movement Time: {movementTime}')
        print(f'Rotation Factor: {ndeltas}')
        print(f'Minimum epsilon: {MIN_EPSILON}')
        print(f'Reward for deliver: {ArriveReward}')
        print(f'Stop Loss: {stopLoss}, number of samples considered: {nLosses}, threshold: {lThreshold}')
        print('----------------------------------')
        earth1, _, _, _ = initialize(env, populationData, inputPath + 'Gateways.csv', radioKM, inputParams, movementTime, locations, outputPath, matching=matching)
        earth1.outputPath = outputPath
        
        print('Saving ISLs map...')
        islpath = outputPath + '/ISL_maps/'
        os.makedirs(islpath, exist_ok=True) 
        earth1.plotMap(plotGT = True, plotSat = True, edges=True, save = True, outputPath=islpath, n=earth1.nMovs)
        plt.close()
        print('----------------------------------')

        progress = env.process(simProgress(simulationTimelimit, env))
        startTime = time.time()
        env.run(simulationTimelimit)
        timeToSim = time.time() - startTime

        if testType == "Rates":
            plotRatesFigures()
        else:
            results, allLatencies, pathBlocks, blocks = getBlockTransmissionStats(timeToSim, inputParams['Locations'], inputParams['Constellation'][0], earth1)
            print(f'DataBlocks lost: {earth1.lostBlocks}')
            
            # save & plot ftirst 2 GTs path latencies
            plotSavePathLatencies(outputPath, GTnumber, pathBlocks)

            # Throughput figures
            print('Plotting Throughput...')
            plot_packet_latencies_and_uplink_downlink_throughput(blocks, outputPath, bins_num=30, save = True, plot_separately = plotAllThro)
            plot_throughput_cdf(blocks, outputPath, bins_num = 100, save = True, plot_separately = plotAllThro)
            
            if pathing == "Deep Q-Learning" or pathing == 'Q-Learning':
                save_plot_rewards(outputPath, earth1.rewards, GTnumber)
                if not onlinePhase:
                    eps = earth1.DDQNA.epsilon if pathing == "Deep Q-Learning" else earth1.epsilon
                else:
                    eps = earth1.LEO[0].sats[0].DDQNA.epsilon if pathing == "Deep Q-Learning" else earth1.epsilon
                # save epsilons
                if Train:
                    epsDF = save_epsilons(outputPath, eps, GTnumber)
                    save_training_counts(outputPath, earth1.trains, GTnumber)
                else:
                    epsDF = None

                # save & plot all paths latencies
                print('Plotting latencies...')
                plotSaveAllLatencies(outputPath, GTnumber, allLatencies, epsDF)
            
            if pathing == "Deep Q-Learning":
                # save losses
                save_losses(outputPath, earth1, GTnumber)
                if FL_Test and const_moved:
                    print('Plotting CKA values...')
                    plot_cka_over_time(earth1.CKA, outputPath, GTnumber)
                
            else:
                print('Plotting latencies...')
                plotSaveAllLatencies(outputPath, GTnumber, allLatencies)

        plotShortestPath(earth1, pathBlocks[1][-1].path, outputPath)
        if not onlinePhase:
            plotQueues(earth1.queues, outputPath, GTnumber)

        print('Plotting link congestion figures...')
        plotCongestionMap(earth1, np.asarray(blocks), outputPath + '/Congestion_Test/', GTnumber, plot_separately=plotAllCon)

        print(f"number of gateways: {GTnumber}")
        print('Path:')
        print(pathBlocks[1][-1].path)
        print('Bottleneck:')
        print(findBottleneck(pathBlocks[1][-1].path, earth1))

        '''
        # add data for percentages bar plot
        # percentages['Queue time']       .append(results.meanQueueLatency)
        # percentages['Propagation time'] .append(results.meanPropLatency)
        # percentages['Transmission time'].append(results.meanTransLatency)
        # percentages['GTnumber']         .append(GTnumber)

        save congestion test data
        print('Saving congestion test data...')
        blocks = []
        for block in receivedDataBlocks:
            blocks.append(BlocksForPickle(block))
        blockPath = outputPath + f"./Results/Congestion_Test/{pathing} {float(pd.read_csv('inputRL.csv')['Test length'][0])}/"
        os.makedirs(blockPath, exist_ok=True)
        try:
            np.save("{}blocks_{}".format(blockPath, GTnumber), np.asarray(blocks),allow_pickle=True)
        except pickle.PicklingError:
            print('Error with pickle and profiling')
        '''

        # save learnt values
        if pathing == 'Q-Learning':
            saveQTables(outputPath, earth1)
        elif pathing == 'Deep Q-Learning':
            saveDeepNetworks(outputPath + '/NNs/', earth1)

        # percentages.clear()
        receivedDataBlocks  .clear()
        createdBlocks       .clear()
        pathBlocks          .clear()
        allLatencies        .clear()
        upGSLRates          .clear()
        downGSLRates        .clear()
        interRates          .clear()
        intraRate           .clear()
        del results
        del earth1
        del env
        del _
        gc.collect()

        if len(GTs)>1:
            print('----------------------------------')
            print('Time:')
            end_time_GT = datetime.now()
            print(end_time_GT.strftime("%H:%M:%S"))
            print('----------------------------------')
            elapsed_time_GT = end_time_GT - start_time_GT
            print(f"Elapsed time for {GTnumber} GTs: {elapsed_time_GT}")
            print('----------------------------------')

    # plotLatenciesBars(percentages, outputPath)

    print('----------------------------------')
    print('Time:')
    end_time = datetime.now()
    print(end_time.strftime("%H:%M:%S"))
    print('----------------------------------')
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")
    print('----------------------------------')


###############################################################################
##############################     Main     ###################################
###############################################################################


if __name__ == '__main__':
    os.makedirs(outputPath, exist_ok=True) 
    sys.stdout = Logger(outputPath + 'logfile.log')

    RunSimulation(GTs, './', outputPath, populationMap, radioKM=rKM)
    # cProfile.run("RunSimulation(GTs, './', outputPath, populationMap, radioKM=rKM)")
