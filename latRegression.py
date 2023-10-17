import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import scipy
import os


# Following is added due to incompatibility issues of numba with Python version > 3.7
class BlocksForPickle:
    def __init__(self, block):
        self.size = 64800  # size in bits
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


def getReg(data, numbPaths, pathMetric, resultsPath, alpha, test_case, n_packets=200, get_cdf=False, printUnstable=False):
    # split blocks into paths
    paths = [[[] for _ in range(numbPaths)] for _ in range(numbPaths)]
    latencies = np.zeros([numbPaths, numbPaths])
    t_values = np.zeros([numbPaths, numbPaths])
    p_values = []
    isNotZero = np.zeros([numbPaths, numbPaths])

    names = np.zeros(23)
    for block in data:
        name = block.ID.split("_")
        names[int(name[0])] = int(name[0])

    done = False
    i = 1
    while not done:
        if names[i] == 0:
            names = np.delete(names, i)
        else:
            i += 1
        if i >= len(names):
            done = True

    for block in data:
        name = block.ID.split("_")
        paths[np.where(names == int(name[0]))[0][0]][np.where(names == int(name[1]))[0][0]].append(block)

    print("Getting slope for {} gateways... ".format(numbPaths))
    total_num_paths = 0
    avg_totLatency = 0
    avg_propLatency = 0
    avg_txLatency = 0

    for source in range(numbPaths):
        for destination in range(numbPaths):
            if paths[source][destination]:
                lats = []

                for block in paths[source][destination][-n_packets:]:
                    lats.append(block.totLatency*1000)
                    avg_propLatency += block.propLatency*1e3
                    avg_txLatency += block.txLatency*1e3

                total_num_paths+=1
                reg = LinearRegression().fit(np.asarray([x for x in range(len(lats))]).reshape(-1, 1), np.asarray(lats))
                latencies[source,destination] = reg.coef_[0]        # Slope
                avg_totLatency += np.mean(lats)/(numbPaths*(numbPaths-1))
                
                s_x = np.sqrt(np.var(lats))
                errs = 0
                for latIndex, lat in enumerate(lats):
                    errs += (lat - (reg.intercept_ + reg.coef_[0]*latIndex))**2
                sigma2 = 1/(len(lats)-2) * errs
                seBeta1 = np.sqrt(sigma2)/(s_x*np.sqrt(len(lats)))
                if test_case =="two-sided":
                    t = np.abs(reg.coef_[0] - 0)/seBeta1
                    p_value = 2*(1-scipy.stats.t.cdf(t,len(lats)-2))
                    if t > scipy.stats.t.ppf(q=1-alpha/2,df=len(lats)-2):
                        isNotZero[source,destination] = 1
                elif test_case =="one-sided-lesser":
                    t = (reg.coef_[0] - 0)/seBeta1
                    p_value = 1-scipy.stats.t.cdf(t,len(lats)-2)
                    if t > scipy.stats.t.ppf(q=1-alpha,df=len(lats)-2):
                        isNotZero[source,destination] = 1
                else:    
                    t = (reg.coef_[0] - 0)/seBeta1
                    p_value = scipy.stats.t.cdf(t,len(lats)-2)
                    if t < scipy.stats.t.ppf(q=alpha,df=len(lats)-2):
                        isNotZero[source,destination] = 1
                t_values[source,destination] = t
                p_values.append(p_value)

    if printUnstable:
        print(np.nonzero(isNotZero))
    avg_propLatency /= (total_num_paths*n_packets)
    avg_txLatency /= (total_num_paths*n_packets)
    avg_queueLatency = avg_totLatency - avg_propLatency-avg_txLatency
    DF_t = pd.DataFrame(t_values)
    DF_z = pd.DataFrame(isNotZero)
    DF = pd.DataFrame(latencies)
    DF_t.to_csv(resultsPath + "t_blocks_{}.csv".format(pathMetric,numbPaths))       # t_values
    DF.to_csv(resultsPath + "slope_blocks_{}.csv".format(pathMetric,numbPaths))     # Slopes
    DF_z.to_csv(resultsPath + "nonZeroSlopes_blocks_{}.csv".format(pathMetric,numbPaths))   # Test result

    return np.sum(isNotZero), p_values, total_num_paths, np.array([avg_totLatency, avg_propLatency, avg_txLatency, avg_queueLatency])


def main():

    pathMetric = "dataRate"         # Choice between dataRate, latency, and Q-Learning
    path = "../Results/Congestion_test/{} 1s/".format(pathMetric)
    resultsPath = "../Results/Congestion_test/Results/{} 1s/".format(pathMetric)

    if not os.path.exists(resultsPath):
        # Create a new directory because it does not exist
        os.makedirs(resultsPath)

    significance_level = 0.05
    min_GWs = 6
    max_GWs = 10
    nonZeroes = []
    ratioNonZeroes = []
    average_latencies = np.zeros((max_GWs-min_GWs,4))
    test_type = "one-sided-lesser"
    p_values_vec = np.zeros((1,2))
    n_packets_for_regression = 200

    for numbGts in range(min_GWs,max_GWs):
        data = np.load(path + "blocks_{}.npy".format(numbGts), allow_pickle=True)
        print("no_paths {}".format(numbGts))
        numbNotZero, p_vals, no_paths, average_latencies[numbGts-min_GWs,:] = getReg(data,numbGts, pathMetric, resultsPath, significance_level, test_type, n_packets_for_regression, printUnstable=True)
        print("Average latency is {:0.3f}".format(average_latencies[numbGts-min_GWs,0]))
        nonZeroes.append(numbNotZero)
        ratioNonZeroes.append(numbNotZero/no_paths)
        p_values_vec = np.vstack((p_values_vec,np.hstack((np.ones((no_paths,1))*numbGts,np.reshape(p_vals,(no_paths,1))))))
    DF = pd.DataFrame(nonZeroes, index=list(range(min_GWs,max_GWs)))
    DF.to_csv(resultsPath + "numbNonZero.csv")
    DF = pd.DataFrame(ratioNonZeroes, index=list(range(min_GWs,max_GWs)))
    DF.to_csv(resultsPath + "ratioNonZero.csv")
    DF_avg_lat = pd.DataFrame(average_latencies, index=list(range(min_GWs,max_GWs)), columns=['total delay','prop_delay','transmission_delay','queue_delay'])
    DF_avg_lat.to_csv(resultsPath + "avgLatency.csv")
    p_values_vec = np.delete(p_values_vec, 0, 0)
    np.savetxt(resultsPath + "pVals.txt", p_values_vec)
    print(DF_avg_lat)
    return 0


if __name__ == '__main__':
    main()
