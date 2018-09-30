from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.electric import align_two_meters
from nilmtk.disaggregate import Disaggregator
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import neural_network
import metrics
from sklearn.preprocessing import StandardScaler


print("========== OPEN DATASETS ============")
train_building = 1
test_building = 1
sample_period = 6
meter_key = 'microwave'
dsPathsList = ['daeStackTrain.h5','gruStackTrain.h5','rnnStackTrain.h5']
dsPathsList_Test = ['daeStackTest.h5','gruStackTest.h5','rnnStackTest.h5']

#Read Y (target) file
dsPathY = '/home/nick/ukdale.h5'
trainYDS = DataSet(dsPathY)
trainY_elec = trainYDS.buildings[train_building].elec
trainY_meter = trainY_elec.submeters()[meter_key]
testY_elec = trainYDS.buildings[test_building].elec
testY_meter = testY_elec.submeters()[meter_key]

trainXGen_list = []
trainX_listForScale = []
for path in dsPathsList:
    train = DataSet(path)
    train_elec = train.buildings[train_building].elec
    train_meter = train_elec.submeters()[meter_key]
    # Align the Y file with the X file (smaller). Normally this would be needed only 1 time, but it's also a way to read the X meters chunk-by-chunk
    aligned_meters = align_two_meters(train_meter, trainY_meter)
    trainXGen_list.append(aligned_meters)
    #Another iterator to pass through data to get their scaling stats
    train_series = train_meter.power_series(sample_period=sample_period)
    trainX_listForScale.append(train_series)
test_series_list = []
for path in dsPathsList_Test:
    test = DataSet(path)
    test_elec = test.buildings[test_building].elec
    test_meter = test_elec.submeters()[meter_key]
    test_series = test_meter.power_series(sample_period=sample_period)
    test_series_list.append(test_series)

#========================= Scale data (fit)===================================
# (Read once all the train data to fit the scaler)
# Note: there's probably no need to align meters here, because data will be read, just to have the stats for scaling calculated
scaler = StandardScaler()
run = True
trainX = []
columnInd = 0
#1st chunk of each series + the target series (NtS: Make this a function that returns the trainX numpy array)
for trainXIt in trainX_listForScale:
    chunkX = next(trainXIt)
    if(trainX == []):
        trainX = np.zeros([len(chunkX),len(trainX_listForScale)]) #Initialize the array that will hold all of the series as columns
    trainX[:, columnInd] = chunkX[:]
    columnInd +=1

while run:
    scaler.partial_fit(trainX)  # fit only on training data
    try:
        trainX = []
        columnInd = 0
        # 1st chunk of each series + the target series (NtS: Make this a function that returns the trainX numpy array)
        for trainXIt in trainX_listForScale:
            chunkX = next(trainXIt)
            if (trainX == []):
                trainX = np.zeros([len(chunkX), len(
                    trainX_listForScale)])  # Initialize the array that will hold all of the series as columns
            trainX[:, columnInd] = chunkX[:]
            columnInd += 1
    except:
        run = False

#========================= Train chunks ================================
run = True
trainX = []
columnInd = 0
#1st chunk of each series + the target series (NtS: Make this a function that returns the trainX numpy array)
for trainXGen in trainXGen_list:
    chunkALL = next(trainXGen)
    chunkX = chunkALL['master'] #Only the X chunk is needed
    if(trainX == []):
        trainX = np.zeros([len(chunkX),len(trainXGen_list)]) #Initialize the array that will hold all of the series as columns
    trainX[:, columnInd] = chunkX[:]
    if(columnInd == 0): #Use the alligned Y meter of the first dataset (the rest are the same)
        chunkY = chunkALL['slave']
        chunkY.fillna(0, inplace=True)
        trainY = np.zeros([len(chunkY),1])
        trainY[:,0] = chunkY[:] # ??Should Y also be scaled (in scikit-learn tips, the scaling is only applied to the X vectors) ??
    columnInd += 1
trainX = scaler.transform(trainX)

# Go through chunks and feed them to the model
clf = linear_model.SGDRegressor()
#clf = neural_network.MLPRegressor(hidden_layer_sizes=(1,), activation='identity')
while(run):
    # Partial Fit (incremental learning)
    clf.partial_fit(trainX, trainY)
    # print(trainX)
    # print(trainY)

    try:
        trainX = []
        columnInd = 0
        #Gen next chunk of each series + the target series
        for trainXGen in trainXGen_list:
            chunkALL = next(trainXGen)
            chunkX = chunkALL['master']  # Only the X chunk is needed
            if (trainX == []):
                trainX = np.zeros([len(chunkX), len(
                    trainXGen_list)])  # Initialize the array that will hold all of the series as columns
            trainX[:, columnInd] = chunkX[:]
            if (columnInd == 0):  # Use the alligned Y meter of the first dataset (the rest are the same)
                chunkY = chunkALL['slave']
                chunkY.fillna(0, inplace=True)
                trainY = np.zeros([len(chunkY),1])
                trainY[:, 0] = chunkY[:]
            columnInd += 1
        trainX = scaler.transform(trainX)
    except:
        run = False


#================= Predictions for each chunk of input file + Write them to the output (same h5 format/structure)
MIN_CHUNK_LENGTH = 300 # Depends on the submodels of the ensemble
timeframes = []
building_path = '/building{}'.format(test_meter.building())
mains_data_location = building_path + '/elec/meter1'
data_is_available = False
disag_filename = "ens-out.h5"
output_datastore = HDFDataStore(disag_filename, 'w')

run = True
testX = []
columnInd = 0
#1st chunk of each series
for testXGen in test_series_list:
    chunk = next(testXGen)
    if(testX == []):
        testX = np.zeros([len(chunk),len(test_series_list)]) #Initialize the array that will hold all of the series as columns
    testX[:, columnInd] = chunk[:]
    columnInd += 1
testX = scaler.transform(testX)

while run:
    if len(chunk) < MIN_CHUNK_LENGTH:
        continue
    print("New sensible chunk: {}".format(len(chunk)))

    timeframes.append(chunk.timeframe)
    measurement = chunk.name

    pred = clf.predict(testX)
    column = pd.Series(pred, index=chunk.index, name=0)
    appliance_powers_dict = {}
    appliance_powers_dict[0] = column
    appliance_power = pd.DataFrame(appliance_powers_dict)
    appliance_power[appliance_power < 0] = 0

    # Append prediction to output
    data_is_available = True
    cols = pd.MultiIndex.from_tuples([chunk.name])
    meter_instance = train_meter.instance() # ??? does it matter if it's test meter ???
    df = pd.DataFrame(
        appliance_power.values, index=appliance_power.index,
        columns=cols, dtype="float32")
    key = '{}/elec/meter{}'.format(building_path, meter_instance)
    output_datastore.append(key, df)

    # Append aggregate data to output
    mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
    output_datastore.append(key=mains_data_location, value=mains_df)

    try:
        testX = []
        columnInd = 0
        # Next chunk of each series
        for testXGen in test_series_list:
            chunk = next(testXGen)
            if (testX == []):
                testX = np.zeros([len(chunk), len(
                    test_series_list)])  # Initialize the array that will hold all of the series as columns
            testX[:, columnInd] = chunk[:]
            columnInd += 1
        testX = scaler.transform(testX)
    except:
        run = False

# Save metadata to output
if data_is_available:

    disagr = Disaggregator()
    disagr.MODEL_NAME = 'EnsTest'

    disagr._save_metadata_for_disaggregation(
        output_datastore=output_datastore,
        sample_period=sample_period, # ??? is it a problem that is hardcoded ???
        measurement=measurement,
        timeframes=timeframes,
        building=test_meter.building(),
        meters=[test_meter])


#======================== Calculate Metrics =====================================
result = DataSet(disag_filename)
res_elec = result.buildings[test_building].elec
rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], testY_meter)
print("============ Recall: {}".format(rpaf[0]))
print("============ Precision: {}".format(rpaf[1]))
print("============ Accuracy: {}".format(rpaf[2]))
print("============ F1 Score: {}".format(rpaf[2]))
print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec[meter_key], testY_meter)))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec[meter_key], testY_meter)))