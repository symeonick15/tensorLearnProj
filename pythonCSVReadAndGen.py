import csv
import os
import numpy as np

featureDatasetDir = '/home/nick/Downloads/low_freq/house_1'
featureDatasetName = '/channel_1.dat'

labelDatasetDir = '/home/nick/Downloads/low_freq/house_1'
labelDatasetName = '/channel_5.dat'

def getTimestampFromLine(row):
    return int(row[0])


def saveBatch(rows, oldDSDir, oldDSName):
    newDir = oldDSDir + "/proc"
    newPath = newDir + oldDSName
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    np.save(newPath,rows)

    # --- CSV WRITER ---
    # with open(newPath, 'w') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=' ')
    #     for row in rows:
    #         spamwriter.writerow([row])

def setSlidingWindowToBatch_Features(rows, windowSize, batchSize):
    npRows = np.array(rows)
    offset = 0
    batchWithWindow = [npRows[i+offset:i+offset+windowSize,1]
                       for i in range(batchSize-windowSize) ]
    batchWithWindow = np.reshape(batchWithWindow, (len(batchWithWindow), windowSize, 1))
    return batchWithWindow

def setSlidingWindowToBatch_Labels(rows, windowSize, batchSize):
    npRows = np.array(rows)
    offset = -1
    batchWithWindow = npRows[windowSize+offset:batchSize+offset,1]
    return batchWithWindow

def findInitTimestamp(oldDsDirX,oldDSNameX,oldDsDirY,oldDSNameY):
    with open(oldDsDirX+oldDSNameX, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        firstElemOfX = getTimestampFromLine( reader.__next__() )
    with open(oldDsDirY+oldDSNameY, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        firstElemOfY = getTimestampFromLine( reader.__next__() )

    if(firstElemOfX > firstElemOfY):
        return firstElemOfX
    else:
        return firstElemOfY

def preprocessData(oldDsDir,oldDSName,batchSize,windowSize,minTS,isFeatureSet):
    listOfRows = []
    with open(oldDsDir+oldDSName, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        lineCounter = 1;
        prevElem = getTimestampFromLine( reader.__next__() )
        while(prevElem < minTS):
            prevElem = getTimestampFromLine( reader.__next__() ) 
        firstElem = prevElem
        for row in reader: 
            curElem = getTimestampFromLine(row)
            if(curElem - prevElem > 1):
                #print(prevElem, " - ", curElem) #  -- Prints timegaps
                for ts in range(prevElem+1,curElem,1):
                    listOfRows.append([ts,row[1]]) 
                    #print(ts)

            listOfRows.append(row)
            prevElem = curElem
            lineCounter += 1
            if (len(listOfRows) >= batchSize): 
                if(isFeatureSet):
                    print(len(listOfRows))
                    rowsToSave = setSlidingWindowToBatch_Features(listOfRows,windowSize, batchSize)
                else:
                    print(len(listOfRows))
                    rowsToSave =setSlidingWindowToBatch_Labels(listOfRows, windowSize, batchSize)
                saveBatch(rowsToSave,oldDsDir,oldDSName)
                break



maxOfFirstTimestamps = findInitTimestamp(featureDatasetDir,featureDatasetName,labelDatasetDir,labelDatasetName)
preprocessData(featureDatasetDir,featureDatasetName,10000,10,maxOfFirstTimestamps,True)  # TEST #1 batchSize = 100000
preprocessData(labelDatasetDir,labelDatasetName,10000,10,maxOfFirstTimestamps,False)
