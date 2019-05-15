import configuration
import train

##############################################################
#Configuration settings
#############################################################
testingClassifiers = configuration.getTestingClassifiers()
testingOptions = configuration.getTestingOptions()
testingTypes = configuration.getTestingTypes()
selectedDataset = configuration.getSelectedDataset()
isPlottingConfusionMatrix = configuration.getPlotSetting()
scoringMetrics = configuration.getScoringMetrics()
isRecordingResults = configuration.getRecordingSetting()

#########################################################
# RUN
#########################################################
for dataFile in selectedDataset:
    for dataType in testingTypes:
        for classifier in testingClassifiers:
            for option in testingOptions:
                    train.testDataset( classifier, option, dataType, dataFile, scoringMetrics, isPlottingConfusionMatrix, isRecordingResults)