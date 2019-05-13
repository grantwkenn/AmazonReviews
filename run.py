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

#########################################################
# RUN
#########################################################
for classifier in testingClassifiers:
    for option in testingOptions:
        for dataType in testingTypes:
            train.testDataset(classifier, option, dataType, selectedDataset, scoringMetrics, isPlottingConfusionMatrix)