import configuration
import train

##############################################################
#Configuration settings
#############################################################
testingClassifiers = configuration.getTestingClassifiers()
testingFeatures = configuration.getTestingFeatures()
testingTypes = configuration.getTestingTypes()
selectedDataset = configuration.getSelectedDataset()
isPlottingConfusionMatrix = configuration.getPlotSetting()
scoringMetrics = configuration.getScoringMetrics()

#########################################################
# RUN
#########################################################
for classifier in testingClassifiers:
    for feature in testingFeatures:
        for dataType in testingTypes:
            train.testDataset(classifier, feature, dataType, selectedDataset, scoringMetrics, isPlottingConfusionMatrix)