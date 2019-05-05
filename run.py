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

#########################################################
# RUN
#########################################################
for classifier in testingClassifiers:
    for feature in testingFeatures:
        for dataType in testingTypes:
            train.testDataset(classifier, feature, dataType, selectedDataset, isPlottingConfusionMatrix)