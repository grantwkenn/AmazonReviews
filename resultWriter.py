import csv

def writeResults (targetFilePath, option, classifier, dataType, precision, recall, f1, dataPath, elapsedTime):
    file = open(targetFilePath, "a+")
    fileWriter = csv.writer(file, quotechar="", quoting=csv.QUOTE_NONE, delimiter=',', escapechar='\\')

    fileWriter.writerow([
        option,
        classifier,
        dataType,
        precision,
        recall,
        f1,
        dataPath,
        elapsedTime
    ])
    file.close()