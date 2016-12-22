"""
Random Forest Classification Example.
"""
from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="PythonRandomForestClassificationExample")

    if 0:
        data = MLUtils.loadLibSVMFile(sc, 'output.txt')
        sameModel = RandomForestModel.load(sc, "rf.model")
        predictions = sameModel.predict(data.map(lambda x: x.features))
        real_and_predicted = data.map(lambda lp: lp.label).zip(predictions)
        real_and_predicted=real_and_predicted.collect()
        print("real and predicted values")
        for value in real_and_predicted:
            print(value)
        print(predictions)
        exit()
    
    
    # $example on$
    # Load and parse the data file into an RDD of LabeledPoint.
    data = MLUtils.loadLibSVMFile(sc, 'output.txt')
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    #  Empty categoricalFeaturesInfo indicates all features are continuous.
    #  Note: Use larger numTrees in practice.
    #  Setting featureSubsetStrategy="auto" lets the algorithm choose.
    model = RandomForest.trainClassifier(trainingData,numClasses=25, categoricalFeaturesInfo={},
                                         numTrees=100, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=10, maxBins=32)

    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification forest model:')
    #print(model.toDebugString())

    # Save and load model
    model.save(sc, "rf.model")
    exit();
    #sameModel = RandomForestModel.load(sc, "rf.model")
    # $example off$
