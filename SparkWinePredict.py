import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Wine_Training").getOrCreate()

def loadFile(dataPath):
    data1 = (spark.read
          .format("csv")
          .options(header='true',delimiter=';')
          .load(dataPath))
    data2 = data1.withColumnRenamed('"""""fixed acidity""""',"fixed acidity") \
        .withColumnRenamed('""""volatile acidity""""',"volatile acidity")\
        .withColumnRenamed('""""citric acid""""',"citric acid") \
        .withColumnRenamed('""""residual sugar""""',"residual sugar")\
        .withColumnRenamed('""""chlorides""""',"chlorides") \
        .withColumnRenamed('""""free sulfur dioxide""""',"free sulfur dioxide")\
        .withColumnRenamed('""""total sulfur dioxide""""',"total sulfur dioxide") \
        .withColumnRenamed('""""density""""',"density")\
        .withColumnRenamed('""""pH""""',"pH") \
        .withColumnRenamed('""""sulphates""""',"sulphates")\
        .withColumnRenamed('""""alcohol""""',"alcohol") \
        .withColumnRenamed('""""quality"""""',"quality")

    return data2

from pyspark.sql.functions import col
def changeType(data):
    wine_dataset = data.select(col('fixed acidity').cast('float'),
                             col('volatile acidity').cast('float'),
                             col('citric acid').cast('float'),
                             col('residual sugar').cast('float'),
                             col('chlorides').cast('float'),
                             col('free sulfur dioxide').cast('float'),
                             col('total sulfur dioxide').cast('float'),
                             col('density').cast('float'),
                             col('pH').cast('float'),
                             col('sulphates').cast('float'),
                             col('alcohol').cast('float'),
                             col('quality').cast('float')
                            )
    wine_dataset.show();
    return wine_dataset

def usingVectorAssembly(data):
    features_set = ['fixed acidity',
                        'volatile acidity',
                        'citric acid',
                         'residual sugar',
                         'chlorides',
                         'free sulfur dioxide',
                         'total sulfur dioxide',
                         'density',
                         'pH',
                         'sulphates',
                         'alcohol'
                ]
    assembler = VectorAssembler(inputCols=features_set, outputCol='features')
    vector_data = assembler.transform(data)
    return vector_data

from pyspark.ml.classification import LogisticRegression
def train_data(data):
    regression = LogisticRegression(labelCol='quality', featuresCol='features',maxIter=10, regParam=0.3, elasticNetParam=0.8)

    model = regression.fit(data)
    model.write().overwrite().save("s3://winequalitybucket/regression.model/")
    return model

from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml import Pipeline
def fetchmodel():
    regression_model = LogisticRegressionModel.load("s3://winequalitybucket/regression.model/")
    return regression_model

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
def predict(data, model):
    predicted_data = model.transform(data)

    return predicted_data

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
def label_accuracy(predictions):
    evaluator = MulticlassClassificationEvaluator(
        labelCol='quality',
        predictionCol='prediction',
        metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    return accuracy

def process(dataPath):
    load_data = loadFile(dataPath)
    typeOfData = changeType(load_data)
    vector_data = usingVectorAssembly(typeOfData)

    trained_data = train_data(vector_data)
    fetch_trained_model = fetchmodel()

    predicted_data = predict(vector_data, fetch_trained_model)
    accuracy = label_accuracy(predicted_data)
    print('Test Accuracy = ', (100*accuracy),'\n')


import sys
file_path = sys.argv[1]
process(file_path)
