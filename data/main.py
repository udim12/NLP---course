import tensorflow_datasets as tfds
import pandas as pd

import expirements.visualization
from data_loading import ds,info
from preprocessing import changeByteToStr
from preprocessing import preprocessText,splitData
from expirements.visualization import visualizationDistribution
# printing the dataset information:

def data_main():
    print(info.splits['train'].num_examples)
    print(info.splits['test'].num_examples)
    print(info.features)
    print(info.features['article'])

    # extracting the data from the tfds object into dataframe object
    tf_df = tfds.as_dataframe(ds.take(100000), info)
    df_byte = pd.DataFrame(tf_df).rename(columns={"article":"text", "highlights":"y"})[["text","y"]]

    df = changeByteToStr(df_byte)
    # cleaning the dataset, applying text preprocessing cleaning
    df = preprocessText(df)
    # visualization:
    visualizationDistribution(df)
    # splitting the data
    X_train, X_test, y_train, y_test = splitData(df,0.1,42)
    


