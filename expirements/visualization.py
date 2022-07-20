import matplotlib.pyplot as plt
import pandas as pd

def visualizationDistribution(df):
    text_word_count = []
    summary_word_count = []

    # populate the lists with sentence lengths
    for i in df["text_clean"]:
          text_word_count.append(len(i.split()))

    for i in df["y_clean"]:
          summary_word_count.append(len(i.split()))

    length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})
    length_df.hist(bins = 30)
    plt.show()
