import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_config_csv():
    sheets = pd.read_excel('evidence/configurations.xlsx',sheet_name=None)#,pd.read_excel('evidence/configurations.xlsx', sheet_name=['RandomForest'])
    return sheets["J48"],sheets["RandomForest"]

def get_only_metrics(df):
    metrics=["TP Rate", "FP Rate", "Precision", "Recall", "F Measure", "ROC Areas"]
    df = df[metrics]
    return df


def bar_chart(df, title,x_label, labels):
    fig, ax = plt.subplots(1,figsize=(15, 5))
    x = np.arange(len(df.index))

    bar_width = 0.1

    counter =0
    for i in df.columns:
        bar = ax.bar(x+(bar_width*counter), df[i],
                 width=bar_width)
        counter = counter +1

    # Fix the x-axes.
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend(df.columns, loc="center left",bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(x_label)
    ax.set_ylabel('Score')
    ax.set_title(title)


    return plt, ax

def annotate_j48(plt):
    plt.annotate('Accuracy decreases upon changing min number of instances per leaf',
            xy=(12, 0.8),
            xytext=(0.55, 0.82),
            textcoords='figure fraction',
            fontsize=8,
            arrowprops=dict(facecolor='black', shrink=0.1)
            )
    return plt


j48,randForest = read_config_csv()

# Plot results of different J48 Configurations
j48plt, j48ax = bar_chart(j48.iloc[:,5:],"Comparison of J48 Configurations", "Configurations", np.arange(15))
j48plt = annotate_j48(j48plt)
j48plt.show()

# Plot results of changing number of iterations
iterationLabels = np.sort(np.unique(randForest.loc[:,'NumIteration']))
RFIterations = randForest.iloc[6:15,:].sort_values('NumIteration').loc[:,'TP Rate':]
rPlt, rAx = bar_chart(RFIterations,"Changing Number of Iterations","Iterations",iterationLabels)
rPlt.show()
