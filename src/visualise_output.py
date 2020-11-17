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


def bar_chart(df, title,x_label, labels, value_labels = False):
    fig, ax = plt.subplots(1,figsize=(15, 5))
    x = np.arange(len(df.index))

    bar_width = 0.1

    counter =0
    for i in df.columns:
        bar = ax.bar(x+(bar_width*counter), df[i],
                 width=bar_width, tick_label="hi")
        counter = counter +1
        if value_labels is True:
            draw_value_labels(bar,ax)

    # Fix the x-axes.
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend(df.columns, loc="center left",bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(x_label)
    ax.set_ylabel('Score')
    ax.set_title(title)

    return plt, ax


def draw_value_labels(bar,ax):
    for rect in bar:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', fontsize=6)
 

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


# Plot results of changing max depth
maxDepthLabels = np.sort(np.unique(randForest.loc[:,'Max Depth']))
RFMaxDepth = randForest.iloc[22:,:].sort_values('Max Depth').loc[:,'TP Rate':]
rPlt, rAx = bar_chart(RFMaxDepth,"Changing Max Depth","Max Depth",maxDepthLabels[1:])
rPlt.show()

# Plot results of changing bag size percentage
bagSizeLabels = np.sort(np.unique(randForest.loc[:,'Bag Size Percent']))
RFMaxDepth = randForest.iloc[14:18,:].sort_values('Bag Size Percent').loc[:,'TP Rate':]
rPlt, rAx = bar_chart(RFMaxDepth,"Changing Bag Size Percentage","Bag Size Percentage (%)",bagSizeLabels[1:],value_labels=True)


rPlt.show()