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

def bar_chart_j48(df):
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
    ax.set_xticklabels(x)
    ax.legend(df.columns, loc="center left",bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Configurations')
    ax.set_ylabel('Score')
    ax.set_title("Comparison of Decision Tree Configurations")

    # plt.annotate('Accuracy decreases upon changing min number of instances per leaf',
    #             xy=(12, 0.8),
    #             xytext=(0.55, 0.82),
    #             textcoords='figure fraction',
    #             fontsize=8,
    #             arrowprops=dict(facecolor='black', shrink=0.1)
    #             )


    plt.show()


def parallel_plot(x_data):
    fig, ax = plt.subplots(1)

    fig = pd.pandas.plotting.parallel_coordinates(
        x_data, 'Index', linewidth=0.3, axvlines=False)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_xticklabels(['TP Rate','FP Rate'  ,'Precision'  ,'Recall'  ,'F Measure'  ,'ROC Areas'])
    ax.set_title("Comparison of Decision Tree Configurations")
    ax.legend(title='Config')
    plt.show()


j48,randForest = read_config_csv()
j48 = get_only_metrics(j48)
randForest = get_only_metrics(randForest)

bar_chart_j48(j48)
bar_chart_j48(randForest)
