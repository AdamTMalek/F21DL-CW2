import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_config_csv():
    return pd.read_excel('evidence/configurations.xlsx')
    
def get_only_metrics(df):
    df = df.drop(['Binary Splits','Pruning','Confidence Factor', 'Minimal Number of Instances Permissible per Leaf','Confusion Matrix'], axis=1)
    #df = df.rename (index={0:"Configuration"})
    df.index.name ="Configuration"
    # indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    # indexdf = pd.DataFrame(indexes)
    # df['Index']=indexdf
    return df

def bar_chart(df):
    fig, ax = plt.subplots(1,figsize=(15, 5))
    x = np.arange(len(df.index))

    # Define bar width. We'll use this to offset the second bar.
    bar_width = 0.1

    # Note we add the `width` parameter now which sets the width of each bar.
    # for (colName,colData) in df.iteritems():

    b1 = ax.bar(x, df['TP Rate'],
                width=bar_width)
    b2 = ax.bar(x+bar_width, df['FP Rate'],
                width=bar_width)
    b3 = ax.bar(x+bar_width*2, df['Precision'],
                width=bar_width)
    b4 = ax.bar(x+(bar_width*3), df['Recall'],
                width=bar_width)
    b5 = ax.bar(x+(bar_width*4), df['F Measure'],
                width=bar_width)
    b6 = ax.bar(x+(bar_width*5), df['ROC Areas'],
                width=bar_width)


    # Fix the x-axes.
    indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(indexes)
    ax.legend(df.columns, loc="center left",bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Configurations')
    ax.set_ylabel('Score')
    ax.set_title("Comparison of Decision Tree Configurations")

    plt.annotate('Accuracy Decrease',
                xy=(12, 0.8),
                xytext=(0.55, 0.82),
                textcoords='figure fraction',
                fontsize=16,
                arrowprops=dict(facecolor='black', shrink=0.1)
                )


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


df = read_config_csv()
newdf = get_only_metrics(df)
bar_chart(newdf)
