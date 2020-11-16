import pandas as pd

import matplotlib.pyplot as plt

def read_config_csv():
    return pd.read_excel('evidence/configurations.xlsx')
    
def get_only_metrics(df):
    df = df.drop(['Binary Splits','Pruning','Confidence Factor', 'Minimal Number of Instances Permissible per Leaf','Confusion Matrix'], axis=1)
    #df = df.rename (index={0:"Configuration"})
    df.index.name ="Configuration"
    indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    indexdf = pd.DataFrame(indexes)
    df['Index']=indexdf
    return df

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
print(newdf)
parallel_plot(newdf)