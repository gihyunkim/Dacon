import seaborn as sns
import matplotlib.pyplot as plt

def normalization(inputs):
    return (inputs - inputs.min()) / (inputs.max() - inputs.min())

def get_corr(inputs):
    relate = inputs.corr(method="pearson")
    print(relate)
    relate.to_csv("./correlation.csv", mode="w")
    sns.heatmap(relate, annot=True, cmap="Greys")
    plt.show()