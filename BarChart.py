import matplotlib.pyplot as plt


def BarplotMe(Values, Labels, Y_label, title=None, X_label="Model"):
    # Plot the bar graph
    plot = plt.bar(Labels, Values)

    # Add the data value on head of the bar
    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width() / 2.,
                 1.002 * height, '%d' % int(height), ha='center', va='bottom')

    # Add labels and title
    plt.title(title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)

    # Display the graph on the screen
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt

speed = [1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant',
         'rabbit', 'giraffe', 'coyote', 'horse']
df = pd.DataFrame({'speed': speed,
                   'lifespan': lifespan}, index=index)
ax = df.plot.bar(rot=0)

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() ),fontsize=8)

plt.show()


Labels = ["BiGG","GraphRNN-S", "GraphRNN", "GRAN", "GraphVAE", "GraphVAE-MM"]
Values = [50, 60, 75, 45, 70, 105]
Y_label = "Train Time (Sec)"
title = "Average Trainning Time Per-Epoch"

BarplotMe(Values, Labels, Y_label = Y_label, title = title)

# Generation Time
Labels = ["BiGG","GraphRNN-S", "GraphRNN", "GRAN", "GraphVAE", "GraphVAE-MM"]
Values = [50, 60, 75, 45, 70, 105]
Y_label = "Generation Time (Sec)"
title = "Average Generation Time Per-Batch"

BarplotMe(Values, Labels, Y_label = Y_label, title = title)

print("hi")

