import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# stats
from scipy import stats

# missing values
import missingno as msno

import polymer

polymer.train_df.info(show_counts=True, verbose=True)


msno.matrix(polymer.train_df[polymer.targets])
plt.show()
