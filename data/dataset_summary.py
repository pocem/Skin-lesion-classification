import pandas as pd

# read the metadata to see the class distribution of real diagnostics
df = pd.read_csv(r'')

counts = df['diagnostic'].value_counts()

# Get the percentages
percentages = df['diagnostic'].value_counts(normalize=True) * 100

# Combine them into a nice new DataFrame for display
summary_df = pd.DataFrame({'count': counts, 'percentage': percentages})

print("Lesion Type Distribution Summary:")
print(summary_df)

#BCC           815   38.754161
#ACK           634   30.147408
#NEV           220   10.461246
#SEK           201    9.557775
#SCC           184    8.749406
#MEL            49    2.330005

#MALIGNANT --- BCC + SCC + ACK + MEL --- 1682 ~ 80%
#BENIGN ------ NEV + SEK --------------- 421 ~ 20% 
