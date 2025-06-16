import pandas as pd

df = pd.read_csv(r'C:\Users\misog\portfolio\Machine learning skin lesion project\matched_data\metadata_matched.csv')

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