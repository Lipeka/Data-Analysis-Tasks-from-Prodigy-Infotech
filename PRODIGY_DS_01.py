import pandas as pd
import matplotlib.pyplot as plt
file_path = r"/Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_84031.csv"
data = pd.read_csv(file_path)
variable_to_visualize = 'Region'
plt.figure(figsize=(10, 6))
data[variable_to_visualize].value_counts().plot(kind='bar')
plt.title('Distribution of ' + variable_to_visualize)
plt.xlabel(variable_to_visualize)
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
