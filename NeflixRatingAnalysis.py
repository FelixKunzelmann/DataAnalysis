import kaggle
import pandas as pd
import zipfile
import os
import matplotlib.pyplot as plt

# Download the Netflix dataset from Kaggle
# Dataset: https://www.kaggle.com/shivamb/netflix-shows
dataset_name = 'shivamb/netflix-shows'
download_path = './netflix-shows'

# Download the dataset
kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)

# Load the dataset into a pandas DataFrame
csv_file = os.path.join(download_path, 'netflix_titles.csv')
df = pd.read_csv(csv_file)

# Display the first few rows
print(df.head())

# Calculate rating counts separately for Movies and TV Shows
movies = df[df['type'] == 'Movie']
tv_shows = df[df['type'] == 'TV Show']

movie_ratings = movies['rating'].value_counts(dropna=False).sort_index()
tv_ratings = tv_shows['rating'].value_counts(dropna=False).sort_index()

print('\nMovie rating counts:')
print(movie_ratings)

print('\nTV Show rating counts:')
print(tv_ratings)

# Plot rating distribution for Movies and TV Shows
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

movie_ratings.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Movie Ratings')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Count')

tv_ratings.plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title('TV Show Ratings')
axes[1].set_xlabel('Rating')

plt.tight_layout()
plt.show()
