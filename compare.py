from collections import defaultdict
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def get_text(file_name: str) -> str:
	chars_to_replace = ['\n', '.', ',', ';', ':', '"', '\'', '”', '-', '’', '᾽']	
	with open(file_name, 'r') as file:
		big_str = file.read()
		for char in chars_to_replace:
			big_str = big_str.replace(char, ' ')
	words = big_str.split(' ')
	words = [re.sub(r"\[.{1,5}\]", " ", word) for word in words]
	words = [word for word in words if len(word) > 0 and not word.isspace()]
	return ' '.join(words)


texts = [
	get_text('crito.txt'),
	get_text('euthyphro.txt'),
	get_text('laws.txt')
]


labels = ['crito', 'euthyphro', 'laws']
 
def create_heatmap(similarity, cmap = "YlGnBu"):
  df = pd.DataFrame(similarity)
  df.columns = labels
  df.index = labels
  fig, ax = plt.subplots(figsize=(5,5))
  sns.heatmap(df, cmap=cmap)
  plt.show()


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
arr = X.toarray()

print(cosine_similarity(arr))
print(create_heatmap(cosine_similarity(arr)))

