import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
combine_feature=pd.read_csv("combine_features.csv")
cv = CountVectorizer(stop_words='english') #creating new CountVectorizer() object
count_matrix = cv.fit_transform(combine_feature["combine_feature"]) #feeding combined strings(movie contents) to CountVectorizer() object
cosine_sim = cosine_similarity(count_matrix,count_matrix)

df = combine_feature.reset_index()
indices = pd.Series(df.index, index=df['title'])
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    if title.lower() in list(df["title"]):
        idx = indices[title.lower()]
    else:
        idx = indices[df["title"][random.randrange(1,4000)]]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    d=combine_feature
    d=d.apply(lambda x: x.str.capitalize() .str.strip() if isinstance(x, object) else x)
    return d['title'].iloc[movie_indices]

print(get_recommendations("The Dark Knight Rises"))

