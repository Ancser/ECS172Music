# Multi-Stage Recommendation Workflow Skill

## 0. Purpose

This note summarizes the Steam recommendation workflow we built, the concepts that were initially confusing, the final decisions, and how to reuse the same process for a future music recommendation algorithm.

The core lesson is:

```text
Stage 1 should optimize candidate recall.
Stage 2 should optimize final ranking quality.
Do not assume the signal that feels most personal is automatically best.
Validate each stage separately.
```

## 1. What We Built

We built a multi-stage recommender for Steam games.

Final pipeline:

```text
data
  -> Stage 1 retrieval
  -> top-300 candidate pool
  -> Stage 2 ranking
  -> top-10 recommendations
  -> validation metrics
  -> submission.csv
```

Final command:

```bat
cd /d F:\ancserProject\ECS172Steam
python steam.py --stage1-weights 0.10,0.10,0.80 --stage2-grid-step 0.20
```

Final best validation result:

```text
Algorithm: stage2_grid_manual_ranker
Stage 1 weights:
  content    0.10
  item-CF    0.10
  popularity 0.80

Stage 2 weights:
  content_score_norm 0.00
  itemcf_score_norm  0.20
  popularity_norm    0.20
  avg_playtime_norm  0.20
  tag_overlap        0.40

Retrieval@300: 0.47834
Recall@10:     0.06830
NDCG@10:       0.05293
Proxy:         0.06062
```

## 2. What Was Confusing At First

### 2.1 Two-Stage vs Two-Tower

Important distinction:

```text
Two-stage = system pipeline
Two-tower = model architecture
```

Our project is two-stage:

```text
Stage 1: retrieve candidates
Stage 2: rerank candidates
```

It is not a neural two-tower model. A two-tower model could be used as one possible Stage 1 retrieval method, but it is not required for this assignment.

### 2.2 Content Retrieval vs Item-CF vs Matrix Factorization

Content retrieval:

```text
Uses metadata.
Example: title, tags, genres, specs, developer, publisher, sentiment.
Can work for cold items if metadata exists.
```

Item-CF:

```text
Uses user-item interactions.
Example: users who played the same games also played these other games.
Cannot help items with no interactions.
```

Matrix factorization / ALS:

```text
Learns latent user vectors and latent item vectors from the interaction matrix.
Score = user vector dot item vector.
We did not implement ALS/MF in the final Steam code.
```

### 2.3 Interaction Data

In this assignment, every row in `train.csv` is a positive interaction.

Even when:

```text
playtime_minutes = 0
```

the row still counts as:

```text
label = 1
```

But playtime still affects profile strength. A zero-playtime row gets the minimum playtime weight, not a strong preference weight.

### 2.4 Matrix Density and Sparsity

The Steam user-item matrix is very sparse.

```text
10,000 users x 32,132 games = 321,320,000 possible pairs
122,366 observed interactions
density = 0.038%
sparsity = 99.962%
```

Meaning:

```text
Most user-game pairs are unknown.
Most users have very short histories.
Collaborative signals alone are weak for many users/items.
```

This is why content retrieval and popularity fallback matter.

## 3. Dataset Study

Before modeling, we studied the data distribution.

Important observations:

```text
train interactions: 122,366
train users: 10,000
catalog games: 32,132
games observed in train: 8,036
unseen catalog games: 24,096
average user games: 12.24
median user games: 7
cold users <= 3 games: 1,702
```

Interpretation:

```text
Many users have limited history.
Many games have no interaction data.
Popularity is likely useful.
Metadata is necessary for cold or sparse items.
```

For a future music dataset, do the same initial study:

```text
number of users
number of tracks/artists/albums
number of interactions
matrix density/sparsity
average listening history per user
median listening history
cold users
cold tracks
play count / listening duration distribution
metadata coverage
timestamp range
```

## 4. Stage 1 Retrieval Workflow

### 4.1 Goal

Stage 1 should retrieve a broad candidate pool.

The goal is not perfect ranking.

The goal is:

```text
Do not miss relevant items.
Maximize Retrieval@K.
```

For Steam, K was:

```text
K = 300
```

### 4.2 Steam Stage 1 Routes

We used three Stage 1 signals:

```text
content retrieval
item-CF retrieval
popularity retrieval
```

Content route:

```text
Build TF-IDF item vectors from metadata.
Build user content profile from weighted average of historical item vectors.
Score candidate by cosine similarity.
```

Item-CF route:

```text
Use interaction data.
For a user's historical games, find other users who played the same games.
Recommend other games those users played.
Normalize co-occurrence by popularity.
```

Popularity route:

```text
Use number of users who interacted with each game.
Helps cold users and sparse histories.
```

### 4.3 Stage 1 Grid Search

At first, we guessed weights manually.

Initial belief:

```text
Item-CF should be strongest because similar players have similar taste.
```

But validation showed:

```text
Item-CF-heavy retrieval had lower Retrieval@300.
```

Results:

```text
equal weights:
  content 0.33, item-CF 0.33, popularity 0.33
  Retrieval@300 = 0.37114

Item-CF heavy:
  content 0.30, item-CF 0.70, popularity 0.00
  Retrieval@300 = 0.33822

grid search:
  content 0.10, item-CF 0.10, popularity 0.80
  Retrieval@300 = 0.47831
```

Lesson:

```text
Popularity was not just a fallback.
It was the strongest Stage 1 recall signal for this dataset.
```

For future music recommendation, do not assume collaborative filtering is automatically best. Test:

```text
content/audio metadata similarity
user-track co-listening
artist/genre popularity
recent trending popularity
```

Then select Stage 1 weights by:

```text
Retrieval@K
```

## 5. Stage 2 Ranking Workflow

### 5.1 Goal

Stage 2 reranks the candidate pool.

Goal:

```text
Maximize top-N ranking quality.
```

For Steam:

```text
Recall@10
NDCG@10
Proxy = (Recall@10 + NDCG@10) / 2
```

### 5.2 Steam Stage 2 Features

We engineered five features:

```text
content_score_norm
itemcf_score_norm
popularity_norm
avg_playtime_norm
tag_overlap
```

Feature meaning:

```text
content_score_norm:
  similarity between user profile and candidate metadata

itemcf_score_norm:
  co-occurrence / shared-player score

popularity_norm:
  normalized number of users who interacted with the item

avg_playtime_norm:
  normalized average playtime for the item

tag_overlap:
  overlap between user history tags and candidate game tags
```

For music, analogous features:

```text
content_score_norm:
  similarity between user music profile and track metadata/audio embedding

itemcf_score_norm:
  co-listening score from users with overlapping listening history

popularity_norm:
  normalized track/artist popularity

avg_playtime_norm:
  average listen duration, completion rate, or play count

tag_overlap:
  overlap between user genre/mood/instrument tags and candidate track tags
```

### 5.3 Logistic Regression Lesson

We tried Logistic Regression as a learned Stage 2 ranker.

It performed much worse:

```text
stage1_fixed_logreg_ranker
Retrieval@300: 0.47827
Recall@10:     0.01117
NDCG@10:       0.00672
Proxy:         0.00895
```

Why:

```text
Logistic Regression optimizes binary classification loss.
The assignment evaluates top-10 ranking.
Negative sampling changes the learning problem.
Probability calibration does not guarantee good top-10 ordering.
```

Lesson:

```text
Do not assume a learned classifier is better.
If the metric is ranking, tune ranking weights directly when possible.
```

### 5.4 Stage 2 Grid Search

We replaced Logistic Regression with manual Stage 2 grid search.

Equal manual baseline:

```text
content_score_norm 0.20
itemcf_score_norm  0.20
popularity_norm    0.20
avg_playtime_norm  0.20
tag_overlap        0.20

Proxy = 0.05953
```

Best grid-searched Stage 2:

```text
content_score_norm 0.00
itemcf_score_norm  0.20
popularity_norm    0.20
avg_playtime_norm  0.20
tag_overlap        0.40

Proxy = 0.06062
```

Result:

```text
Stage 2 grid search improved Proxy from 0.05953 to 0.06062.
Recall@10 increased from 0.06604 to 0.06830.
NDCG@10 stayed almost the same.
```

Interpretation:

```text
Tag overlap helped find more correct items in the top 10.
Content score added little in Stage 2 because content was already used in Stage 1.
```

For future music recommendation, tune Stage 2 by validation metric:

```text
Fix Stage 1 candidate generation.
Build Stage 2 features.
Grid search feature weights.
Select weights by Recall@N / NDCG@N / Proxy.
```

## 6. Cold User and Cold Item Handling

### 6.1 Cold Users

Cold users have very few interactions.

In Steam:

```text
cold users = users with <= 3 observed games
```

Handling:

```text
Use whatever history exists to build a content profile.
Item-CF may be weak because there are few history items.
Popularity fallback stabilizes recommendation.
```

For music:

```text
Use a few listened tracks, artists, genres, or liked songs.
Back off to popular/trending songs when user history is too short.
```

### 6.2 Cold Items

Cold items have no training interactions.

In Steam:

```text
24,096 games had no training interactions.
```

Handling:

```text
Item-CF cannot recommend them through co-occurrence.
Popularity and average playtime are unavailable.
Metadata-based content retrieval gives them a possible path into the candidate pool.
But if final Stage 1 is popularity-heavy, cold items may still be hard to rank highly.
```

For music:

```text
New songs may have no listens yet.
Use metadata/audio features:
  artist
  album
  genre
  mood
  tempo
  audio embedding
  lyrics embedding
  release date
```

## 7. Reusable Algorithm Procedure

Use this workflow for the next music recommendation project.

### 7.1 Data Study

1. Count interactions, users, and items.
2. Compute matrix density and sparsity.
3. Study user history length distribution.
4. Study item popularity distribution.
5. Identify cold users and cold items.
6. Check metadata coverage.
7. Check timestamp range and define temporal validation.

### 7.2 Validation Split

Use temporal holdout:

```text
For each user:
  hold out latest interactions as validation positives
  train on earlier interactions
```

Avoid random split if the task is "what comes next".

### 7.3 Stage 1 Retrieval

Build several retrieval routes:

```text
content route
collaborative route
popularity/trending route
optional latent route
```

Merge candidates and tune retrieval weights by:

```text
Retrieval@K
```

### 7.4 Stage 2 Ranking

Build at least five features:

```text
user-side features
item-side features
interaction-side features
retrieval scores
metadata overlap
```

Tune Stage 2 by:

```text
Recall@N
NDCG@N
Proxy or assignment metric
```

### 7.5 Final Submission

After tuning:

```text
Fix the best Stage 1 weights.
Fix the best Stage 2 weights.
Rerun on full training data.
Generate final submission.
Validate submission format.
```

## 8. Commands From This Project

Full Stage 1 and Stage 2 search:

```bat
cd /d F:\ancserProject\ECS172Steam
python steam.py --stage1-grid-step 0.10 --stage2-grid-step 0.20
```

Final faster run after best Stage 1 is known:

```bat
cd /d F:\ancserProject\ECS172Steam
python steam.py --stage1-weights 0.10,0.10,0.80 --stage2-grid-step 0.20
```

Validation-only tuning:

```bat
cd /d F:\ancserProject\ECS172Steam
python steam.py --validation-only --stage1-weights 0.10,0.10,0.80 --stage2-grid-step 0.20
```

## 9. Final Lesson

The strongest practical lesson:

```text
Do not trust intuition alone.
Separate retrieval and ranking.
Evaluate retrieval with Retrieval@K.
Evaluate ranking with Recall@N and NDCG@N.
Tune each stage for its own job.
```

