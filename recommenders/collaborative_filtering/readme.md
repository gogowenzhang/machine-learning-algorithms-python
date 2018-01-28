
# Item-item Collaborative Filtering Engine

### Pseudocode
```
1. Create the item-item similarity matrix
2. Create the neighborhood: find the n most similar items for each item
3. Predict rating for candidate items for each user: 
   the rating for each candidate item is the weighted average of the ratings of the items that the user has rated before and also in the candidate item's neighborhood. 
4. Recommend the top k items
```

### Cold star
New user: force ther user to rate 5 items or recommende popular items first. 
