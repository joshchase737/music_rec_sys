## Joshua Chase
## CS 274
## Music Recommendation System using Clustering

# Initialize the libraries to be used
library(dplyr)
library(cluster)
library(ggplot2)
library(proxy)
library(scales)
library(Rtsne)
library(clValid)
library(mclust) 
library(MixGHD)
library(clustMixType)
library(factoextra)
library(FactoMineR)
library(Rmixmod)
library(dbscan)
library(kernlab)
library(fclust)
library(proxy) 

### Data Collection / Preprocessing ------------------------------------------------------------

tunes <- read.csv("C:/Users/Josh Chase/Downloads/archive (1)/dataset.csv")

# Drop index column and find songs listed multiple times
tunes_combined <- tunes %>%
  dplyr::select(-X) %>%
  group_by(track_name, artists) %>%
  summarise(
    track_genre = paste(sort(unique(track_genre)), collapse = ", "),
    .groups = "drop"
  )

# drop song duplicates
tunes_dedup <- tunes %>%
  dplyr::select(-X) %>%
  group_by(track_name, artists) %>%
  slice_max(order_by = popularity, n = 1, with_ties = FALSE) %>%
  ungroup()

tunes_final <- tunes_dedup %>%
  left_join(tunes_combined, by = c("track_name", "artists"))

# reorder columns
desired_order <- c(
  names(tunes_final)[1:4],
  "popularity",
  setdiff(names(tunes_final), names(tunes_final)[1:4]) %>% setdiff("popularity")
)
tunes_final <- tunes_final[, desired_order]
tunes_final <- tunes_final[,-20]
tunes_final <- tunes_final %>% rename(track_genre = track_genre.y)

# clustering datasets for mixed/continuous data
tunes_full_clust <- tunes_final[,-c(1, 2, 3, 4, 7, 20)]
tunes_full_clust_cont <- tunes_full_clust[, -c(5, 7, 14)]
tunes_scaled <- scale(tunes_full_clust)

# create a sample of size 1000
set.seed(7)
sample_idx <- sample(nrow(tunes_final), size = 1000)
tunes_pop <- tunes_final[sample_idx, ]

# make sure data is structured correctly, and initialize data for clustering
tunes_clust <- tunes_pop[,-c(1, 2, 3, 4, 7, 20)]
tunes_clust <- tunes_clust %>%
  mutate(key = as.factor(key)) %>%
  mutate(mode = as.factor(mode)) %>%
  mutate(time_signature = as.factor(time_signature))
num_cols <- sapply(tunes_clust, is.numeric)
tunes_clust_scaled <- tunes_clust
tunes_clust_scaled[num_cols] <- scale(tunes_clust[num_cols])
tunes_clust_cont <- tunes_clust[, -c(5, 7, 14)]
tunes_clust_cont_scaled <- scale(tunes_clust_cont)

## k-means algorithm -----------------------------------------------------------

# Find best number of clusters based on silhouette and dunn
sil_avg <- NULL
dunn <- NULL
for (k in 5:30) {
  sil_vals <- numeric(20)
  dunn_vals <- numeric(20)
  for (i in 1:20) {
    set.seed(7 + i)
    kmeans <- kmeans(x = tunes_clust_cont, centers = k, nstart = 10)
    sil <- silhouette(kmeans$cluster, dist(tunes_clust_cont))
    sil_vals[i] <- mean(sil[, 3])
    dunn_vals[i] <- dunn(distance = dist(tunes_clust_cont), clusters = kmeans$cluster, Data = tunes_clust_cont)
  }
  sil_avg[k] <- mean(sil_vals)
  dunn[k] <- mean(dunn_vals)
}
sil_avg
dunn

# Runtime
system.time({
  kmeans <- kmeans(x = tunes_clust_cont, centers = 30, nstart = 10)
})
# 30 clusters, 0.54 sil, 0.001 dunn

# actually initialize the clustering
set.seed(7)
kmeans <- kmeans(x = tunes_clust_cont, centers = 30, nstart = 10)
centroids <- kmeans$centers

# Assign back to a complete dataset
tunes_pop_clustered <- tunes_pop %>%
  filter(row_number() %in% as.numeric(rownames(tunes_clust_cont))) %>%
  mutate(cluster = as.factor(kmeans$cluster))


## Step 3: Recommendation System--------------------------------------------------------

# Function for assigning clusters based on means
assign_cluster <- function(song_row) {
  song <- as.numeric(song_row)
  distances <- apply(centroids, 1, function(centroid) {
    sum((song - centroid)^2)
  })
  return(which.min(distances))
}

# assign clusters to full dataset
tunes_final$cluster <- apply(tunes_full_clust_cont, 1, assign_cluster)

# Distance to centroid
distance_to_centroid <- function(song_row, cluster_id) {
  song <- as.numeric(song_row)
  centroid <- centroids[cluster_id, ]  
  return(sqrt(sum((song - centroid)^2)))
}

# Cosine similarity
cosine_sim <- function(a, b) {
  if (all(a == 0) || all(b == 0)) return(0)
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}



# Recommendation function ------------------------------------------------------

recommend_songs <- function(song_title, artist_name = NULL, n = 5, preferred_genres = NULL, 
                            sim_weight = 0.5, dist_weight = 0.5,
                            popularity_filter = "Doesn't Matter") {
  
  # Filter input song by title and artist
  if (!is.null(artist_name)) {
    input_song <- tunes_final %>% 
      filter(tolower(track_name) == tolower(song_title),
             tolower(artists) == tolower(artist_name))
  } else {
    input_song <- tunes_final %>% filter(tolower(track_name) == tolower(song_title))
  }
  
  # Error messages
  if (nrow(input_song) == 0) {
    message("Song not found.")
    return(NULL)
  } else if (nrow(input_song) > 1) {
    message("Multiple songs found. Please specify artist_name.")
    return(input_song %>% dplyr::select(track_name, artists, cluster, track_genre, popularity))
  }
  
  # initialize input cluster and index
  input_cluster <- input_song$cluster[1]
  input_index <- which(tolower(tunes_final$track_name) == tolower(song_title) & 
                         tolower(tunes_final$artists) == tolower(ifelse(is.null(artist_name), input_song$artists[1], artist_name)))[1]
  
  # Create normalized features for similarity
  normalize <- function(x) (x - min(x)) / (max(x) - min(x))
  normalized_features <- as.data.frame(lapply(tunes_full_clust_cont, normalize))
  
  input_features <- normalized_features[input_index, , drop = FALSE]
  candidates_indices <- setdiff(which(tunes_final$cluster == input_cluster), input_index)
  
  # error for no genre
  if (!is.null(preferred_genres)) {
    preferred_genres <- tolower(preferred_genres)
    genre_matches <- sapply(tolower(tunes_final$track_genre[candidates_indices]), function(g) {
      any(sapply(preferred_genres, function(pref) grepl(pref, g)))
    })
    candidates_indices <- candidates_indices[genre_matches]
  }
  
  # popularity filter
  if (popularity_filter == "Very Popular") {
    candidates_indices <- candidates_indices[tunes_final$popularity[candidates_indices] > 90]
  } else if (popularity_filter == "Popular") {
    candidates_indices <- candidates_indices[tunes_final$popularity[candidates_indices] > 75]
  } else if (popularity_filter == "Somewhat Popular") {
    candidates_indices <- candidates_indices[tunes_final$popularity[candidates_indices] > 60]
  }
  
  # if there are no songs recommended
  if (length(candidates_indices) == 0) {
    message("No candidates found with the given preferences.")
    return(NULL)
  }
  
  # recommended song features
  candidates <- tunes_final[candidates_indices, ]
  candidate_features <- normalized_features[candidates_indices, , drop = FALSE]
  
  # similarities
  sims <- apply(candidate_features, 1, function(row) {
    cosine_sim(as.numeric(input_features), as.numeric(row))
  })
  
  # Use unscaled data for centroids
  unscaled_candidate_features <- tunes_full_clust_cont[candidates_indices, , drop = FALSE]
  dists <- apply(unscaled_candidate_features, 1, function(row) {
    distance_to_centroid(row, input_cluster)
  })
  
  # distances
  dists_norm <- (dists - min(dists)) / (max(dists) - min(dists))
  distance_score <- 1 - dists_norm
  
  candidates$similarity <- sims
  candidates$distance_to_centroid <- dists
  candidates$combined_score <- sim_weight * sims + dist_weight * distance_score
  
  # output recommendations
  recommendations <- candidates %>%
    arrange(desc(combined_score), desc(popularity)) %>%
    dplyr::select(track_name, artists, cluster, track_genre, popularity, similarity, distance_to_centroid, combined_score) %>%
    head(n)
  
  return(recommendations)
}

# Call k-means recommendation function
rk <- recommend_songs("Sweater Weather", "The Neighbourhood", n = 10, preferred_genres = c("indie", "rock"), popularity_filter = "Popular")
View(rk)

### Trying other Clustering Techniques -----------------------------------------

## K-means Scaled

# find number of clusters
sil_avg <- NULL
dunn <- NULL
for (k in 5:30) {
  sil_vals <- numeric(20)
  dunn_vals <- numeric(20)
  for (i in 1:20) {
    set.seed(7 + i)
    kmeans <- kmeans(x = tunes_clust_cont_scaled, centers = k, nstart = 10)
    sil <- silhouette(kmeans$cluster, dist(tunes_clust_cont_scaled))
    sil_vals[i] <- mean(sil[, 3])
    dunn_vals[i] <- dunn(distance = dist(tunes_clust_cont_scaled), clusters = kmeans$cluster, Data = tunes_clust_cont_scaled)
  }
  sil_avg[k] <- mean(sil_vals)
  dunn[k] <- mean(dunn_vals)
}
sil_avg
dunn

#runtime
system.time({
  kmeans <- kmeans(x = tunes_clust_cont_scaled, centers = 6, nstart = 10)
})
# 6 clusters, sil 0.15, dunn 0.08

## GHD

GHD_scaled <- MGHD(tunes_clust_cont_scaled, G=5:30, modelSel = "BIC")
sil <- silhouette(x = GHD_scaled@map, dist = dist(tunes_clust_cont_scaled))
summary(sil)
dunn(clusters = GHD_scaled@map, Data = tunes_clust_cont_scaled)
# 6 clusters, sil 0.05, dunn 0.06

# Runtime
system.time({
  GHD_scaled <- MGHD(tunes_clust_cont_scaled, G=6, modelSel = "BIC")
})


## Hierarchical

diss <- daisy(tunes_clust, metric = "gower")
diss_scaled <- daisy(tunes_clust_scaled, metric = "gower")

# Unscaled
hc_complete <- hclust(diss, method = "complete")
plot(hc_complete)
hc_complete_clusters <- cutree(hc_complete, h = 0.49)
sil <- silhouette(hc_complete_clusters, diss)
summary(sil)
dunn(distance = diss, clusters = hc_complete_clusters, Data = tunes_clust)
# 11 clusters, sil 0.13, dunn 0.15

# Scaled
hc_complete_scaled <- hclust(diss_scaled, method = "complete")
plot(hc_complete_scaled)
hc_complete_scaled_clusters <- cutree(hc_complete_scaled, h = 0.49)
sil <- silhouette(hc_complete_scaled_clusters, diss_scaled)
summary(sil)
dunn(distance = diss_scaled, clusters = hc_complete_scaled_clusters, Data = tunes_clust_scaled)
# 11 clusters, sil 0.13, dunn 0.15

# Runtime
system.time({
  hc_complete <- hclust(diss, method = "complete")
  plot(hc_complete)
  hc_complete_clusters <- cutree(hc_complete, h = 0.49)
})
# They are the same. 

## Spectral

# use gower dist
diss <- daisy(tunes_clust, metric = "gower")
gower_mat <- as.matrix(diss)
gower_sim <- 1 - gower_mat

# find best cluster count
sil_avg <- NULL
dunn <- NULL
for (k in 5:30) {
  sil_vals <- numeric(20)
  dunn_vals <- numeric(20)
  for (i in 1:20) {
    set.seed(7 + i)
    sc <- specc(as.kernelMatrix(gower_sim), centers = k)
    sil <- silhouette(sc@.Data, diss)
    sil_vals[i] <- mean(sil[, 3])
    dunn_vals[i] <- dunn(distance = diss, clusters = sc@.Data, Data = tunes_clust)
  }
  sil_avg[k] <- mean(sil_vals)
  dunn[k] <- mean(dunn_vals)
}
sil_avg
dunn

# runtime
system.time({
  sc <- specc(as.kernelMatrix(gower_sim), centers = 19)
})
# 19 Clusters work best, sil 0.11 and dunn 0.07

## Fuzzy K-Means

# find best number of clusters
sil_avg <- NULL
dunn <- NULL
for (k in 5:30) {
  sil_vals <- numeric(20)
  dunn_vals <- numeric(20)
  for (i in 1:20) {
    set.seed(7 + i)
    fkm <- FKM(tunes_clust_cont, k = k)
    sil <- silhouette(fkm$clus[,1], dist(tunes_clust_cont))
    sil_vals[i] <- mean(sil[, 3])
    dunn_vals[i] <- dunn(distance = dist(tunes_clust_cont), clusters = fkm$clus[,1], Data = tunes_clust_cont)
  }
  sil_avg[k] <- mean(sil_vals)
  dunn[k] <- mean(dunn_vals)
}
sil_avg
dunn

system.time({
  fkm <- FKM(tunes_clust_cont, k = 18)
})
# Choose 18 clusters, 0.54 sil, 0.005 dunn

## PAM

# Unscaled
diss <- daisy(tunes_clust, metric = "gower")

# best number of clusters
sil_avg <- NULL
dunn <- NULL
for (k in 5:30) {
  sil_vals <- numeric(20)
  dunn_vals <- numeric(20)
  for (i in 1:20) {
    set.seed(7 + i)
    pam_res <- pam(diss, k = k, diss = TRUE)
    sil <- silhouette(pam_res$clustering, diss)
    sil_vals[i] <- mean(sil[, 3])
    dunn_vals[i] <- dunn(distance = diss, clusters = pam_res$clustering, Data = tunes_clust)
  }
  sil_avg[k] <- mean(sil_vals)
  dunn[k] <- mean(dunn_vals)
}
sil_avg
dunn

# runtime
system.time({
  pam_res <- pam(diss, k = 24, diss = TRUE)
})
# Choose 24 clusters, sil 0.26, dunn 0.1

# Scaled
diss <- daisy(tunes_clust_scaled, metric = "gower")

# best k
sil_avg <- NULL
dunn <- NULL
for (k in 5:30) {
  sil_vals <- numeric(20)
  dunn_vals <- numeric(20)
  for (i in 1:20) {
    set.seed(7 + i)
    pam_res <- pam(diss, k = k, diss = TRUE)
    sil <- silhouette(pam_res$clustering, diss)
    sil_vals[i] <- mean(sil[, 3])
    dunn_vals[i] <- dunn(distance = diss, clusters = pam_res$clustering, Data = tunes_clust_scaled)
  }
  sil_avg[k] <- mean(sil_vals)
  dunn[k] <- mean(dunn_vals)
}
sil_avg
dunn
# Same as scaled

## New Recommendation System using PAM

# run pam and set medoids
set.seed(7)
diss <- daisy(tunes_clust, metric = "gower")
pam <- pam(diss, k = 24, diss = TRUE)
medoids_indices <- pam$medoids

# make sure data is formatted correctly
tunes_pop_clustered_pam <- tunes_pop
tunes_pop_clustered_pam$cluster <- as.factor(pam$clustering)
tunes_pop_clustered_pam <- tunes_pop_clustered_pam %>%
  mutate(key = as.factor(key),
         time_signature = as.factor(time_signature),
         mode = as.factor(mode))

medoids <- tunes_clust[medoids_indices,]


# Assign clusters to all rows:
tunes_final_pam <- tunes_final[,-21]
tunes_final_pam <- tunes_final_pam %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(key = as.factor(key),
         time_signature = as.factor(time_signature),
         mode = as.factor(mode))
tunes_full_clust <- tunes_full_clust %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(key = as.factor(key),
         time_signature = as.factor(time_signature),
         mode = as.factor(mode))

# use shunked cluster assigns, doing them all takes too long
assign_clusters_to_full_chunked <- function(all_songs, medoids_matrix, chunk_size = 1000) {
  n_all <- nrow(all_songs)
  n_med <- nrow(medoids_matrix)
  
  dist_matrix <- matrix(NA, nrow = n_all, ncol = n_med)
  
  for (start_idx in seq(1, n_all, by = chunk_size)) {
    end_idx <- min(start_idx + chunk_size - 1, n_all)
    chunk <- all_songs[start_idx:end_idx, , drop = FALSE]
    
    for (i in seq_len(n_med)) {
      combined <- rbind(chunk, medoids_matrix[i, , drop = FALSE])
      diss <- daisy(combined, metric = "gower")
      dist_matrix[start_idx:end_idx, i] <- as.matrix(diss)[1:nrow(chunk), nrow(chunk) + 1]
    }
  }
  
  clusters_assigned <- max.col(-dist_matrix)
  return(clusters_assigned)
}

# assign clusters
clusters <- assign_clusters_to_full_chunked(tunes_full_clust, medoids, chunk_size = 1000)


tunes_final_pam$cluster <- as.factor(clusters)

# gower distance function
distance_to_medoid <- function(candidate_row, medoid_features) {
  dist_obj <- daisy(rbind(candidate_row, medoid_features), metric = "gower")
  as.numeric(dist_obj)
}



# pam song recommendation
recommend_songs_pam <- function(song_title, artist_name = NULL, n = 5, preferred_genres = NULL, 
                                sim_weight = 0.5, dist_weight = 0.5,
                                popularity_filter = "Doesn't Matter") {
  
  # Filter input song by title and artist
  if (!is.null(artist_name)) {
    input_song <- tunes_final_pam %>% 
      filter(tolower(track_name) == tolower(song_title),
             tolower(artists) == tolower(artist_name))
  } else {
    input_song <- tunes_final_pam %>% filter(tolower(track_name) == tolower(song_title))
  }
  
  # error messages
  if (nrow(input_song) == 0) {
    message("Song not found.")
    return(NULL)
  } else if (nrow(input_song) > 1) {
    message("Multiple songs found. Please specify artist_name.")
    return(input_song %>% dplyr::select(track_name, artists, cluster, track_genre, popularity))
  }
  
  # make sure formatting is correct
  input_cluster <- as.integer(as.character(input_song$cluster[1]))
  input_index <- which(tolower(tunes_final_pam$track_name) == tolower(song_title) & 
                         tolower(tunes_final_pam$artists) == tolower(ifelse(is.null(artist_name), input_song$artists[1], artist_name)))[1]
  input_features <- tunes_full_clust[input_index, , drop = FALSE]
  
  
  # Start with songs in the same cluster, excluding the input song
  candidates_indices <- setdiff(
    which(as.integer(as.character(tunes_final_pam$cluster)) == input_cluster),
    input_index
  )
  
  # genre filter
  if (!is.null(preferred_genres)) {
    preferred_genres <- tolower(preferred_genres)
    genre_vec <- tolower(as.character(tunes_final_pam$track_genre[candidates_indices]))
    genre_matches <- sapply(genre_vec, function(g) {
      any(sapply(preferred_genres, function(pref) grepl(pref, g)))
    })
    candidates_indices <- candidates_indices[genre_matches]
  }
  
  # popularity filter
  if (popularity_filter == "Very Popular") {
    candidates_indices <- candidates_indices[tunes_final_pam$popularity[candidates_indices] > 90]
  } else if (popularity_filter == "Popular") {
    candidates_indices <- candidates_indices[tunes_final_pam$popularity[candidates_indices] > 75]
  } else if (popularity_filter == "Somewhat Popular") {
    candidates_indices <- candidates_indices[tunes_final_pam$popularity[candidates_indices] > 60]
  }
  
  # check if recommended songs
  if (length(candidates_indices) == 0) {
    message("No candidates found with the given preferences.")
    return(NULL)
  }
  
  # initialize candidates
  candidates <- tunes_final_pam[candidates_indices, ]
  candidate_features <- tunes_full_clust[candidates_indices, , drop = FALSE]
  
  # similarity
  sims <- sapply(1:nrow(candidate_features), function(i) {
    1 - as.numeric(daisy(rbind(input_features, candidate_features[i, , drop = FALSE]), metric = "gower")[1])
  })
  
  # distance 
  medoid_features <- medoids[input_cluster, , drop = FALSE]
  dists <- sapply(1:nrow(candidate_features), function(i) {
    distance_to_medoid(candidate_features[i, , drop = FALSE], medoid_features)
  })
  
  # error message
  if (all(is.na(dists)) || length(dists) == 0) {
    message("Could not compute distances to medoid.")
    return(NULL)
  }
  
  # normalize distances
  dists_norm <- (dists - min(dists, na.rm = TRUE)) / (max(dists, na.rm = TRUE) - min(dists, na.rm = TRUE))
  distance_score <- 1 - dists_norm
  
  # candidates
  candidates$similarity <- sims
  candidates$distance_to_medoid <- dists
  candidates$combined_score <- sim_weight * sims + dist_weight * distance_score
  
  #output recommendations
  recommendations <- candidates %>%
    arrange(desc(combined_score), desc(popularity)) %>%
    dplyr::select(track_name, artists, cluster, track_genre, popularity, similarity, distance_to_medoid, combined_score) %>%
    head(n)
  
  return(recommendations)
}

# Call pam recommendations
bl <- recommend_songs_pam("Sweater Weather", "The Neighbourhood", n = 5, preferred_genres = c("indie", "rock"), popularity_filter = "Popular")
View(bl)
