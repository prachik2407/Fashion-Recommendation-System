# Fashion Recommendation System

## Problem Statement
The goal of this project is to develop a fashion recommendation system that recommends visually similar fashion products to users. The system will extract features from user-uploaded product images and match them with other visually similar images. This functionality will help users find products that match their preferences based on visual similarity, enhancing their shopping experience.

## Features
- **Image Upload**: Users can upload images of fashion products they are interested in.
- **Feature Extraction**: The system extracts visual features from the uploaded images.
- **Similarity Matching**: The system matches the extracted features with a database of fashion products to find visually similar items.
- **Recommendations**: Users receive a list of fashion products that are visually similar to the uploaded image.

## Dataset
The dataset used for this project was downloaded from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset?resource=download). After downloading the dataset from Kaggle, the first step was to load the `styles.csv` and `images.csv` files with error handling to ensure data integrity. Once the data was loaded, it was verified by checking the first few rows and the shape of the dataframes to confirm that the data was correctly loaded. Next, the keys were extracted  from both `styles_df` and `images_df` to find common keys, which were then used to merge the two dataframes into a combined dataframe. This combined dataframe was then split into training and validation sets. To prepare for training, the directories are created for the training and validation sets, with subdirectories for each class based on `articleType`. Finally, the images were copied to their respective directories for training and validation sets. You can download them from this [One Drive link](https://jklujaipur-my.sharepoint.com/:f:/g/personal/prachikhandelwal_jklu_edu_in/EuRTpIkqs6NLk3YQCjeFVlQBeKpHlILKJkoopBKtHlA4cg?e=02JImN).