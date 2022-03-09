# Artwork Classification
The aim of this project is to develop an artwork classifier that uses both visual features and contextual information in form of KG (knowledge graph). Given an artwork image, the classifier want to predict the style and the genre. The KG used is a new version of ArtGraph KG (https://arxiv.org/abs/2105.15028). 

## Artgraph
ArtGraph is a knowledge graph based on WikiArt and DBpedia, that integrates a rich body of information about artworks, artists, painting schools, etc.
In recent works ArtGraph has been used for multi-output classification tasks. The objective is that given an artwork we want to predict its style and genre.

## Main requirements

numpy

pandas

pillow

pytorch 1.9

torchvision

pytorch_geometric 2.0.1

mlflow 1.2.0

scikit-learn

## Usage
Clone the repository, then

`cd multi-modal-art-classifier/src`

You can use [conda](https://docs.conda.io/en/latest/) to create a new environment running `conda env create --file=environments.yml`. 

### Single-task approach: 
An example of training a classifier which learn to predict the artworks' style.

1. Generate the node embeddings.
```
python train_gnn_embeddings.py --label style
```

2. Train the function which will learn to project visual features to the node embedding vector space.
```
python train_projector.py --node_embedding <node_embedding_name>.pt
```

3. Generate the projectors (the node embedding used by the artworks into the validation and test set)
```
python generate_projections.py
```

4. Train the classifier. For each artwork we need to provide the contextual embeddings. 
```
python train_new_multimodal.py 
    --emb_desc new_multimodal_style 
    --emb_train <train_node_embeddings>.pt 
    --emb_valid <valid_node_embeddings>.pt 
    --emb_test <test_node_embeddings>.pt 
    --emb_type <embedding_type> 
    --exp <experiment_name>
    --label style 
    --epochs <n_epochs> 
    --lr <learning_rate>
```


## Authors and acknowledgment
Vincenzo Digeno, v.digeno@studenti.uniba.it

Prof. Gennaro Vessio