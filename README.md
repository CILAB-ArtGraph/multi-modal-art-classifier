# Artwork Classification
The aim of this project was to develop an artwork classifier that uses both visual features and contextual information in form of KG (knowledge graph). Given an artwork image, the classifier want to predict the style and the genre. The KG used is a new version of ArtGraph KG (https://arxiv.org/abs/2105.15028). 

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

timm

dvc

## Models

- ResNet50. It uses only visual information. (`src/models/models.py`)

- ContextNet (Garcia et al, https://link.springer.com/article/10.1007/s13735-019-00189-4) (`src/models/models_kg.py`)

- A first multimodal approach proposed by the CILAB lab (Castellano et al, https://arxiv.org/abs/2105.15028) (`src/models/models_kg.py`)

- A new multimodal approach (`src/models/models_kg.py`).

Each model has two version: a single-task one and a multi-task one.

## Usage
Clone the repository, then

`cd multi-modal-art-classifier/src`

You can use [conda](https://docs.conda.io/en/latest/) to create a new environment running `conda env create --file=environment.yml`. 

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

For each model there is a train_*.py script.

## Download the dataset and the embeddings
You will need `dvc` for getting the dataset and the embeddings (download [here](https://drive.google.com/drive/folders/1omiDdfeC--Nb7X8Z0i2t8bfRpRInvBAe?usp=sharing) the artwork images). Anyway, in the folder `checkpoints` you can find all the generated models ready for testing. If you do not want to use dvc, download the folders `dataset`,
`projections` and `checkpoints` from [here](https://drive.google.com/drive/folders/1VD3E4h2hJMOUBloj9SyoZkheJohV60eB?usp=sharing)

Run `dvc` pull. After that wait until the required folders and files are downloaded.

Now, you can run the best model (it will use the best embeddings):
```
python train_new_multimodal_multitask.py
    --architecture vit
```

You can run it step by step using the notebook in `notebooks/proposed_model_multitask.ipynb`

## Authors and acknowledgment
Vincenzo Digeno, v.digeno@studenti.uniba.it

Prof. Gennaro Vessio