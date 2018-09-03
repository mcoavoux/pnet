
This is the code for the following article:

**Privacy-preserving Neural Representation of Text**  
Maximin Coavoux, Shashi Narayan, Shay B. Cohen  
EMNLP 2018  
[[preprint]](https://arxiv.org/abs/1808.09408) [[bib]](emnlp2018.bib)


### Launch experiments

See `dependencies.txt` for a list of dependencies.

Download and preprocess data (might take a long time since it also trains an LDA model on the blog dataset).

    cd dataset
    sh download_data.sh
    cd ..

To launch an experiment:

    cd src
    ## see python main.py --help for full description of options and list of dataset ids
    python main.py <modelname> <dataset_id> ( | --atraining | --generator | --ptraining --alpha <float>)
    # e.g.
    python main.py mymodel tp_fr --atraining 

The options to use the defense methods (during the training of the main model) that are used in the article are the following:

    --atraining Adversarial classification
    --generator Adversarial generation
    --ptraining --alpha <float> declustering


