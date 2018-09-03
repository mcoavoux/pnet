

sh prepare_ag_corpus.sh &

sh prepare_blogs.sh &

sh prepare_trustpilot_dataset.sh & wait

python preprocess_blogs.py


