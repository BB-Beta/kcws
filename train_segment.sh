#clear segment_data folder
echo "clear environment"
rm segment_data/*
#convert training file to character based training file with each line a sentence
echo "build character raw training file"
python kcws/train/process_anno_file.py 2014 segment_data/pre_chars_for_w2v.txt
#extract character vocablary from character based training file
echo "count vocab"
./bazel-bin/third_party/word2vec/word2vec -train segment_data/pre_chars_for_w2v.txt -save-vocab segment_data/pre_vocab.txt -min-count 3
#replace low frequency character to unk
echo "replace unk"
python kcws/train/replace_unk.py segment_data/pre_vocab.txt segment_data/pre_chars_for_w2v.txt segment_data/chars_for_w2v.txt
#train character word embedding
echo "train char embedding"
./bazel-bin/third_party/word2vec/word2vec -train segment_data/chars_for_w2v.txt -output segment_data/vec.txt -size 50 -sample 1e-4 -negative 5 -hs 1 -binary 0 -iter 5
#build training total files
echo "generate training file"
./bazel-bin/kcws/train/generate_training segment_data/vec.txt 2014 segment_data/all.txt
#split train and test files from total files
echo "build train and test dataset"
python kcws/train/filter_sentence.py segment_data/all.txt segment_data/train.txt segment_data/test.txt
#export character vocab from character embedding to model path
echo "export character vocabulary"
./bazel-bin/kcws/cc/dump_vocab segment_data/vec.txt kcws/models/basic_vocab.txt
#training
echo "training model"
python kcws/train/train_cws_lstm.py --word2vec_path segment_data/vec.txt --train_data_path segment_data/train.txt --test_data_path segment_data/test.txt --max_sentence_len 80 --learning_rate 0.001
#extract model
echo "export model"
python tools/freeze_graph.py --input_graph logs/graph.pbtxt --input_checkpoint logs/model.ckpt --output_node_names "transitions,Reshape_7" --output_graph kcws/models/seg_model.pbtxt
