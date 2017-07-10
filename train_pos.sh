#convert training file to pos train version, with only contain word, exclude pos
python kcws/train/prepare_pos.py 2014  pos_data/pos_lines.txt
#count word vocab
bazel-bin/third_party/word2vec/word2vec -train pos_data/pos_lines.txt -min-count 5 -save-vocab pos_data/pre_word_vec.txt
#replace unk
python kcws/train/replace_unk.py pos_data/pre_word_vec.txt pos_data/pos_lines.txt pos_data/pos_lines_with_unk.txt
#train word embedding
bazel-bin/third_party/word2vec/word2vec -train pos_data/pos_lines_with_unk.txt -output pos_data/word_vec.txt -size 150 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0  -cbow 0 -iter 3 -min-count 5 -hs 1
#build pos tag vocabulary, and train file with word and pos in line
python kcws/train/stats_pos.py 2014 pos_data/pos_vocab.txt  pos_data/lines_withpos.txt
#build training data
bazel-bin/kcws/train/generate_pos_train pos_data/word_vec.txt segment_data/vec.txt  pos_data/pos_vocab.txt 2014 pos_data/pos_train.txt
#build train and test dataset
sort -u pos_data/pos_train.txt>pos_data/pos_train.u
shuf pos_data/pos_train.u >pos_data/pos_train.txt
head -n 230000 pos_data/pos_train.txt >pos_data/train.txt
tail -n 51362 pos_data/pos_train.txt >pos_data/test.txt
#export word vocabulary
./bazel-bin/kcws/cc/dump_vocab pos_data/word_vec.txt kcws/models/word_vocab.txt
#training
python kcws/train/train_pos.py --train_data_path pos_data/train.txt --test_data_path pos_data/test.txt --log_dir pos_logs --word_word2vec_path pos_data/word_vec.txt --char_word2vec_path segment_data/vec.txt
#export model
python tools/freeze_graph.py --input_graph pos_logs/graph.pbtxt --input_checkpoint pos_logs/model.ckpt --output_node_names "transitions,Reshape_9" --output_graph kcws/models/pos_model.pbtxt
