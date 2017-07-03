#include <iostream>
#include "base/base.h"
#include "kcws/cc/tf_seg_model.h"
#include "kcws/cc/pos_tagger.h"
#include "tensorflow/core/platform/init_main.h"

DEFINE_string(model_path, "kcws/models/seg_model.pbtxt", "the model path");
DEFINE_string(vocab_path, "kcws/models/basic_vocab.txt", "char vocab path");
DEFINE_string(pos_model_path, "kcws/models/pos_model.pbtxt", "the pos tagging model path");
DEFINE_string(word_vocab_path, "kcws/models/word_vocab.txt", "word vocab path");
DEFINE_string(pos_vocab_path, "kcws/models/pos_vocab.txt", "pos vocab path");
DEFINE_int32(max_sentence_len, 80, "max sentence len ");
DEFINE_string(user_dict_path, "", "user dict path");
DEFINE_int32(max_word_num, 50, "max num of word per sentence ");

int main(int argc, char** argv)
{
	tensorflow::port::InitMain(argv[0], &argc, &argv);
	google::ParseCommandLineFlags(&argc, &argv, true);
	//load model
	kcws::TfSegModel model;
	CHECK(model.LoadModel(FLAGS_model_path,
						FLAGS_vocab_path,
						FLAGS_max_sentence_len,
						FLAGS_user_dict_path))
	  << "Load model error";
	if (!FLAGS_pos_model_path.empty()) {
	kcws::PosTagger* tagger = new kcws::PosTagger;
	CHECK(tagger->LoadModel(FLAGS_pos_model_path,
							FLAGS_word_vocab_path,
							FLAGS_vocab_path,
							FLAGS_pos_vocab_path,
							FLAGS_max_word_num)) << "load pos model error";
	model.SetPosTagger(tagger);
	}
	//do segmentation
	while(true){
		std::vector<std::string> result;
        	std::vector<std::string> tags;
		std::string sentence;
		std::cout << "输入:"; 
		std::cin >> sentence;
		std::cout<<"******" << sentence << std::endl;
		if (model.Segment(sentence, &result, &tags)) {
			if (result.size() == tags.size()) {
				for (int i = 0; i < result.size(); i++) {
					printf("tok:%s\t\tpos:%s\n", result[i].c_str(), tags[i].c_str());
				}
			} else {
				for (std::string str : result) {
					printf("tok:%s", str.c_str());
				}
			}
		}
		else{
			printf("Segment Failed\n");
		}
	}

	return 0;
}
