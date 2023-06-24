# Collaborative Reasoning on Multi-Modal Semantic Graphs for Video-Grounded Dialogue Generation

This is the official code for the paper [Collaborative Reasoning on Multi-Modal Semantic Graphs for Video-Grounded Dialogue Generation](https://arxiv.org/abs/2210.12460) (EMNLP 2022 Findings)

*We study video-grounded dialogue generation, where a response is generated based on the dialogue context and the associated video. The primary challenges of this task lie in (1) the difficulty of integrating video data into pre-trained language models (PLMs) which presents obstacles to exploiting the power of large-scale pre-training; and (2) the necessity of taking into account the complementarity of various modalities throughout the reasoning process. Although having made remarkable progress in video-grounded dialogue generation, existing methods still fall short when it comes to integrating with PLMs in a way that allows information from different modalities to complement each other. To alleviate these issues, we first propose extracting pertinent information from videos and turning it into reasoning paths that are acceptable to PLMs. Additionally, we propose a multi-agent reinforcement learning method to collaboratively perform reasoning on different modalities (i.e., video and dialogue context). Empirical experiment results on two public datasets indicate that the proposed model can significantly outperform state-of-the-art models by large margins on both automatic and human evaluations.*

## Overview

![model](image/README/model.png)

we first propose extracting pertinent information from videos and turning it into reasoning paths that are acceptable to PLMs. Additionally, we propose a multi-agent reinforcement learning method to collaboratively perform reasoning on different modalities (i.e., video and dialogue context)

## Installation

```
pip install requirements.txt
```

## Data Preparation

*We will maintain the code for creating the multi-modal semantic graph, but you can reproduce it following the details in the paper.*

- Download [DSTC7-AVSD](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge)
- Download [Twitch-FIFA](https://github.com/ramakanth-pasunuru/video-dialogue)
- Install [word2vec](https://code.google.com/archive/p/word2vec/)
- Coreference Resolution: [AllenNLP](https://github.com/allenai/allennlp-models) | [Huggingface](https://github.com/huggingface/neuralcoref) | [references](https://github.com/NeuroSYS-pl/coreference-resolution) | [ChatGPT](https://chat.openai.com/)
- Open IE: [OPENIE 5.1](https://github.com/dair-iitd/OpenIE-standalone) | [OPENIE6](https://github.com/dair-iitd/openie6) | [Stanford OpenIE](https://nlp.stanford.edu/software/openie.html) | [ChatGPT](https://chat.openai.com/)
- Video Action Extraction: [PytorchVideo](https://pytorchvideo.org/) | [SLOWFAST](https://github.com/facebookresearch/SlowFast)
- The data structure should look like the following (default)

```
data/
├── avsd/ # avsd dataset
    ├── vggish
    ├── i3d_flow
    ├── i3d_rgb
    ├── train.json
    ├── val.json
    └── test.json
├── mm-graph/semantic_graph/ # multi-modal semantic graph
    ├── avsd_context/traj.json
    ├── avsd_caption/traj.json
    └── avsd_video/traj.json

```

## MARL_SG

*It is worth noting that we add the implementation of random reward ([Discovering Diverse Multi-Agent Strategic Behavior via Reward Randomization](https://arxiv.org/abs/2103.04564)) and simulated annealing ([Generating Informative Dialogue Responses with Keywords-Guided Networks](https://arxiv.org/abs/2007.01652)) in this version of code for further study.*

### Train

```
python marl_vt_train_rr.py \
	--train_path data/avsd/train.json \
	--valid_path data/avsd/valid.json \
	--fea_path data/avsd/ \
	--context_traj_path data/mm-graph/semantic_graph/avsd_context/traj.json \
	--caption_traj_path data/mm-graph/semantic_graph/avsd_caption/traj.json \
	--video_traj_path data/mm-graph/semantic_graph/avsd_video/traj.json \
	--train_batch_size 4 \
	--valid_batch_size 4 \
```

### Test

```
python marl_vt_generate.py \
	--test_set data/avsd/test.json \
	--context_traj_path data/mm-graph/semantic_graph/avsd_context/traj.json \
	--caption_traj_path data/mm-graph/semantic_graph/avsd_caption/traj.json \
	--video_traj_path data/mm-graph/semantic_graph/avsd_video/traj.json \
	--beam_size 5 \
	--max_length 18 \
	--min_length 1 \
	--penalty 0.4 \
	--ckptid SELECTED_CKPT
```

## Citation

```
@inproceedings{zhao-etal-2022-collaborative,
    title = "Collaborative Reasoning on Multi-Modal Semantic Graphs for Video-Grounded Dialogue Generation",
    author = "Zhao, Xueliang  and
      Wang, Yuxuan  and
      Tao, Chongyang  and
      Wang, Chenshuo  and
      Zhao, Dongyan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.442",
    pages = "5988--5998","",
}

```
