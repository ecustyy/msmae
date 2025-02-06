Modality-specificity multi-aware evidence fusion algorithm using CFP and OCT for fundus diseases diagnosis

There are two modalities, each of which uses separate branches.

CFP modality:
MSAS-ViT: cfp_mask_train.py (train); cfp_mask_test.py (test);
Graph-ViT: cfp_graph_train.py (train); cfp_graph_test.py (test);

OCT modality:
MSAS-ViT: oct_mask_train.py (train); oct_mask_test.py (test);
Graph-ViT: oct_graph_train.py (train); oct_graph_test.py (test);

Decision branch：evfusion.py (evidence fusion)

