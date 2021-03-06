## CSCI699-HW1

### Introduction:
- This repository contains all the codes for experiment and visualization used for our report: **Visualizing the effect of domain adaptation in the loss
landscape of RoBERTa**
- Our work primarily focuses on visualizing the loss landscape of domain addapted RoBERTa during its fine-tuning process
- Our work was motivated by the Don't Stop Pre-training ACL 2020 paper (https://github.com/allenai/dont-stop-pretraining)

### Organizations:
- ``data`` contains the both the training and test dataset for the citation intent classification. It also contains our final visualized loss landscapes of three different RoBERTa models: 
	- RoBERTa-base
	- RoBERTa + CS DAPT (in-domain adaptation)
	- RoBERTa + NEWS DAPT (out-of-domain adaptation)
- ``raw_experiment_codes`` contains three Notebook files that were actually used during our experiment and visualization.
- ``cleaned_exp_codes.ipynb`` contains the final cleaned and annotated version of our experiment code.



