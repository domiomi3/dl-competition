##### Which track(s) are you participating in? {Fast Networks Track, Large Networks Track, Both}
Fast Networks Track

##### What are the number of learnable parameters in your model?
Fast Networks Track -

##### Briefly describe your approach
Fast Networks Track - Our approach is based on the EfficientNetV2 model (Tan & Le, 2019a). It is trained from scratch on 
OxfordFlowers102 dataset and then fine-tuned on our dataset. We used Bayesian Optimization for 40 epochs, 
then train final model for 100 epochs.

Mingxing Tan, & Quoc V. Le (2021). EfficientNetV2: Smaller Models and Faster Training. CoRR, abs/2104.00298

##### Command to train your model from scratch
Fast Networks Track -
python src/main.py -m EfficientNetv2STuned -n dl_comp_effnetv2_test -b 16 -e 100 -d rand_augment -wb


##### Command to evaluate your model
Fast Networks Track -
python src/evaluate_model.py --model EfficientNetv2STuned 
--saved-model-file dl_comp_effnetv2_1_model_lr=0.00100_wd=0.01000_m=0.01000_dr=0.00001_cmix_prob=0.50000_beta=0.10000