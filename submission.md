#### Which track(s) are you participating in? {Fast Networks Track, Large Networks Track, Both}
Fast Networks Track

#### What are the number of learnable parameters in your model?
Fast Networks Track - 12,810

Large Networks Track - N/A
#### Briefly describe your approach
Fast Networks Track - To avoid heavy computations and satisfy the constraint on the model size we used 
[EfficientNetV2-S](https://arxiv.org/pdf/2104.00298.pdf) pretrained on ImageNet21k. To classify images from the 
given flower dataset, we unfroze the last fully-connected layer following loosely the Transfer Learning experiment 
made by the authors on OxfordFlowers102 dataset which yielded promising results (97.9±0.13). Additionally, to avoid 
overfitting that was likely to occur due to the high complexity model used on the fairly small dataset, we added 
regularization techniques such as CutMix, RandAugment, Dropout, and Weight Decay. We further used Bayesian 
Optimization to search for the optimal hyperparameter configuration by implementing [NePS](https://github.com/automl/neps) 
pipeline and running it for 20 iteration, 40 training epochs each. 
The final model was trained with the optimal hyperparameter configuration for 100 epochs.

Large Networks Track - N/A
#### Command to train your model from scratch
Fast Networks Track -
```
python src/main.py --model EfficientNetv2STuned 
                   --num_epochs 100 
                   --batch_size 16 
                   --data_augmentations rand_augment 
                   --learning_rate 0.00192 
                   --weight_decay 0.00054 
                   --dropout 0.46928 
                   --cutmix_prob 0.88355 
                   --beta 0.49222 
                   --exp_name fast_model
                   --competition
```

Large Networks Track - N/A
#### Command to evaluate your model
Fast Networks Track -
```
python src/evaluate_model.py --model EfficientNetv2STuned 
                             --saved_model_file fast_model 
                             --data_augmentations resize_to_224x224
```

Large Networks Track - N/A
#### Note
This project was finished thanks to the great contribution of M'Saydez Campbell from _one shot learners_ group
(Aria Ranjbar, Jade Martin-Lise, M’Saydez Campbell). 
Please, consider that in case we win.
