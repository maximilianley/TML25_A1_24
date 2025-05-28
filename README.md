# TML25_A1_24
Trustworthy ML Assignment 1

# Approach 1: Naive Filtering
My first and simplest idea was to to build the target model (resnet18), feed it the the private dataset as input and see what the confidences look like. If out of the 44 classes there would've been one confidence value above for example 0.95, then I spposed that this datapoint/image would have been for sure in the training set, as the confidence here is unusually high.

For this I set a threshold on 0.95 which acts as a filter. If any of the class predictions is at or above this threshold, then I would classify the image as a member, otherwise as a non-member.

This ultimately would not work when sending this as answer to the server. I concluded that the target model must generalize well, s.t. even non-members might be classified high.
Note that this approach does not work at all with the provided public data set, which contains membership labels.

So I attempted a different approach.


# Approach 2: LiRA Attack
I understood that instead of naively choosing a threshold, I had to somehow extract patterns in how the target model outputs confidences for member and non-member images.

The idea is simple:
Feed the target model with images, and then feed the output (44 logits) of the target model to a second inference model, which then outputs 0 or 1.

Out of the Membership Inference attacks that follow this idea, one that seemed useful was the Likelihood Ratio Attack. Here one has to create and train several shadow models (each having the same architecture as the target model (resnet18)), define training datasets (subset of the original complete dataset) for each of those shadow models, and train the shadow models with these datasets.
A shadow model would then output a 44 logit tensor for each input it receives, and this 44 logit output serves as input for the "master" classifier.

While trying to implement this approach, I realized however that in LiRA one has to create a shadow model for EACH data point, as LiRA only is effective if one trains with two datasets which differ only one (and not multiple) datapoints; in this case this would be all data points from which I know that they are non-members.

Here one would use the provided public dataset, as it contains the membership information for every data point and is there only fitting to be used for training.

LiRA for this task however is unfeasible, because one would have to define 2 x 10000 shadow models (and train them) for 10000 non-member data points.

This is not practical.


# Approach 3: Meta-Classifier
The third idea is very similar to the idea earlier, in the sense that the pipeline "img -> shadow_model -> output -> master_model -> 0/1" still remains. However I leave the idea of training for each single data point. Now instead, I define 10 shadow models, partition the public dataset into 10 pairwise disjoint subsets, and train each shadow model with one such subset.

Training the shadow models is done in the script "./shadow_training/shadow_trainer_main.py".

Each shadow models obtains 2000 inputs, 1000 member images and 1000 non-member images.
Even though the target model's weights are already provided in "01_MIA.pt", training shadow models form scratch gives more variation for later when training the master classifier.
The number 10 was chosen for:
- feasibility, as training takes a long time
- was deemed not to be too low, such as to provide enough variation and output quality for training the meta-classifier (master)

Note that for training the shadow models I cannot consider the private data set in any way. The reason is the following:

Training the shadows with the private set wouldn't be a problem, as the private set also provides the correct class label for the image (one class out of 44 possible classes).
The problem lies in the second step, when training the meta classifier (from now on called the "master model").
As for the shadow models, I need to do a supervised learning for my master model. This means, for every input of 44 confidence values, I need a corresponding label "0" or "1", telling me if this confidence corresponds to a membership or not. The private data set does not provide me this, only the public set does.

All 10 shadow_models can be found in the directory "/shadow_training/shadow_models/".

After all 10 shadow models have been trained, I need to evaluate once more them with the same data they have been trained. This is a necessary step i.o. to get the training dataset with the "44 confidence values" to train the master model. This procedure is done in the script "/shadow_training/evaluate/shadows_shuffled.py". Finally I reassemble all of their outputs back into a dataset of 20000 entries, which I call "evaluate_shadow_pub_shuffled.pt".



Training the master model.
The main difficulty here was how to actually construct the binary master classifier. I experimented around, but in the end I kept the model architecture as follows:
Every layer has a ReLU activation, just the output layer has no activation.
I train the master model with CrossEntropyLoss, which expects raw logits from the output layer.

Then I also asked myself whether I should feed the master model 44 logits or 44 confidence values as inputs. I conclude it was better to feed confidence values, as these are within 0 and 1. If it were logits, negative values would have been directly cut off by the ReLU in the first layer.

In addition I also enhanced the input data to somewhat extrapolate confident prediction values, so that the model would be able to more easily learn them during training.

Training this model takes a significant amount of time. The key realization is that the more layers one adds, the more this allows for expressiveness of features. This in turn shows during training, as the cross-entropy loss manages to go further down. For simplicity:

- a master model with 3 layers plateaus at loss of more or less 0.69, which is almost the initial loss when no learning has taken place yet
- a model with 6,7,8 layers can already reach a loss of 0.65

In addition, depending on how long I let the model train (number of epochs) the loss can decrease significantly, sometimes having reached 0.51 . The more epochs the model trains, the more it overfits.

After finishing training, I evaluate my master model once more on the training data to see how accurately it predicts a label for a respective input. With the loss value mentioned above, the models managed to predict with an accuracy of at least 70%, usually around 72% and 73%.

The training is done in the scripts "shadow_training/train_master_new_attempt_<number>.py", where the model used for submission on the server was trained in "train_master_new_attempt_6.py".

The model weights for this particular model are saved as "master_linear_46_44_64_128_180_150_105_64_32_2.pt".

Note that the model has 46 inputs instead of 44, to accomodate for the input normalization and enhancements done pre-training. It also has two output nodes, which (in confidence value format) always sum to 1.
For membership inference it suffices to look only at the second node, which represents '1' or 'member'. If above 0.5, then the confidence input is classified as member, non-member otherwise.

Also note that the private dataset in this approach is only used at the very end after the master model is trained. The private data is fed through the resnet18 target model, then the confidence outputs are fed to the master, and the master classifies. This is the last step and it is implemented in "/shadow_training/evaluate_master_new_attempt_6.py".

One observation I should make is that in general I let my models train for a very high number of epochs, for example 150 or 200. This is usually not recommended in ML because it means strong overfitting (even though in my case the loss just keeps going down all the same, even if slowly). But in this case overfitting is good, as it allows for stronger recognition of specific data points.


