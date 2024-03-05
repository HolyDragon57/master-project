# master-project

Input: 45min-length video

Output: 45min-length video with action annotation

## data-preprocess

Output: short video clip - action label dataset

> Video clips are cut in two dimension: temporal and spatial

# Four stages

## human-tracking

ByteTrack

Output: spatial video cut

## temporal-action-localization

moving window

Output: temporal video cut

> Hyperparameter: 3-second window

## video-action-recognition

ClipAction

Output: recognition result file

## re-annotation

OpenCV

Output: video with annotation

# Problems

* Interactive action
* Long-range movement action
* Final action recognition model
  * If large model, how to finetune it? (PEFT & loss)