# Video Search Engine

Authors:

* [Abby Gray](inserturl.com)
* [Akshat Shrivastava](Github.com/AkshatSh)
* [Kevin Bi](inserurl.com)
* [Sarah Yu](inserturl.com)

Semantically be able to search through a database of videos (using generated summaries)

## Plan

Our project will, broadly defined, be attempting video searching through video summarization. To do this we propose the following objectives and resulting action plan:

* Break videos down into semantically different groups of frames
* Recognize objects in an image (i.e. a frame)
* Convert a frame to text
* Merge summaries of all frames of a video into one large overall summary 
* Build a search engine to query videos via summary.

## Goals

For our project, we have come up with a basic goal we plan to reach by the time of the presentation, and a stretch goal we hope to reach if time permits

**Basic Goal:** We will recognize objects through the YOLO algorithm (https://pjreddie.com/darknet/yolo/). Convert each frame to text using the algorithm mentioned in this paper (https://arxiv.org/abs/1707.07102). Come up with basic heuristic for skipping frames so not too much overlap in the summary. Surface all of this through a simple UI to search a video database.

**Stretch Goal:** Investigate other methods for reducing noise in frames (Generative Adversarial Networks), Investigate grouping together semantically similar frames to one common representation to make better summaries.

## Data Sets to Use

### [Common Object Data Set](http://cocodataset.org/#home) 

Consists of labeled images for image captioning


### [Sum Me Data Set](https://people.ee.ethz.ch/~gyglim/vsum/)

Consists of action videos that can be used to test summaries.



