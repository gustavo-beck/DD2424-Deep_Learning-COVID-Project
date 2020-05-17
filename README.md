# Transfer learning for COVID-19 detection
SARS-COV-2 constitutes a novel virus with a relatively high infection rate [2].
As such, early and effective detection is needed in order to track and impede the
spread of the pandemic. We propose a Deep Learning approach for detecting
COVID-19 based on X-ray images. Aside from detection, our network provides
insight about the affected area, the core locations of infection and how similar diseases are when represented by networks. Our findings illustrate promising results
both in localisation as well as AUC, reaching a ROC curve performance of 0.88 on
the COVID-19 X-ray Dataset. Our report also analyses the importance of Transfer
Learning, Model Architecture and Data Pre-processing.

Our  model  builds  upon  well  established  architectures  for  robustness  and  efficacy.   It  consists  of a core deep Convolutional Neural Network which is augmented with a fully connected layer forclassification. Mainly two types of architetures were explored: DenseNet121 and ResNet50. 

Our Deep Neural Network approach produced noteworthy results, showcasing great potential androbustness in diagnostic tasks.  Its latent representation was able to properly separate the classesand created distinct clusters, by producing a new vector space that can model sample characteristicsmore accurately.  Furthermore, we showed that the network can be utilized for other tasks, not justdiscriminative,  by  applying  SOMs  to  uncover  class  relationships  and  using  heatmaps  (CAM)  tohighlight the affected region.  Using the latter, the network’s decision can be investigated to avoidoverfitting due to data set particularities (e.g. small and unbalanced data sets). Our experiments showthat an effective way of avoiding this is using pre-trained models, as they initialize the networks inbetter regions of the learning space, avoiding such undesirable overfit solutions.

## Results on ChestX14 dataset
### Roc Curves
<img src="figures/totalTest_DenseNet.png">

### CAM examples

<div class="row">
  <div class="column">
    <img src="figures/heatmap_00000001_000.png_Cardiomegaly.png" width = "150" >
  </div>
  <div class="column">
    <img img src="figures/heatmap_00000011_006.png_Atelectasis.png" width="150">
  </div>
  <div class="column">
    <img src="figures/heatmap_00000022_001.png_Fibrosis.png" width="150">
  </div>
</div>

### Comparison Relative Results

<img src="figures/comparativeResults.png" >

## Results on COVID-19 Dataset
### Result Table
<img src="figures/CovidTable.png">

### CAM examples

<div class="row">
  <div class="column">
    <img src="figures/heatmap_9fdd3c3032296fd04d2cad5d9070d4_jumbo.jpeg_.png" width = "150">
  </div>
  <div class="column">
    <img img src="figures/heatmap_E63574A7-4188-4C8D-8D17-9D67A18A1AFA.jpeg_.png" width="150">
  </div>
  <div class="column">
    <img src="figures/heatmap_auntminnie-d-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg_.png" width="150">
  </div>
</div>

## 15 Classification problem (COVID + chestXray14)

<img src="figures/ROC_ADRIAN_TEST.png">

## Data Visualization (SOMs and tSNE)


<div class="row">
  <div class="column">
    <img src="figures/Picture2.png" width = "250">
  </div>
  <div class="column">
    <img img src="figures/Picture1.png" width="250">
  </div>
  <div class="column">
    <img src="figures/Picture3.png" width="250">
  </div>
</div>


