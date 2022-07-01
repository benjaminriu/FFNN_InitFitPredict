# FFNN\_InitFitPredict
Implementation of Feed-Forward Neural Networks (FFNNs) training schemes for supervised learning (regression, binary classification, multi-class classification). It combines the ease-of-use of the scikit-learn API and well-chosen default parameters with the efficiency and versatility of the pytorch library: 
- The classes "FeedForwardRegressor" and "FeedForwardClassifier" are used through the functions "\_\_init\_\_()", "fit(X, y)", "predict(X)" where X is a numpy tensor (tabular observations or images for eg.) and y a numpy vector. Functions "score(X, y)", "predict_proba(X)" and "decision_function(X)" are also implemented.
- Training is done using pytorch modules and automatic differentiation (default architecture corresponds to a simple MultiLayerPerceptron for tabular datasets, but any pytorch module corresponding to a FFNN architecture can be used).

# Why would I ever use this instead of [skorch](https://github.com/skorch-dev/skorch)?
In many ways, this projects aims to do the same things as the [skorch](https://github.com/skorch-dev/skorch) package. The latter is a much bigger project, managed by a much larger team. If your goal is to design a state-of-the-art machine learning pipeline to win a kaggle, the implementation  of the AdaCap training method by FFNN\_InitFitPredict is the only thing which might justify using FFNN\_InitFitPredict instead of skorch (see "key missing features section below"). If you are an academic or practitionner who wants to quickly get a sense of how a neural network would perform on a dataset, in a project in which deep learning is not the main focus, there are a few reasons which justifies why (as of late 2022) you might want to use FFNN\_InitFitPredict instead of skorch or [fastai](https://github.com/fastai/fastai). Without going into details, the cost of entry for FFNN\_InitFitPredict is smaller than skorch or fastai since it does not introduce a new API and does not require changing the default parameters to outperform a basic random forest. 

| Project | scikit-learn API | Deep Learning | Default mode = Performance |
| ----------- | ----------- | ----------- | ----------- |
| FFNN\_InitFitPredict | Yes |  Yes | Yes |
| [skorch](https://github.com/skorch-dev/skorch) |  Yes |  Yes | No |
| [fastai](https://github.com/fastai/fastai) | No |  Yes | Yes |
| [catboost](https://github.com/catboost/catboost) | Yes | No | Yes |

# The implementation encapsulates
- compatibility with all pytorch optimizers, learning rate schedulers, modules and associated schemes (weight-decay, drop-out, batch-normalization, ...),
- batch-learning,
- target pre-processing (standardization for regression, label encoding for multi-class, class imbalance, ...),
- target dithering,
- train/validation splitting with stratification (using a custom stratification scheme for regression), 
- stopping strategies (early-stopping, convergence, divergence, max iterations, max epochs, max run-time),
- training metrics recording and verbosity management,
- train/eval mode switching (including output activation function switch for classification),
- gpu/cpu detection and switch,
- pytorch and cuda random seed fixation (including drop-out),
- gpu rapid access memory safeguards,
- AdaCap training scheme (see how to cite section),
- ...


# Repository content

- DemoFFNN.ipynb is a tutorial on how to use this repository.
- feed_forward_neural_network.py implements "FeedForwardRegressor" and "FeedForwardClassifier" and can be used as a stand-alone. 
- architectures.py contains pytorch modules corresponding to several FFNN architectures : Multilayer Perceptrons, Gated Linear Units, Resblocks and a basic ConvNet. Gated Linear Units and Resblocks are derived from the article [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (see how to cite section).
- parameters_examples.py provides hyper-parameters combinations corresponding to relevant learning schemes and networks shapes for small and large tabular datasets in both regression and classification. It also includes provides appropriate parameters for Marginal Contrastive Discrimination (see how to cite section).
- requirements.txt lists required libraries.

# Requirements: numpy, scikit-learn, scipy, pytorch

```bash
conda create --name FFNN_demo python=3.7
conda activate FFNN_demo
conda install ipykernel scikit-learn=0.23.2 scipy=1.6.2 -c anaconda -y
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y 
conda deactivate
wget 'https://github.com/benjaminriu/FFNN_InitFitPredict.git'
cd FFNN_InitFitPredict
```

### if no gpu available:
- just replace 
```bash
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
```
- with 
```bash
conda install pytorch=1.9.1 -c pytorch -y
```

# How to cite

If you use the Feed-Forward Neural Networks implementation in your research, you can cite it as follows:
```
@article{riu2022mcd,
  title={MCD : Marginal Contrastive Discrimination},
  author={Riu, Benjamin},
  journal={arXiv preprint : arXiv:2106.11959},
  year={2022}
}
```

If you use the AdaCap training scheme in your research, you can cite it as follows:
```
@article{meziani2022adacap,
  title={AdaCap: Adaptive Capacity control for Feed-Forward Neural Networks},
  author={Meziani, Katia and Lounici, Karim and Riu, Benjamin},
  journal={arXiv preprint arXiv:2205.07860},
  year={2022}
}
```

If you use the Gated Linear Units and Resblock architectures in your research, you can cite them as follows:
```
@article{gorishniy2021revisiting,
  title={Revisiting deep learning models for tabular data},
  author={Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={18932--18943},
  year={2021}
}
```

# To do next list and key missing features

- compatibility with pytorch DataLoaders in addition to numpy format,
- compatibility with custom pytorch loss functions,
- compatibility with custom stopping criterions,
- compatibility with more than one output architectures (eg: architectures yielding intermediary quantities during training)
- compatibility with custom (missing from pytorch) optimizers and lr\_schedulers 
- compatibility with custom stopping criterions
- encapsulating state back-up, state loading, partial\_fit and warm-start,
- encapsulating weight-freezing, refitting and transfer learning,
- encapsulating observation normalization and domain-specific data-augmentation schemes (eg.: image rotations),
- comment code.
