# FFNN_InitFitPredict
Implementation of Feed-Forward Neural Networks (FFNNs) training schemes for supervised learning (regression, binary classification, multi-class classification). It combines the ease-of-use of the scikit-learn API and well-chosen default parameters with the efficiency and versatility of the pytorch library: 
- The classes "FeedForwardRegressor" and "FeedForwardClassifier" are used through the functions "\_\_init\_\_()", "fit(X, y)", "predict(X)" where X is a numpy tensor (tabular dataset or images for eg.) and y a numpy vector. Functions "score(X, y)", "predict_proba(X)" and "decision_function(X)" are also implemented.
- Training is done using pytorch modules and automatic differentiation (default architecture corresponds to a simple MultiLayerPerceptron for tabular datasets, but any pytorch module corresponding to a FFNN architecture can be used). 

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
- parameters_examples.py provides hyper-parameters combinations corresponding to relevant learning schemes and networks shapes for small and large tabular datasets in both regression and classification. It also includes provides appropriate parameters for Marginal Contrastive Discrimination (see how to cite).
- requirements.txt lists required libraries.

# Requirements: numpy, scikit-learn, scipy, pytorch

```bash
conda create --name FFNN_demo python=3.7
conda activate FFNN_demo
conda install ipykernel scikit-learn=0.23.2 scipy=1.6.2 -c anaconda -y
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y 
conda deactivate
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
- encapsulating state back-up, state loading, partial\_fit and warm-start,
- encapsulating weight-freezing, refitting and transfer learning,
- encapsulating domain-specific data-augmentation schemes (eg.: image rotations).
