import architectures
from copy import deepcopy

# GO-TO SETTING FOR QUICK RESULTS ON SMALL DATASETS:
# Multilayer Perceptron
mlp = {"lr_scheduler": "OneCycleLR",
       "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 200},
       "max_iter": 200,
       "learning_rate": 1e-3,  # useless with OneCycleLR
       "hidden_nn": architectures.DenseLayers,
       "hidden_params":  {"width": 512, "depth": 2, "dropout": 0.2, "batch_norm": True}
       }

# GO-TO SETTING FOR HIGH PERFORMANCES (CLASSIFICATION):
# Gated Linear Units
glu = {"lr_scheduler": "OneCycleLR",
       "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 500},
       "max_iter": 500,
       "learning_rate": 1e-3,  # useless with OneCycleLR
       "hidden_nn": architectures.GLULayers,
       "hidden_params":  {"width": 512, "depth": 3, "dropout": 0.2, "batch_norm": True}
       }

# GO-TO SETTING FOR HIGH PERFORMANCES (REGRESSION):
# Self-Normalizing Network + AdaCap
snn = {"adacap": True,
       "lr_scheduler": "OneCycleLR",
       "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 500},
       "max_iter": 500,
       "learning_rate": 1e-3,  # useless with OneCycleLR
       "hidden_nn": architectures.DenseLayers,
       "hidden_params":  {"width": 512, "depth": 2, "activation": "SELU", "initializer_params": {"gain_type": 'linear'}}
       }


# GO-TO SETTING FOR VERY LARGE DATASETS
# Multilayer Perceptron (always using batch-learning, running 20 epochs, not 20 iterations which might be very slow)
mlpbatch = {"lr_scheduler": "OneCycleLR",
            "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 200},
            "max_iter": 200,
            "epochs": True,
            "max_runtime": 3600,
            "learning_rate": 1e-3,  # useless with OneCycleLR
            "hidden_nn": architectures.DenseLayers,
            "hidden_params":  {"width": 512, "depth": 2, "dropout": 0.2, "batch_norm": True}
            }


# GO-TO SETTING FOR MCD (MarginalContrastiveDiscrimination):
# Multilayer Perceptron
mlpmcd = {"lr_scheduler": "OneCycleLR",
          "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 200},
          "max_iter": 200,
          "max_runtime": 120,
          "validation_fraction": False,
          "early_stopping_criterion": False,
          "learning_rate": 1e-3,  # useless with OneCycleLR
          "center_target": True,
          "rescale_target": False,
          "loss_imbalance": False,
          "hidden_nn": architectures.DenseLayers,
          "hidden_params":  {"width": 256,
                             "depth": 2,
                             "output": 1,
                             "dropout": 0.75,
                             "batch_norm": True,
                             }
          }

#############################################
# OTHER SETTINGS
#############################################
# ResNet
resnet = {"lr_scheduler": "OneCycleLR",
          "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 500},
          "max_iter": 500,
          "learning_rate": 1e-3,  # useless with OneCycleLR
          "hidden_nn": architectures.ResidualLayers,
          "hidden_params":  {"width": 512, "depth": 2, "block_depth": 2, "dropout": 0.2, "batch_norm": True}
          }

# Self-Normalizing Network without AdaCap
snn = {"lr_scheduler": "OneCycleLR",
       "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 500},
       "max_iter": 500,
       "learning_rate": 1e-3,  # useless with OneCycleLR
       "hidden_nn": architectures.DenseLayers,
       "hidden_params":  {"width": 512, "depth": 2, "activation": "SELU", "initializer_params": {"gain_type": 'linear'}}
       }

# Gated Linear Units but with ReLU function on gates
reglu = {"lr_scheduler": "OneCycleLR",
         "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 500},
         "max_iter": 500,
         "learning_rate": 1e-3,  # useless with OneCycleLR
         "hidden_nn": architectures.GLULayers,
         "hidden_params":  {"width": 512, "depth": 3, "gate_activation": "ReLU", "gate_initializer_params": {"gain_type": 'relu'}, "dropout": 0.2, "batch_norm": True}
         }

# Resnet with Gated Linear Units inside resblocks
resnetglu = {"lr_scheduler": "OneCycleLR",
             "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 500},
             "max_iter": 500,
             "learning_rate": 1e-3,  # useless with OneCycleLR
             "hidden_nn": architectures.ResidualGLULayers,
             "hidden_params":  {"width": 512, "depth": 2, "block_depth": 2, "dropout": 0.2, "batch_norm": True}
             }

# Resnet with Gated Linear Units inside resblocks with ReLU function on gates
resnetreglu = {"lr_scheduler": "OneCycleLR",
               "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 500},
               "max_iter": 500,
               "learning_rate": 1e-3,  # useless with OneCycleLR
               "hidden_nn": architectures.ResidualGLULayers,
               "hidden_params":  {"width": 512, "depth": 2, "block_depth": 2, "gate_activation": "ReLU", "gate_initializer_params": {"gain_type": 'relu'}, "dropout": 0.2, "batch_norm": True}
               }

# Multilayer Perceptron (deeper network)
mlpdeep = {"lr_scheduler": "OneCycleLR",
           "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 200},
           "max_iter": 200,
           "batch_size": 256,
           "validation_fraction": False,
           "max_runtime": 3600,
           "learning_rate": 1e-3,  # useless with OneCycleLR
           "hidden_nn": architectures.DenseLayers,
           "hidden_params":  {"width": 256, "depth": 6, "dropout": 0.2, "batch_norm": True}
           }

# Self-Normalizing Network without AdaCap (deeper network)
snndeep = {"lr_scheduler": "OneCycleLR",
           "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 500},
           "max_iter": 500,
           "learning_rate": 1e-3,  # useless with OneCycleLR
           "hidden_nn": architectures.DenseLayers,
           "hidden_params":  {"width": 512, "depth": 6, "activation": "SELU", "initializer_params": {"gain_type": 'linear'}}
           }

# Multilayer Perceptron (wider network)
mlpwide = {"lr_scheduler": "OneCycleLR",
           "lr_scheduler_params": {"max_lr": 1e-3, "total_steps": 1000},
           "max_iter": 1000,
           "max_runtime": 3600,
           "learning_rate": 1e-4,  # useless with OneCycleLR
           "hidden_nn": architectures.DenseLayers,
           "hidden_params":  {"width": 2048, "depth": 2, "dropout": 0.2, "batch_norm": True}
           }

# ResNet for large datasets (always using batch-learning, running 20 epochs, not 20 iterations which might be very slow)
resnetbatch = {"lr_scheduler": "OneCycleLR",
               "lr_scheduler_params": {"max_lr": 1e-2, "total_steps": 50},
               "max_iter": 50,
               "epochs": True,
               "max_runtime": 3600,
               "learning_rate": 1e-3,  # useless with OneCycleLR
               "hidden_nn": architectures.ResidualLayers,
               "hidden_params":  {"width": 512, "depth": 2, "block_depth": 2, "dropout": 0.2, "batch_norm": True}
               }
