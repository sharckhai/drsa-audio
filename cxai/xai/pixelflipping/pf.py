import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import zennit
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from zennit.types import Linear, Convolution, Activation, BatchNorm
from zennit.rules import Epsilon, ZPlus, Norm, Pass, WSquare, Gamma, Flat, AlphaBeta, ZBox
from zennit.composites import SpecialFirstLayerMapComposite, NameMapComposite, NameLayerMapComposite, LayerMapComposite, MixedComposite
from zennit.core import Stabilizer
from zennit.core import BasicHook, Hook, stabilize
from zennit.rules import NoMod
import copy

from cxai.xai.explain.attribute import compute_relevances
from cxai.xai.pixelflipping.core import Flipper
from cxai.utils.visualization import plot_aupcs

"""from attribute import compute_relevances
from core import Flipper
from custom_xai import MTMRule
from utils import HiddenPrints"""


# define rula mapper
rule_mapper = {
    'epsilon': Epsilon,
    'gamma': Gamma,
    'zplus': ZPlus,
    'alphabeta': AlphaBeta,
    'flat': Flat,
    'wsquare': WSquare,
    'pass': Pass,
    'norm': Norm,
    #'mtm': MTMRule,
}


# TODO: return one representative example PER CONFIGURATION, concat them and return to plot relevance heatmaps afterwards



class PixelFlipping:

    def __init__(self,
                 model: nn.Sequential,
                 input_batch: torch.Tensor,
                 perturbation_size: int = 8,
                 perturbation_mode: str = 'constant',
                 num_classes: int = 10,
                 #modified: bool = False,
                 data_normaliaztion: str = 'normalized',
                 device: torch.device = torch.device('cpu'),
                 ) -> None:
        """
        lrp conf
        dict with first layer, dense etc and values
        mapper to map rules to layers?  OPTIONAL

        Parameters:
        -----------
        model: nn.Sequential
            model to perform attribution technique on
        input_batch: torch.Tensor
            Batched inputs (mel-specs) with shape (batch, channel, heightm width). 
            NOTE: Classes have to represented by the same number of samples and have to be ordered in consecutive, increasing order by class index!
        forward_func: callable
            Represents the model and ouputs prediction scores (Logits).
        pixel_flipper: PixelFlipping
            Instance of PixelFlipping class which performs the pixel flipping for one specific attribution configuration.
        modified_model: bool
            Boolean to indicate if model has previously been transformed to have a reverse logsumexp layer at the end.
        
        Returns:
        --------
        bla
        
        """
        self.device = device
        self.input_batch = input_batch.to(self.device)
        self.num_classes = num_classes
        self.samples_per_class = self.input_batch.size(0) // self.num_classes
        self.model = model.to(self.device)
        self.model.eval()

        # define forward func callable to get predictions after pixel flipping
        def forward_func(input_nchw): return model(input_nchw)
        self.forward_func = forward_func

        # create pixel flipper
        self.pixel_flipper = Flipper(perturbation_size=perturbation_size, perturbation_mode=perturbation_mode, 
                                     data_normaliaztion=data_normaliaztion, device=device)


    def __call__(self,
                 configuration_grid: list[dict],
                 stabilizers: dict = None,
                 canonizer: zennit.canonizers = SequentialMergeBatchNorm(),
                 scaled_gamma: bool = False,
                 composites: list = None,
                 plot: bool = True,
                 ):
        
        """
        Loops over the configurations and perform pixel flipping for each configuration.

        Parameters:
        -----------
        configuration grid: list[dict[Tuple]]
            A list of dicts with values for lrp rules. For name of rule refer to rule_mapper!
            conf_grid = [
                {'convloutional': ('gamma', 0.2), 'dense': ('epsilon', 1e-3), 'first_layer': ('flat',), 'name_map': (name_map,)},]

            
                
        composites: list[zennit.composites]
            Composites for eaech configuration. If provided, only provide names for each configuration in configuration grid.


        Returns:
        --------

        
        """
        # configurations for lrp
        self.canonizer = canonizer
        self.stabilizers = stabilizers
        self.aupc_scores = {}
        self.averaged_pertubed_prediction_logits = {}
        self.pertubed_inputs = {}
        self.heatmaps = {}

        # iterate over configurations
        for i, lrp_configuration in enumerate(configuration_grid):

            # get name of configuration for logging and printing
            configuration_name = self._get_configuration_name(lrp_configuration)

            print(f'Starting pixel flipping algorithm for configuration {lrp_configuration}')
            print(f'Key: {configuration_name}')
            print('-'*5)

            # create composite
            if composites:
                composite = composites[i]
            else:
                if scaled_gamma == 'peak4':
                    composite = self._get_scaled_composite_peak4(lrp_configuration)
                elif scaled_gamma == 'toy':
                    composite = self._get_scaled_composite_toy(lrp_configuration)
                elif scaled_gamma == 'toynone':
                    composite = self._get_scaled_composite_toy(lrp_configuration)
                else:
                    composite = self._get_composite(lrp_configuration)

            # get relevances
            relevances = []
            # bacthed attribution that gpu doesnt run out of memory
            for i in range(self.num_classes):
                relevances.append(compute_relevances(self.model, self.input_batch[i*self.samples_per_class:(i+1)*self.samples_per_class], composite=composite, class_idx=i))
            relevances = torch.concat(relevances, axis=0)
            print(relevances.shape)
            self.heatmaps[configuration_name] = relevances
            
            # perform pixel flipping
            aupc_per_instance, pertubed_predictions, flips_per_perturbation_step = self.pixel_flipper(
                forward_func=self.forward_func,
                input_batch=self.input_batch.clone().detach(),
                R=relevances,
            )

            self.aupc_scores[configuration_name] = aupc_per_instance
            self.averaged_pertubed_prediction_logits[configuration_name] = pertubed_predictions
            #self.pertubed_inputs[configuration_name] = pertubed_inputs

            # print configuration and aupc
            print('AUPC-Score: %10.5f' % aupc_per_instance.mean())
            print('-'*10)

        if plot:
            plot_aupcs(self.aupc_scores, self.averaged_pertubed_prediction_logits, flips_per_perturbation_step)
            
        return self.aupc_scores, self.averaged_pertubed_prediction_logits, flips_per_perturbation_step, self.heatmaps
    

    def _get_scaled_composite_toy(self, lrp_configuration):
        # params for lrp
        gamma = lrp_configuration['convolutional'][-1]
        stab1 = 1e-7
        stab2 = 1e-7
        eps = lrp_configuration['dense'][-1]

        name_map = [
            # block 1
            (['features.0'], Flat(stabilizer=stab1) if lrp_configuration['first_layer'][0] == 'flat' else WSquare(stabilizer=stab1)),
            # block 2
            (['features.3'], Gamma(gamma=gamma, stabilizer=stab2)),
            # block 3
            (['features.6'], Gamma(gamma=gamma, stabilizer=stab2)),
            # last conv block
            (['features.9'], Gamma(gamma=gamma/2, stabilizer=stab2)),

            (['features.12'], Gamma(gamma=gamma/4, stabilizer=stab2)),
            # fc block
            (['classifier.0'], Epsilon(epsilon=eps)),
            (['classifier.2'], Epsilon(epsilon=eps)),
            (['classifier.4'], Epsilon(epsilon=eps)),
        ]


        composite_name_map = NameMapComposite(
            name_map=name_map,
            canonizers=[self.canonizer],
        )

        return composite_name_map
    
    
    def _get_scaled_composite_peak4(self, lrp_configuration):
        # params for lrp
        gamma = lrp_configuration['convolutional'][-1]
        stab1 = 1e-7
        stab2 = 1e-7
        eps = lrp_configuration['dense'][-1]

        name_map = [
            # block 1
            (['features.0'], Flat(stabilizer=stab1) if lrp_configuration['first_layer'][0] == 'flat' else WSquare(stabilizer=stab1)),
            # block 2
            (['features.3'], Gamma(gamma=gamma, stabilizer=stab2)),
            # block 3
            (['features.6'], Gamma(gamma=gamma, stabilizer=stab2)),
            # last conv block
            (['features.9'], Gamma(gamma=gamma/2, stabilizer=stab2)),

            (['features.12'], Gamma(gamma=gamma/4, stabilizer=stab2)),
            # fc block
            (['classifier.0'], Epsilon(epsilon=eps)),
            (['classifier.3'], Epsilon(epsilon=eps)),
            (['classifier.6'], Epsilon(epsilon=eps)),
        ]


        composite_name_map = NameMapComposite(
            name_map=name_map,
            canonizers=[self.canonizer],
        )

        return composite_name_map
    
    def _get_scaled_composite_toyNone(self, lrp_configuration):
        # params for lrp
        gamma = lrp_configuration['convolutional'][-1]
        stab1 = 1e-7
        stab2 = 1e-7
        eps = lrp_configuration['dense'][-1]

        name_map = [
            # block 1
            (['features.0'], Flat(stabilizer=stab1) if lrp_configuration['first_layer'][0] == 'flat' else WSquare(stabilizer=stab1)),
            # block 2
            (['features.3'], Gamma(gamma=gamma, stabilizer=stab2)),
            # block 3
            (['features.6'], Gamma(gamma=gamma/2, stabilizer=stab2)),
            # last conv block
            (['features.9'], Gamma(gamma=gamma/2, stabilizer=stab2)),

            (['features.12'], Gamma(gamma=gamma/4, stabilizer=stab2)),
            # fc block
            (['classifier.0'], Epsilon(epsilon=eps)),
            (['classifier.2'], Epsilon(epsilon=eps)),
            (['classifier.4'], Epsilon(epsilon=eps)),
        ]


        composite_name_map = NameMapComposite(
            name_map=name_map,
            canonizers=[self.canonizer],
        )

        return composite_name_map


    def _get_composite(self, lrp_configuration):

        assert 'convolutional' in lrp_configuration, 'rule for convolutional layers has to be passed'
        assert 'dense' in lrp_configuration, 'rule for dense layers has to be passed'
        assert 'first_layer' in lrp_configuration, 'rule for first layer layers has to be passed'

        # build layer map
        # get convolutional rule from layer_mapping
        conv_rule = self._get_rule('convolutional', lrp_configuration)
        dense_rule = self._get_rule('dense', lrp_configuration)
        first_layer_rule = self._get_rule('first_layer', lrp_configuration)

        if 'name_map' in list(lrp_configuration.keys()):
            
            composite = NameLayerMapComposite(
                name_map=lrp_configuration['name_map'],
                layer_map=[
                    (Activation, Pass()),
                    (Convolution, conv_rule),
                    (Linear, dense_rule)
                ],
                canonizers=[self.canonizer],
            )
        else:
            # build composite
            composite = SpecialFirstLayerMapComposite(
                first_map=[(Convolution, first_layer_rule)],
                layer_map=[
                    (Activation, Pass()),
                    (Convolution, conv_rule),
                    (Linear, dense_rule),
                ],
                canonizers=[self.canonizer],
            )

        return composite



    def _get_name_map(self, lrp_configuration):
        
        # construct name map if necessary
        name_map = []

        # add rule mapping for first layer
        for key in lrp_configuration:
            if key != 'convolutional' and key != 'dense' and key != 'first_layer':
                name_map.append(([key], self._get_rule(layertype=key, lrp_configuration=lrp_configuration)))
        
        return name_map



    def _get_rule(self, layertype, lrp_configuration):
        """
        returns a zennit rule

        Parameters:
        ---------
        layertype: str
            Can be layertype of special layer name for rule mapping.

        """

        # check if rule is valid
        if lrp_configuration[layertype][0] not in rule_mapper:
            raise ValueError(f'Not a valid zennit rule for {layertype} layers!')
        
        # get rule name as string
        rule = lrp_configuration[layertype][0]
        
        # get stabilizer for ruletype if it was provided
        if self.stabilizers:
            if layertype in self.stabilizers:
                stabilizer = self.stabilizers[layertype]
        else:
            stabilizer = 1e-7

        # instanciate zennit rule
        if rule == 'gamma':
            return Gamma(gamma=lrp_configuration[layertype][1], stabilizer=stabilizer)
        elif rule == 'epsilon':
            return Epsilon(epsilon=lrp_configuration[layertype][1])
        elif rule == 'alphabeta':
            alpha = lrp_configuration[layertype][1]
            # we infer beta from alpha because of the condition: aplhpa - beta = 1
            return AlphaBeta(alpha=alpha, beta=alpha-1, stabilizer=stabilizer)
        else:
            # all other rules just need stabilizer so we can get the rule from rule_mapper and only pass a stabilizer arg
            return rule_mapper[rule](stabilizer=stabilizer)


    def _get_configuration_name(self, lrp_configuration):
        # construct configuration string
        conf: str = ''
        for key in lrp_configuration:
            ruletype = lrp_configuration[key][0]
            if ruletype == 'alphabeta':
                conf += 'alpha_%3.1f_beta_%3.1f' % (lrp_configuration[key][1], (lrp_configuration[key][1]-1.0))
            elif ruletype == 'zplus':
                conf += ruletype + '_'
            elif key == 'first_layer':
                conf += ruletype
            elif key == 'name_map':
                continue
            else:
                conf += ruletype + '_' + str(lrp_configuration[key][1]) + '_' 
        return conf#[:-1]
    


    def plot_aupcs(self, flips_per_perturbation_step, title='EpsGammaWSquare'):

        for key in self.aupc_scores:
            # get data for each configuration 
            x_flipped_patches = np.cumsum(np.array(flips_per_perturbation_step)) / np.array(flips_per_perturbation_step).sum() * 100
            y_prediction_logits = np.array(self.averaged_pertubed_prediction_logits[key])#.flatten()

            if key[:5] == 'alpha':
                label = r'$\alpha$' + ' = %3.1f, ' + r'$\beta$' + ' = %3.1f, AUPC: %.3f' % (float(key[6:9]), float(key[6:9])-1, self.aupc_scores[key].mean())
            elif key[:5] == 'zplus':
                label = 'zplus, AUPC: %.3f' % (self.aupc_scores[key].mean())
            else:
                label = r'$\gamma$' + ' = %.2f, AUPC: %.3f' % (float(str.split(key, '_')[1]), self.aupc_scores[key].mean())
            #plt.plot(x_flipped_patches, y_prediction_logits, label='%15s AUPC: %6.2f' % (key, self.aupc_scores[key]), marker='o')
            plt.plot(x_flipped_patches, y_prediction_logits, label=label, marker='o')
            plt.title(f'AUPC Curve {title}')
            plt.xlabel('Flipped patches [%]')
            plt.ylabel('Averaged target class logit')
            plt.grid(ls=':', alpha=0.5)
            plt.legend()

