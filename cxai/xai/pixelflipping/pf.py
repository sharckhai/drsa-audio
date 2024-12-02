from typing import Tuple, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import zennit
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.types import Linear, Convolution, Activation
from zennit.rules import Epsilon, ZPlus, Norm, Pass, WSquare, Gamma, Flat, AlphaBeta, BasicHook
from zennit.composites import SpecialFirstLayerMapComposite, NameMapComposite, NameLayerMapComposite, Composite

from cxai.xai.explain.attribute import compute_relevances
from cxai.xai.pixelflipping.core import Flipper
from cxai.utils.visualization import plot_aupcs


rule_mapper = {
    'epsilon': Epsilon,
    'gamma': Gamma,
    'zplus': ZPlus,
    'alphabeta': AlphaBeta,
    'flat': Flat,
    'wsquare': WSquare,
    'pass': Pass,
    'norm': Norm,
}

class PixelFlipping:
    """Performs pixel flipping experiments.

    This class coordinates several pixel flipping experiments, given a NN model, an input batch, and 
    a configuration grid which spacifies differetn LRP configurations to attribute relevances down to the inputs.

    Attributes:
        TODO
    """

    def __init__(
        self,
        model: nn.Sequential,
        input_batch: torch.Tensor,
        perturbation_size: int = 8,
        perturbation_mode: str = 'constant',
        num_classes: int = 10,
        data_normaliaztion: str = 'normalized',
        device: torch.device = torch.device('cpu'),
    ) -> None:
        """Init the pixelflipping class.

        NOTE: Classes have to represented by the same number of samples and have to be ordered in consecutive, 
        increasing order by class index!
        
        Args:
            model (nn.Sequential): Moidel to perform the evaluation on.
            input_batch (torch.Tensor): Batched inputs (mel-specs) with shape (batch, channel, heightm width). 
            perturbation_size (int, optional): Size of the patches that are being flipped.
            perturbation_mode (str, optional): Perturbation mode. Options: ['constant', 'inpainting'].
            num_classes (int, optional): Number of classes present in the data.
            data_normalization (str, optional): Normalize inpainted image regions.
            device (str | torch.device, optional): Device.
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
        self.pixel_flipper = Flipper(
            perturbation_size=perturbation_size, 
            perturbation_mode=perturbation_mode, 
            data_normaliaztion=data_normaliaztion, 
            device=device
        )

    def __call__(
        self,
        configuration_grid: list[dict],
        stabilizers: dict = None,
        canonizer: zennit.canonizers = SequentialMergeBatchNorm(),
        scaled_gamma: bool = False,
        composites: list = None,
        plot: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], ]:
        """Loops over the configurations and executes pixel flipping for each configuration.        
        
        Args:
            TODO
        
        Returns:
            tuple: A tuple containing:
                - aupc_scores (Dict[str, np.ndarray]): AUPC scores per configuration.
                - averaged_pertubed_prediction_logits (Dict[str, np.ndarray]): Averaged logit scores per perturbation step per configuration.
                - flips_per_perturbation_step (np.ndarray): List that defines the number of patches flipped in each perturbatuion step.
                - heatmaps (Dict[str, torch.Tensor]): Heatmaps computed with given condiguration.
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
    
    def _get_scaled_composite_toy(self, lrp_configuration: Dict[str, Any]) -> Composite:
        """Construct a zennit composite for relevance attribution in the current configuration.

        NOTE: This function defines a custom name_map for specific experiments.

        Args:
            lrp_configuration (Dict[str, Any]): Defines which LRP rules are being used fpr relevance attribution.

        Returns:
            composite (Composite): Zennit Composite for relevance redistribution.
        """
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

        return NameMapComposite(
            name_map=name_map,
            canonizers=[self.canonizer],
        )    
    
    def _get_scaled_composite_peak4(self, lrp_configuration: Dict[str, Any]) -> Composite:
        """Construct a zennit composite for relevance attribution in the current configuration.

        NOTE: This function defines a custom name_map for specific experiments.

        Args:
            lrp_configuration (Dict[str, Any]): Defines which LRP rules are being used fpr relevance attribution.

        Returns:
            composite (Composite): Zennit Composite for relevance redistribution.
        """
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

        return NameMapComposite(
            name_map=name_map,
            canonizers=[self.canonizer],
        )

    def _get_composite(self, lrp_configuration: Dict[str, Any]) -> Composite:
        """Construct a zennit composite for relevance attribution in the current configuration.

        Args:
            lrp_configuration (Dict[str, Any]): Defines which LRP rules are being used fpr relevance attribution.

        Returns:
            composite (Composite): Zennit Composite for relevance redistribution.
        """
        assert 'convolutional' in lrp_configuration, 'rule for convolutional layers has to be passed'
        assert 'dense' in lrp_configuration, 'rule for dense layers has to be passed'
        assert 'first_layer' in lrp_configuration, 'rule for first layer layers has to be passed'

        # build layer map
        # get convolutional rule from layer_mapping
        conv_rule = self._get_rule('convolutional', lrp_configuration)
        dense_rule = self._get_rule('dense', lrp_configuration)
        first_layer_rule = self._get_rule('first_layer', lrp_configuration)

        if 'name_map' in list(lrp_configuration.keys()):
            # build composite
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

    def _get_name_map(self, lrp_configuration) -> Dict[str, BasicHook]:
        """Create a name map for some configuration.
        
        Args:
        lrp_configuration (Dict[str, Any]): Defines which LRP rules are being used fpr relevance attribution.

        Returns:
            name_map (Dict[str, BasicHook]): Maps zennit (LRP) rules to model layers.
        """
        name_map = []
        # add rule mapping for first layer
        for key in lrp_configuration:
            if key != 'convolutional' and key != 'dense' and key != 'first_layer':
                name_map.append(([key], self._get_rule(layertype=key, lrp_configuration=lrp_configuration)))
        return name_map

    def _get_rule(self, layertype: str, lrp_configuration: Dict[str, Any]) -> BasicHook:
        """Constructs a LRP rule from the givein configuration.
        
        Args:
            layertype (str): Can be layertype of special layer name for rule mapping.
            lrp_configuration (Dict[str, Any]): Defines which LRP rules are being used fpr relevance attribution.

        Returns:
            rule (BasicHook): Zennit LRP rule.
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
        """Creates string as configuration identifier."""
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
        return conf

    def plot_aupcs(self, flips_per_perturbation_step, title='EpsGammaWSquare'):
        """Creates an AUPC plot.
        
        Outputs a line plot displaying the averaged AUPC scores in each perturbation step for each configuration.
        """
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