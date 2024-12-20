import numpy as np
import torch
import cv2


class Flipper:
    """Core for pixelflipping experiments.

    Given a model, an input batch, and precalculated relevance heatmaps for the provided inputs, 
    this class flips performs patch flipping on all inputs while simultaneously tracking model outputs.
    
    Attributes:
        perturbation_size (int): Size of the patches that are being flipped.
        perturbation_mode (str): Perturbation mode. Options: ['constant', 'inpainting'].
        data_normaliaztion (str): Data normalization technique. Used for inpainting case.
        device (str | torch.device): Device.
    """

    def __init__(
        self, 
        perturbation_size: int = 16, 
        perturbation_mode: str = 'constant',
        data_normaliaztion: str = 'normalized',
        device: str | torch.device = torch.device('cpu')
    ) -> None:
        """Init core flipper.

        This class is repeatedly called from PixelFlipping in pf.py, and computes the pixel flipping metric
        with the provided revealnce maps obtained according to a given LRP configuration.
        
        Args:
            perturbation_size (int, optional): Size of the patches that are being flipped.
            perturbation_mode (str, optional): Perturbation mode. Options: ['constant', 'inpainting'].
            data_normaliaztion (str, optional): Data normalization technique. Used for inpainting case.
            device (str | torch.device, optional): Device.
        """
        self.device = device
        self.perturbation_size = perturbation_size
        self.perturbation_mode = perturbation_mode
        self.data_normaliaztion = data_normaliaztion

    def __call__(
        self, 
        forward_func: callable,
        input_batch: torch.Tensor, 
        R: torch.Tensor,
        flipping_mode: str = None,
    ) -> np.ndarray:
        """Executes the pixel flipping for a specific configuration.r

        NOTE: This class works on batched inputs and respective relvance maps. Each class has to be 
        represented by the same number of samples. Samples have to be sorted in consecutive order (from class 0-N)!

        Args:
            forward_func (callable): Represents the model and ouputs prediction scores (Logits).
            input (torch.tensor): Batched inputs. Shape has to be: (batch, channel, height, width).
            R (torch.Tensor): Batched relevance matrices. Has to be in same order as input.
            flipping_mode (str, optional): Flip patches at random or according to relevances.

        Returns:
            tuple: A tuple containing:
                - aupc_per_instance (np.ndarray): AUPC score per instance.
                - perturbed_predictions (np.ndarray): Averaged logit score in each perturbation step across all instances in the batch.
                - flips_per_perturbation_step (List[int]): Flips per perturbation step.
        """
        # input_batch [batch, c, height, width], R shape [batch, n_concepts, c, height, width]
        # NOTE: In case of standard attribution set n_concepts = 1
        if flipping_mode != 'random':
            if R.dim() < 5: R.unsqueeze(1) # in case of standard attribution, add virtual concept dimension with n_concepts == 1

        # CASE Concept FLipping
        self.n_concepts = R.size(1) if R is not None else 1
        self.forward_func = forward_func
        self.flipping_mode = flipping_mode
        self.batch_size, self.num_channels, self.height, self.width = input_batch.size()
        self.num_patches = (self.height // self.perturbation_size) * (self.width // self.perturbation_size)

        # generate list of patch indices to flip in each step
        if self.flipping_mode == 'random':
            # generate an array of patch idcs to flip for each sample in batch
            sorted_patch_indices_by_relevance = torch.stack(
                [torch.randperm(self.num_patches, device=self.device) for _ in range(self.batch_size)]
            )
            self.sorted_patch_indices_by_relevance = sorted_patch_indices_by_relevance.reshape(self.batch_size, 1, -1)
        else:
            self.R = R.to(self.device)
            self.sorted_patch_indices_by_relevance = self._generate_patches()

        # get first prediction score of original input
        pertubed_input_batch        = input_batch.clone().detach().requires_grad_(False)
        pertubed_inputs             = [pertubed_input_batch.detach().cpu().numpy()] # for inspection purposes
        perturbed_predictions        = [self._get_prediction_score(pertubed_input_batch)]
        flips_per_perturbation_step = np.array([0])
        flipped_patches             = 0
        masks_per_step              = []
        # generate raw mask filled with ones (bool)
        masks = torch.ones(
            (self.batch_size, self.num_channels, self.height, self.width), 
            dtype=torch.int16, 
            device=self.device
        ).requires_grad_(False)

        print('Flipping patches...')

        # loop until all pixels are flipped
        while flipped_patches < self.num_patches:

            # each perturbation step we flip step**2 patches (last step we flip all remaining pathces)
            if len(flips_per_perturbation_step)**2 < self.num_patches - flipped_patches:
                patches_to_flip = len(flips_per_perturbation_step)**2
            else:
                patches_to_flip = self.num_patches - flipped_patches

            # get mask for flipping operation
            masks *= self._get_flipping_mask(patches_to_flip, flipped_patches)
            masks_per_step.append(masks.detach().cpu().numpy())

            # flip 
            pertubed_input_batch = self._flip(pertubed_input_batch, masks).to(self.device)
            pertubed_inputs.append(pertubed_input_batch.detach().cpu().numpy())

            # [batch]
            logits = self._get_prediction_score(pertubed_input_batch)
            perturbed_predictions.append(logits)

            flips_per_perturbation_step = np.append(flips_per_perturbation_step, patches_to_flip)
            flipped_patches += patches_to_flip

        # [perturbation_steps, num_samples]
        perturbed_predictions = np.stack(perturbed_predictions, axis=0)#.reshape(len(flips_per_perturbation_step), self.n_classes, -1)
        masks_per_step = np.stack(masks_per_step, axis=0)

        # calculate the area under the pixel flipping curve for each instance
        aupc_per_instance = self._calculate_aupc(perturbed_predictions, flips_per_perturbation_step)
        perturbed_predictions = perturbed_predictions.mean(axis=1)
        return aupc_per_instance, perturbed_predictions, flips_per_perturbation_step#, pertubed_inputs

    def _flip(self, pertubed_inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Flip inputs as defined in the prebuild masks.
        
        Args:
            perturbed_inputs (torch.Tensor): Current inputs batches during som eperturbation iteration.
            masks (torch.Tensor): Perturbation masks.

        Returns:
            perturbed_inputs (torch.Tensor): Perturbed inputs.
        """
        # copy masks and inputs
        pertubed_inputs_copy = pertubed_inputs.clone().detach()
        masks_copy = masks
        
        if self.perturbation_mode == 'constant':
            return pertubed_inputs_copy * masks_copy #+ torch.abs(masks_copy-1)*(-1)  # mask pathces with zeros
        
        elif self.perturbation_mode == 'inpainting':
            # cv2.inpaint expects the patches to inpaint in mask filled with ones else zero
            masks = torch.abs(masks_copy - 1)
            inpainted_images = []
            
            # loop over images and masks to generate patch inpainting
            for image, mask in zip(pertubed_inputs_copy.view(self.batch_size, self.height, self.width, self.num_channels), 
                                   masks.view(self.batch_size, self.height, self.width, self.num_channels)):

                image = image.squeeze(0).cpu().numpy()
                mask = mask.squeeze(0).cpu().numpy().astype(int)

                inpainted_image = cv2.inpaint(image, mask.astype(np.uint8), inpaintRadius=self.perturbation_size//2, flags=cv2.INPAINT_TELEA)

                if self.data_normaliaztion == 'normalized':
                    # normalize the inpainted image patch seperate from original image
                    normalized_patch = ((inpainted_image.reshape(mask.shape) - np.min(inpainted_image)) / (np.max(inpainted_image) - np.min(inpainted_image))) * mask

                    # set patch in original image to 0 and add the normalized inpainted image in which inly the patch remains
                    inpainted_image = image * np.abs(mask - 1) + normalized_patch

                if self.data_normaliaztion == 'min':
                    # normalize the inpainted image patch seperate from original image
                    normalized_patch = (2*((inpainted_image.reshape(mask.shape) - np.min(inpainted_image)) / (np.max(inpainted_image) - np.min(inpainted_image)))-1) * mask

                    # set patch in original image to 0 and add the normalized inpainted image in which inly the patch remains
                    inpainted_image = image * np.abs(mask - 1) + normalized_patch

                inpainted_images.extend(torch.tensor(inpainted_image[None], dtype=torch.float32, device=self.device).requires_grad_(False))

            return torch.stack(inpainted_images, dim=0).reshape(self.batch_size, self.num_channels, self.height, self.width).to(self.device)
        else:
            raise ValueError('Provided perturbation mode not available. Possible perturbation modes are "constant" and "inpainting".')

    def _generate_patches(self) -> torch.Tensor:
        """Generates a list of patch indices that have to be flipped during the flipping operation.

        Patches are numbered from top left to bpttom right.
        
        Returns:
            sorted_patch_indices_by_relevance (torch.Tensor): 
        """

        # shape [batch, n_concepts, height, width] (get rid of channel dimension without squeezing n_concept dimension if n_concept == 1)
        R_ = self.R.view(self.batch_size, self.n_concepts, self.height, self.width).clone().detach()

        ###### applying relu????
        R_ = torch.clamp(R_, min=0)

        # shape [batch, n_concepts, patches, perturbation_size, perturbation_size]
        R_patched = R_.unfold(-1, self.perturbation_size,self.perturbation_size)\
                        .unfold(-3, self.perturbation_size, self.perturbation_size)\
                            .reshape(self.batch_size, self.n_concepts, -1, self.perturbation_size, self.perturbation_size)

        patch_relevances = R_patched.sum(dim=(-2,-1)).view(self.batch_size, self.n_concepts, -1)

        # shape [batch, n_concepts, num_patches]
        sorted_patch_indices_by_relevance = torch.argsort(patch_relevances, descending=True, dim=-1)#.to(self.device)
        return sorted_patch_indices_by_relevance

    def _get_flipping_mask(self, patches_to_flip: int, flipped_patches: int) -> torch.Tensor:
        """Function to build flipping mask. Depending on perturbation step, the mask includes several patches.
        
        Args:
            patches_to_flip (int): Precalculated number of patches that have to be flipped during this perturbation step.
            flipped_patches (int): How many patches have already been flipped.

        Returns:
            mask (torch.Tensor): Tensor of ones with the size of input. NOTE: Patches to be pertub are filled with zeros.
        
        NOTE: Concept Flipping
        We could either reshape the array of patches to flip, make them unique (unification) and mask.
        Problem then is, that batched patch-flips aren't possible anymore, since it is possible that each 
        sample has a different number of patches that have to be flipped (several concepts can have 
        overlapping patches with highest relevance).
        """        
        # get list of patch indices that have to be flipped
        patch_idcs = self.sorted_patch_indices_by_relevance[..., flipped_patches:flipped_patches+patches_to_flip]
        # FOR NOW: append patches to flip from all 4 concepts
        patch_idcs = patch_idcs.transpose(-2,-1).reshape(self.batch_size, -1)

        # generate raw mask filled with ones (bool)
        masks = torch.ones(
            (self.batch_size, self.num_channels, self.height, self.width),
            dtype=torch.int16, 
            device=self.device
        ).requires_grad_(False)
            
        # get startpoints of the patches to mask for every datapoint
        x_locations = patch_idcs // int(self.width/self.perturbation_size) * self.perturbation_size
        y_locations = patch_idcs % int(self.width/self.perturbation_size) * self.perturbation_size

        # this loop masks single patches
        for i in range(patches_to_flip*self.n_concepts):

            # get location of patch for single datapoints
            x_locations_ = x_locations[:,i]
            y_locations_ = y_locations[:,i]

            # Calculate indices for broadcasting
            batch_indices = torch.arange(self.batch_size, device=self.device).view(-1, 1, 1).expand(-1, self.perturbation_size, self.perturbation_size)
            x_indices = x_locations_.view(-1, 1, 1) + torch.arange(self.perturbation_size, device=self.device).view(1, -1, 1)
            y_indices = y_locations_.view(-1, 1, 1) + torch.arange(self.perturbation_size, device=self.device).view(1, 1, -1)

            # Update the mask to False (0) at the specified locations
            masks[batch_indices, :, x_indices, y_indices] = 0
            
        # for debugging
        assert masks.size() == (self.batch_size, self.num_channels, self.height, self.width), 'mask size is wrong'
        assert (torch.abs(masks-1)).sum() <= self.perturbation_size**2 * patches_to_flip * self.batch_size * self.n_concepts, 'too many masked patches!'
        return masks

    def _get_prediction_score(self, pertubed_inputs: torch.Tensor) -> np.ndarray:
        """Calculates the averaged prediction score over all samples in the batch.
        
        Args:
            pertubed_inputs (torch.Tensor): Expects balanced batch containing instances in consecutive order.
        
        Returns:
            pred_scores (np.ndarray): Prediciton scores of perturbed input samples.
        """
        # obtain model outputs, shape [batch, n_classes]
        out = self.forward_func(pertubed_inputs)
        n_classes = out.size(1)
        self.n_classes = n_classes

        # only keep class specific output, shape ->[batch, 1]
        class_specific_outputs = out[
            np.arange(self.batch_size), 
            np.repeat(np.arange(n_classes), 
            self.batch_size//n_classes if self.batch_size//n_classes > 0 else 1)
        ]
        # apply relu, reshape and get avergae score per class
        pred_scores = torch.clamp(class_specific_outputs, min=0).squeeze()
        return pred_scores.detach().cpu().numpy()

    def _calculate_aupc(
        self, 
        pertubed_predictions: np.ndarray, 
        flips_per_perturbation_step: int
    ) -> np.ndarray:
        """Calculates the AUPC-score with self.flips_per_perturbation_step and self.perturbed_input.
        
        Args:
            pertubed_predictions (np.ndarray)
            flips_per_perturbation_step (int):

        Returns:
            aupc_score (np.arrray): AUPC score for every instance, array has shape [num_classes, samples_per_class]
        """
        # pertubed_predictions: shape [perturbation_steps, batch]
        frac = (pertubed_predictions[:-1] - pertubed_predictions[1:]) / 2
        weights = np.cumsum(flips_per_perturbation_step[1:]) / flips_per_perturbation_step[1:].sum()
        aupc_per_instance = (weights[None].T*frac).sum(axis=0)

        # aupcs per instance sorted by classes, shape [num_classes, samples_per_class]
        aupc_per_class = aupc_per_instance.reshape(self.n_classes, -1)
        return aupc_per_class
    


