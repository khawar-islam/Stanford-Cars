import cv2
import numpy as np
import torch


def top5_accuracy(output, target):
    """Computes the top-5 accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max((1, 5))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_5 = correct[:5].reshape(-1).float().sum(0, keepdim=True)
        res = correct_5.div_(batch_size).item()
        return res

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_map = None
        self.gradient = None

        # Register hooks
        self.hook_layers()

    def hook_layers(self):
        def hook_fn_forward(module, input, output):
            self.feature_map = output.detach()

        def hook_fn_backward(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        # Register forward and backward hooks on the last conv layer
        final_conv_layer = self.model.layer4[-1].conv2
        final_conv_layer.register_forward_hook(hook_fn_forward)
        final_conv_layer.register_backward_hook(hook_fn_backward)

    def compute_heatmap(self, input_image, class_id=None):
        # Forward
        logits = self.model(input_image)
        if class_id is None:
            class_id = logits.argmax(dim=1).item()
        target = torch.zeros_like(logits, requires_grad=True)  # Ensure gradients
        target = target.clone()
        target[0][class_id] = 1

        # Backward
        logits.backward(gradient=target, retain_graph=True)

        # Compute weights
        pooled_gradients = torch.mean(self.gradient, dim=[0, 2, 3])
        for i in range(pooled_gradients.size(0)):
            self.feature_map[0][i] *= pooled_gradients[i]

        # Average the channels of the feature_maps
        heatmap = torch.mean(self.feature_map, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max()

        return heatmap

    def overlay_heatmap_on_image(self, heatmap, original_image):
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        return superimposed_img