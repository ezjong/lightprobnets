import torch
import torch.nn as nn


class FGSM(nn.Module):
    def __init__(self, args, key="xe", epsilon=0.1):
        super(FGSM, self).__init__()
        self._args = args
        self._key = key
        self._epsilon = epsilon

    def __call__(self, model_and_loss, example_dict):

        input1 = example_dict["input1"]
        target1 = example_dict["target1"]

        input1v = input1.clone()
        input1v.requires_grad = True
        example_dict["input1"] = input1v

        model_and_loss.zero_grad()
        loss_dict, output_dict = model_and_loss(example_dict)

        loss_dict[self._key].backward()

        new_input = input1 + torch.sign(input1v.grad)*self._epsilon
        new_input = torch.clamp(new_input, 0.0, 1.0)

        example_dict["input1"] = new_input
        example_dict["target1"] = target1

        return example_dict
