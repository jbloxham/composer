# Copyright 2021 MosaicML. All Rights Reserved.

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from composer.models.resnets import Bottleneck


def _sample_bernoulli(probability: torch.Tensor,
                      device_id: int,
                      module_id: int,
                      num_modules: int,
                      generator: torch.Generator,
                      use_same_depth_across_gpus: bool,
                      use_same_gpu_seed: bool = False):
    """Gets a sample from a Bernoulli distribution. Provides functionality to
       have different seeds across GPUs and to have the same set of seeds across GPUs.
    """

    if use_same_gpu_seed:
        sample = torch.bernoulli(probability)
        return sample

    # Get a separate generator for each GPU
    rand_int = torch.randint(low=2**17, high=2**27, size=(1,), dtype=torch.long, generator=generator)
    gpu_seed = (rand_int * (device_id + 1))

    if use_same_depth_across_gpus:
        gpu_generator = torch.Generator().manual_seed(gpu_seed.item())  #type: ignore
        layer_seed = (torch.randperm(num_modules, generator=gpu_generator)[module_id] + 1) * rand_int
    else:
        layer_seed = (module_id + 1) * gpu_seed
    layer_generator = torch.Generator().manual_seed(layer_seed.item())  # type: ignore

    sample = torch.bernoulli(probability, generator=layer_generator)
    return sample


class StochasticBottleneck(Bottleneck):
    """Stochastic ResNet Bottleneck block. This block has a probability of skipping
       the transformation section of the layer and scales the transformation section
       output by ``(1 - drop probability)`` during inference.

       Args:
            drop_rate: Probability of dropping the block. Must be between 0.0 and 1.0.
            module_id: The placement of the block within a network e.g. 0
                for the first layer in the network.
            module_count: The total number of blocks of this type in the network
            use_same_gpu_seed: Set to ``True`` to have the same layers dropped
                across GPUs when using multi-GPU training. Set to ``False`` to
                have each GPU drop a different set of layers. Only used
                with ``"block"`` stochastic method.
            use_same_depth_across_gpus: Set to ``True`` to have the same number
                of blocks dropped across GPUs. Should be set to ``True`` when
                ``drop_distribution`` is ``"uniform"`` and set to ``False``
                for ``"linear"``.
       """

    def __init__(self, drop_rate: float, module_id: int, module_count: int, use_same_gpu_seed: bool,
                 use_same_depth_across_gpus: bool, rand_generator: torch.Generator, **kwargs):
        super(StochasticBottleneck, self).__init__(**kwargs)
        self.drop_rate = torch.tensor(drop_rate)
        self.module_id = module_id
        self.module_count = module_count
        self.use_same_gpu_seed = use_same_gpu_seed
        self.use_same_depth_across_gpus = use_same_depth_across_gpus
        self.rand_generator = rand_generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        sample = _sample_bernoulli(probability=(1 - self.drop_rate),
                                   device_id=x.get_device(),
                                   module_id=self.module_id,
                                   num_modules=self.module_count,
                                   generator=self.rand_generator,
                                   use_same_gpu_seed=self.use_same_gpu_seed,
                                   use_same_depth_across_gpus=self.use_same_depth_across_gpus)

        if not self.training or sample:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if not self.training:
                out = out * (1 - self.drop_rate)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
        else:
            if self.downsample is not None:
                out = self.relu(self.downsample(x))
            else:
                out = identity
        return out

    @staticmethod
    def from_target_layer(module: Bottleneck,
                          module_index: int,
                          module_count: int,
                          drop_rate: float,
                          drop_distribution: str,
                          rand_generator: torch.Generator,
                          use_same_gpu_seed: bool = False):
        """Helper function to convert a ResNet bottleneck block into a stochastic block.
        """
        if drop_distribution == 'linear':
            drop_rate = ((module_index + 1) / module_count) * drop_rate
        use_same_depth_across_gpus = (drop_distribution == 'uniform')
        rand_generator = torch.Generator().manual_seed(
            rand_generator.initial_seed())  # copy the generator for each layer
        return StochasticBottleneck(drop_rate=drop_rate,
                                    module_id=module_index,
                                    module_count=module_count,
                                    use_same_depth_across_gpus=use_same_depth_across_gpus,
                                    use_same_gpu_seed=use_same_gpu_seed,
                                    rand_generator=rand_generator,
                                    inplanes=module.conv1.in_channels,
                                    planes=module.conv3.out_channels // module.expansion,
                                    stride=module.stride,
                                    downsample=module.downsample,
                                    groups=module.conv2.groups,
                                    dilation=module.conv2.dilation)


class StochasticGPT2Block(GPT2Block):

    def __init__(self, drop_rate: float, config):
        self.drop_rate = torch.tensor(drop_rate)
        super().__init__(config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):

        residual = hidden_states
        sample = torch.bernoulli(1 - self.drop_rate)
        if not self.training or sample:
            hidden_states = self.ln_1(hidden_states)
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            if not self.training:
                attn_output *= (1 - self.drop_rate)
            outputs = attn_outputs[1:]
            # residual connection
            hidden_states = attn_output + residual

            if encoder_hidden_states is not None:
                # add one self-attention block for cross-attention
                if not hasattr(self, "crossattention"):
                    raise ValueError(f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                                     "cross-attention layers by setting `config.add_cross_attention=True`")
                residual = hidden_states
                hidden_states = self.ln_cross_attn(hidden_states)
                cross_attn_outputs = self.crossattention(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attn_output = cross_attn_outputs[0]
                if not self.training:
                    attn_output *= (1 - self.drop_rate)
                # residual connection
                hidden_states = residual + attn_output
                outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            if not self.training:
                feed_forward_hidden_states *= (1 - self.drop_rate)
            # residual connection
            hidden_states = residual + feed_forward_hidden_states

        else:
            if use_cache:
                outputs = (None, None, None)
            else:
                outputs = (None,)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

    @staticmethod
    def from_target_layer(module: GPT2Block,
                          config,
                          module_index: int,
                          module_count: int,
                          drop_rate: float,
                          drop_distribution: str,
                          rand_generator: torch.Generator,
                          use_same_gpu_seed: bool = True):
        if drop_distribution == 'linear':
            drop_rate = ((module_index + 1) / module_count) * drop_rate
        return StochasticGPT2Block(drop_rate=drop_rate, config=config)
