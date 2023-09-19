from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import DecisionTransformerModel
from transformers.utils import ModelOutput

from src.agent import Agent
from src.agent import AgentInput
from src.utils.utils import to_one_hot


@dataclass
class DecisionTransformerOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`):
            Environment state predictions
        action_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`):
            Model action predictions
        return_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Predicted returns for each state
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    state_preds: torch.FloatTensor = None
    action_preds: torch.FloatTensor = None
    return_preds: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None


class FDTAgent(Agent, DecisionTransformerModel):
    def __init__(self, config, use_feedback=True, use_missions=True, use_rtg=False, override_use_rtg=False):
        DecisionTransformerModel.__init__(self, config)

        self.create_state_embedding_model()

        self.use_feedback = use_feedback
        self.use_missions = use_missions
        self.override_use_rtg = override_use_rtg
        self.use_rtg = use_rtg or (not self.use_feedback and not self.use_missions and not self.override_use_rtg)
        x = 1 + int(self.use_rtg) + int(self.use_feedback) + int(self.use_missions)


        # we override the parent class prediction functions so we can incorporate the feedback embeddings
        self.predict_state = nn.Linear(x * self.hidden_size, config.state_dim)
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(x * self.hidden_size, config.act_dim)]
                + ([nn.Tanh()] if config.action_tanh else [])
            )
        )
        self.predict_return = nn.Linear(x * self.hidden_size, 1)

    def create_state_embedding_model(self):
        # default to a linear state embedding - override this in child classes
        self.state_embedding_model = nn.Linear(self.config.state_dim, self.hidden_size)

    def _embed_state(self, states):
        return self.state_embedding_model(states)

    def _forward(self, input: AgentInput):
        batch_size, seq_length = input.states.shape[0], input.states.shape[1]

        if input.attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            input.attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=input.states.device
            )

        # embed each modality with a different head
        time_embeddings = self.embed_timestep(input.timesteps)
        mission_embeddings = (
            input.mission_embeddings.reshape(batch_size, seq_length, self.hidden_size)
            + time_embeddings
        )
        state_embeddings = (
            self._embed_state(
                input.states.reshape((-1,) + self.config.state_shape)
                .type(torch.float32)
                .contiguous()
            ).reshape(batch_size, seq_length, self.hidden_size)
            + time_embeddings
        )
        action_embeddings = self.embed_action(input.actions) + time_embeddings
        returns_embeddings = self.embed_return(input.returns_to_go) + time_embeddings
        feedback_embeddings = (
            input.feedback_embeddings.reshape(batch_size, seq_length, self.hidden_size)
            + time_embeddings
        )

        # this makes the sequence look like (R_1, s_1, a_1, f_1, R_2, s_2, a_2, f_2 ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (
                mission_embeddings,
                returns_embeddings,
                feedback_embeddings,
                state_embeddings,
                action_embeddings,
            ),
            dim=1,
        ) # shape (batch_size, 5, seq_length, 128)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3) # shape (batch_size, seq_length, 5, 128)
        stacked_inputs = stacked_inputs.reshape(batch_size, 5 * seq_length, self.hidden_size) # shape (batch_size, 5 * seq_length, 128)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack(
                (
                    input.attention_mask,
                    input.attention_mask,
                    input.attention_mask,
                    input.attention_mask,
                    input.attention_mask,
                ),
                dim=1,
            )
            # MAX LOOKING AT THIS
            .permute(0, 2, 1).reshape(batch_size, 5 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(
                stacked_attention_mask.shape,
                device=stacked_inputs.device,
                dtype=torch.long,
            ),
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=self.config.use_return_dict,
        )
        x = encoder_outputs[0]

        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(
            0, 2, 1, 3
        )  # shape (batch_size, 5, seq_length, hidden_size)

        _m, _r, _f, _s, _a = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

        if self.use_rtg:
            _s = torch.cat([_s, _r], axis=2) # shape (batch_size, seq_length, 2 * hidden_size)
            _a = torch.cat([_a, _r], axis=2)

        if self.use_feedback:
            _s = torch.cat([_s, _f], axis=2) # shape (batch_size, seq_length, 3 * hidden_size)
            _a = torch.cat([_a, _f], axis=2)

        if self.use_missions:
            _s = torch.cat([_s, _m], axis=2) # shape (batch_size, seq_length, 4 * hidden_size)
            _a = torch.cat([_a, _m], axis=2)

        # get predictions
        return_preds = self.predict_return(_a)
        state_preds = self.predict_state(_a)
        action_preds = self.predict_action(_s)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _compute_loss(self, input: AgentInput, output: DecisionTransformerOutput, **kwargs):
        act_dim = output.action_preds.shape[2]
        # bacth_size = output.action_preds.shape[0]
        # seq_length = output.action_preds.shape[1]

        action_preds = output.action_preds.reshape(-1, act_dim)

        action_targets = input.actions.reshape(-1, act_dim)
        action_targets = torch.argmax(action_targets, dim=-1).reshape(-1)

        criterion = CrossEntropyLoss()
        loss = criterion(action_preds, action_targets)
        return loss

        # criterion = CrossEntropyLoss(reduce=False)
        # losses = criterion(action_preds, action_targets)
        # losses = losses.reshape(bacth_size, seq_length)
        # loss = self._custom_masked_mean_loss(losses, input.attention_mask)
        # return loss

        # loss2 = self._masked_mean(losses, input.attention_mask.bool() , dim=-1).mean()
        # return loss2

    def _custom_masked_mean_loss(self, losses, mask):
        mean_episode_losses = torch.zeros(losses.shape[0], device=losses.device)
        for i, episode_losses in enumerate(losses):
            episode_losses = episode_losses[mask[i] > 0]
            mean_episode_loss = torch.mean(episode_losses)
            mean_episode_losses[i] = mean_episode_loss
        return torch.mean(mean_episode_losses)

    def _masked_mean(
        self, vector: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
        """To calculate mean along certain dimensions on masked values.

        Implementation from AllenNLP: https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/allennlp/nn/util.py#L351-L377
        Args:
            vector (torch.Tensor): The vector to calculate mean.
            mask (torch.Tensor): The mask of the vector. It must be broadcastable with vector.
            dim (int): The dimension to calculate mean
            keepdim (bool): Whether to keep dimension
        Returns:
            (torch.Tensor): Masked mean tensor
        """
        replaced_vector = vector.masked_fill(~mask, 0.0)  # noqa: WPS358

        value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
        value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
        return value_sum / value_count.float().clamp(min=self._tiny_value_of_dtype(torch.float))

    def _tiny_value_of_dtype(self, dtype: torch.dtype):
        """
        Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
        issues such as division by zero.

        Implementation from AllenNLP: https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py#L2094C22-L2094C22

        This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
        Only supports floating point dtypes.
        """
        if not dtype.is_floating_point:
            raise TypeError("Only supports floating point dtypes.")
        if dtype == torch.float or dtype == torch.double:
            return 1e-13
        elif dtype == torch.half:
            return 1e-4
        else:
            raise TypeError("Does not support dtype " + str(dtype))

    # function that gets an action from the model using autoregressive prediction
    def get_action(
        self,
        input: AgentInput,
        context=30,
        one_hot=True,
    ):
        device = input.states.device

        input.mission_embeddings = input.mission_embeddings.reshape(1, -1, self.hidden_size)
        input.states = input.states.reshape(1, -1, self.config.state_dim)
        input.actions = input.actions.reshape(1, -1, self.config.act_dim)
        input.returns_to_go = input.returns_to_go.reshape(1, -1, 1)
        input.feedback_embeddings = input.feedback_embeddings.reshape(1, -1, self.hidden_size)
        input.timesteps = input.timesteps.reshape(1, -1)

        input.states = input.states[:, -context:]
        input.actions = input.actions[:, -context:]
        input.returns_to_go = input.returns_to_go[:, -context:]
        input.timesteps = input.timesteps[:, -context:]

        # pad all tokens to sequence length
        # padding = context - input.states.shape[1]
        # input.attention_mask = (
        #     torch.cat(
        #         [
        #             torch.zeros(padding, device=device),
        #             torch.ones(input.states.shape[1], device=device),
        #         ]
        #     )
        #     .to(dtype=torch.long)
        #     .reshape(1, -1)
        # )

        output = self._forward(input)
        action = output.action_preds[0, -1]
        return to_one_hot(action) if one_hot else action
