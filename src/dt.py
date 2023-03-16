from dataclasses import dataclass

import torch
from transformers import DecisionTransformerModel
from transformers.utils import ModelOutput

from src.utils import to_one_hot


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


class FeedbackDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

        # embed image states using very simple conv net
        self.embed_state_convolutional = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 4, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(512, config.hidden_size),
            torch.nn.Tanh(),
        )

    def _forward(
        self,
        states=None,
        actions=None,
        rewards=None,
        returns_to_go=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state_convolutional(
            states.reshape(-1, 3, 7, 7).type(torch.float32).contiguous()
        ).reshape(batch_size, seq_length, self.hidden_size)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        # TODO: add feedback embeddings

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )
        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(
                stacked_attention_mask.shape, device=device, dtype=torch.long
            ),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(
            x[:, 2]
        )  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        if not return_dict:
            return (state_preds, action_preds, return_preds)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward(self, **kwargs):
        output = self._forward(**kwargs)

        # compute and return the loss
        action_preds = output[1]
        action_targets = kwargs["actions"]

        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        return {"loss": torch.mean((action_preds - action_targets) ** 2)}

    # function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, one_hot=False):
        # This implementation does not condition on past rewards
        device = states.device

        states = states.reshape(1, -1, self.config.state_dim)
        actions = actions.reshape(1, -1, self.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -self.config.max_length :]
        actions = actions[:, -self.config.max_length :]
        returns_to_go = returns_to_go[:, -self.config.max_length :]
        timesteps = timesteps[:, -self.config.max_length :]

        # pad all tokens to sequence length
        padding = self.config.max_length - states.shape[1]
        attention_mask = torch.cat(
            [torch.zeros(padding, device=device), torch.ones(states.shape[1], device=device)]
        )
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)

        states = torch.cat(
            [torch.zeros((1, padding, self.config.state_dim), device=device), states],
            dim=1,
        ).float()
        actions = torch.cat(
            [torch.zeros((1, padding, self.config.act_dim), device=device), actions],
            dim=1,
        ).float()
        returns_to_go = torch.cat(
            [torch.zeros((1, padding, 1), device=device), returns_to_go], dim=1
        ).float()
        timesteps = torch.cat(
            [torch.zeros((1, padding), dtype=torch.long, device=device), timesteps],
            dim=1,
        )

        _, action_preds, _ = self._forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        action = action_preds[0, -1]
        return to_one_hot(action) if one_hot else action
