import torch
import torch.nn as nn



class Communicator(nn.Module):
    def __init__(self, action_embed_size, state_embed_size):
        super(Communicator, self).__init__()
        self.lstm_cell = torch.nn.LSTMCell(input_size=action_embed_size, hidden_size=state_embed_size)
    '''
    - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
    - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
    - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
    - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
    '''
    def forward(self, prev_state, prev_vrd_action_embedding, prev_text_action_embedding):
        prev_action = torch.cat([prev_vrd_action_embedding, prev_text_action_embedding], dim=-1)
        output, new_state = self.lstm_cell(prev_action, prev_state)
        return (output, new_state)

