import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy_step(nn.Module):
    def __init__(self, option, m):
        super(Policy_step, self).__init__()

        self.lstm_cell = nn.LSTMCell(input_size=2 * m * option.state_embed_size, hidden_size=2 * m * option.state_embed_size)
        '''
        Params
        input_size – The number of expected features in the input x
        hidden_size – The number of features in the hidden state h

        Inputs: input, (h_0, c_0)
        input of shape (batch, input_size): tensor containing input features
        h_0 of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        c_0 of shape (batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

        Outputs: (h_1, c_1)
        h_1 of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
        c_1 of shape (batch, hidden_size): tensor containing the next cell state for each element in the batch

        '''
        self.l1 = nn.Linear(m * option.state_embed_size, 2 * m * option.state_embed_size)
        self.l2 = nn.Linear(2 * m * option.state_embed_size, m * option.state_embed_size)
        self.l3 = nn.Linear(2 * m * option.state_embed_size, m * option.state_embed_size)

    def forward(self, prev_action, prev_state):

        prev_action = torch.relu(self.l1(prev_action))
        output, ch = self.lstm_cell(prev_action, prev_state)
        output = torch.relu(self.l2(output))
        ch = torch.relu(self.l3(ch))

        ch = torch.cat([output.unsqueeze(0).unsqueeze(0), ch.unsqueeze(0).unsqueeze(0)], dim=1)

        return output, ch

class Policy_mlp(nn.Module):
    def __init__(self, option, m):
        super(Policy_mlp, self).__init__()

        self.hidden_size = option.mlp_hidden_size
        self.embedding_size = option.state_embed_size
        self.mlp_l1 = nn.Linear(2 * m * self.embedding_size, m * self.hidden_size, bias=True)
        self.mlp_l2 = nn.Linear(m * self.hidden_size, m * self.embedding_size, bias=True)

    def forward(self, state_query):
        # state_query = state_query.float()
        hidden = torch.relu(self.mlp_l1(state_query))
        output = torch.relu(self.mlp_l2(hidden))
        return output



class Agent(nn.Module):
    def __init__(self, option, data_loader, graph=None):
        super(Agent, self).__init__()
        self.option = option
        self.data_loader = data_loader
        self.graph = graph
        self.relation_embedding = nn.Embedding(self.option.num_relation, 2 * self.option.state_embed_size)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)
        if self.option.use_entity_embed:
            self.entity_embedding = nn.Embedding(self.option.num_entity, 2 * self.option.state_embed_size)
            self.m = 4
            torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        else:
            self.m = 2
        self.policy_step = Policy_step(self.option, self.m)
        self.policy_mlp = Policy_mlp(self.option, self.m)

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def action_encoder(self, next_relations, next_entities):
        # relation_embedding = self.relation_embedding[next_relations.cpu().numpy()]
        # entity_embedding = self.entity_embedding[next_entities.cpu().numpy()]
        relation_embedding = self.relation_embedding(next_relations)
        entity_embedding = self.entity_embedding(next_entities)

        if self.option.use_entity_embed:
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        else:
            action_embedding = relation_embedding

        return action_embedding

    def step(self, prev_state, prev_relation, current_entities, queries, range_arr, video_text_shared_informs):
        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        # [B, act_emb_sz]
        prev_state = torch.unbind(prev_state, dim=1)
        prev_state = [prev_state[0].squeeze(0), prev_state[1].squeeze(0)]
        video_text_shared_informs = torch.unbind(video_text_shared_informs, dim=1)
        video_text_shared_informs = [video_text_shared_informs[0].squeeze(0), video_text_shared_informs[1].squeeze(0)]
        prev_state = (torch.cat([prev_state[0], video_text_shared_informs[0]], dim=-1),
                        torch.cat([prev_state[1], video_text_shared_informs[1]], dim=-1))                        
        output, new_state = self.policy_step(prev_action_embedding, prev_state)
        # [B, state_emb_sz]
        prev_entity = self.entity_embedding(current_entities)
        if self.option.use_entity_embed:
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            state = output

        actions_id = self.graph.get_out(current_entities)
        # [B, maxout, 2]
        next_relations_id = actions_id[:, :, 0]
        # [B, maxout]
        next_entities_id = actions_id[:, :, 1]
        # [B, maxout]
        candidate_action_embeddings = self.action_encoder(next_relations_id, next_entities_id)
        query_embedding = self.relation_embedding(queries)
        state_query_concat = torch.cat([state, query_embedding], dim=-1)
        output = self.policy_mlp(state_query_concat)
        output_expanded = torch.unsqueeze(output, dim=1)
        # [B, 1, 2*hidden_state]
        prelim_scores = torch.sum(candidate_action_embeddings * output_expanded, dim=2)
        # [B, max_out]
        
        # Masking PAD actions

        comparison_tensor = torch.ones_like(next_relations_id).int() * self.data_loader.relation2num["<pad>"]  # matrix to compare
        mask = next_relations_id == comparison_tensor  # The mask
        dummy_scores = torch.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = torch.where(mask, dummy_scores, prelim_scores)  # [original batch_size * num_rollout, max_num_actions]

        # 4 sample action
        action = torch.distributions.categorical.Categorical(logits=scores) # [original batch_size * num_rollout, 1]
        label_action = action.sample() # [original batch_size * num_rollout,]

        # loss
        # 5a.
        loss = torch.nn.CrossEntropyLoss(reduction='none')(scores, label_action)

        # 6. Map back to true id
        chosen_relation = next_relations_id[list(torch.stack([range_arr, label_action]))]
        chosen_entities = next_entities_id[list(torch.stack([range_arr, label_action]))]

        
        
        return loss, new_state, F.log_softmax(scores, dim=-1), label_action, chosen_entities, chosen_relation

    def test_step(self, prev_state, prev_relation, current_entities, queries, range_arr, video_text_shared_informs):
                  # log_currenct_prob default zeros
        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        # [B, acti_emb_sz]
        prev_state = torch.unbind(prev_state, dim=1)
        prev_state = [prev_state[0].squeeze(0), prev_state[1].squeeze(0)]
        video_text_shared_informs = torch.unbind(video_text_shared_informs, dim=1)
        video_text_shared_informs = [video_text_shared_informs[0].squeeze(0), video_text_shared_informs[1].squeeze(0)]
        prev_state = (torch.cat([prev_state[0], video_text_shared_informs[0]], dim=-1),
                        torch.cat([prev_state[1], video_text_shared_informs[1]], dim=-1))                        
        output, new_state = self.policy_step(prev_action_embedding, prev_state)
        # [B, state_emb_sz]
        prev_entity = self.entity_embedding(current_entities)
        if self.option.use_entity_embed:
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            state = output

        actions_id = self.graph.get_out(current_entities)
        # [B, maxout, 2]
        next_relations_id = actions_id[:, :, 0]
        # [B, maxout]
        next_entities_id = actions_id[:, :, 1]
        # [B, maxout]
        candidate_action_embeddings = self.action_encoder(next_relations_id, next_entities_id)
        query_embedding = self.relation_embedding(queries)
        state_query_concat = torch.cat([state, query_embedding], dim=-1)
        output = self.policy_mlp(state_query_concat)
        output_expanded = torch.unsqueeze(output, dim=1)
        # [B, 1, 2*hidden_state]
        prelim_scores = torch.sum(candidate_action_embeddings * output_expanded, dim=2)
        # [B, max_out]
        
        # Masking PAD actions

        comparison_tensor = torch.ones_like(next_relations_id).int() *  self.data_loader.relation2num["<pad>"]  # matrix to compare
        mask = next_relations_id == comparison_tensor  # The mask
        dummy_scores = torch.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = torch.where(mask, dummy_scores, prelim_scores)  # [original batch_size * num_rollout, max_num_actions]
        # scores_prob = F.log_softmax(scores, dim=-1)
        
        # 4 sample action
        action = torch.distributions.categorical.Categorical(logits=scores) # [original batch_size * num_rollout, 1]
        label_action = action.sample() # [original batch_size * num_rollout,]

        # 5. Map back to true id
        chosen_relation = next_relations_id[list(torch.stack([range_arr, label_action]))]
        chosen_entities = next_entities_id[list(torch.stack([range_arr, label_action]))]
        
        return new_state, F.log_softmax(scores, dim=-1), label_action, chosen_entities, chosen_relation, next_entities_id, next_relations_id

    def set_graph(self, graph):
        self.graph = graph

    def get_dummy_start_relation(self, batch_size):
        dummy_start_item = self.data_loader.relation2num["<start>"]
        dummy_start = torch.ones(batch_size, dtype=torch.int64) * dummy_start_item
        return dummy_start

    def get_reward(self, current_entities, answers, all_correct, positive_reward, negative_reward):
        reward = (current_entities == answers).cpu()

        reward = reward.numpy()
        condlist = [reward == True, reward == False]
        choicelist = [positive_reward, negative_reward]
        reward = np.select(condlist, choicelist)
        return reward

    def print_parameter(self):
        for param in self.named_parameters():
            print(param[0], param[1])