import numpy as np
import torch


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority):
        idx = self.data_pointer + self.capacity - 1
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx

    def min(self):
        start = self.capacity - 1
        end = start + self.size
        if self.size == 0:
            return 0.0
        return np.min(self.tree[start:end])


class PrioritizedReplayBuffer:
    def __init__(self, capacity, priority_exponent, priority_weight,
                 multi_step, discount, frame_stack):
        self.capacity = capacity
        self.priority_exponent = priority_exponent
        self.priority_weight_init = priority_weight
        self.priority_weight = priority_weight
        self.multi_step = multi_step
        self.discount = discount
        self.frame_stack = frame_stack
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.states = np.zeros((capacity, 84, 84), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.transition_ids = np.full(capacity, -1, dtype=np.int64)
        self.ptr = 0
        self.size = 0
        self.n_step_buffer = []
        self.next_transition_id = 0

    def _get_n_step_info(self):
        R = 0.0
        terminal = False
        horizon = min(len(self.n_step_buffer), self.multi_step)

        for i in range(horizon):
            _, _, r, d = self.n_step_buffer[i]
            R += (self.discount ** i) * r
            if d:
                terminal = True
                break

        s0, a0, _, _ = self.n_step_buffer[0]
        return s0, a0, R, terminal

    def append(self, state, action, reward, done):
        self.n_step_buffer.append((state[-1], action, reward, done))

        if done:
            while self.n_step_buffer:
                s0, a0, R, terminal = self._get_n_step_info()
                self._store(s0, a0, R, terminal)
                self.n_step_buffer = self.n_step_buffer[1:]
            return

        if len(self.n_step_buffer) < self.multi_step:
            return

        s0, a0, R, terminal = self._get_n_step_info()
        self._store(s0, a0, R, terminal)
        self.n_step_buffer = self.n_step_buffer[1:]

    def _store(self, state, action, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.transition_ids[self.ptr] = self.next_transition_id
        self.next_transition_id += 1
        priority = self.max_priority ** self.priority_exponent
        self.tree.add(priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _oldest_transition_id(self):
        return self.next_transition_id - self.size

    def _has_transition(self, transition_id):
        if transition_id < self._oldest_transition_id() or transition_id >= self.next_transition_id:
            return False
        slot = transition_id % self.capacity
        return self.transition_ids[slot] == transition_id

    def _slot_for_transition(self, transition_id):
        if not self._has_transition(transition_id):
            raise IndexError(f"Transition {transition_id} is not available in buffer")
        return transition_id % self.capacity

    def _is_valid_index(self, data_idx):
        transition_id = self.transition_ids[data_idx]
        if transition_id < 0:
            return False
        if transition_id - (self.frame_stack - 1) < self._oldest_transition_id():
            return False
        if self.dones[data_idx]:
            return True
        future_transition = transition_id + self.multi_step
        return self._has_transition(future_transition)

    def _build_stacked_frames(self, end_transition_id):
        end_slot = self._slot_for_transition(end_transition_id)
        end_frame = self.states[end_slot]
        stacked = np.zeros((self.frame_stack, 84, 84), dtype=np.float32)
        for j in range(self.frame_stack):
            transition_id = end_transition_id - self.frame_stack + 1 + j
            if not self._has_transition(transition_id):
                stacked[j] = end_frame
                continue
            crosses_boundary = False
            for prev_transition in range(transition_id, end_transition_id):
                prev_slot = self._slot_for_transition(prev_transition)
                if self.dones[prev_slot]:
                    crosses_boundary = True
                    break
            if crosses_boundary:
                stacked[j] = end_frame
            else:
                stacked[j] = self.states[self._slot_for_transition(transition_id)]
        return stacked

    def sample(self, batch_size):
        indices = []
        tree_indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx, priority, data_idx = self.tree.get(s)
            while not self._is_valid_index(data_idx):
                s = np.random.uniform(segment * i, segment * (i + 1))
                tree_idx, priority, data_idx = self.tree.get(s)
            indices.append(data_idx)
            tree_indices.append(tree_idx)
            priorities.append(priority)
        indices = np.array(indices)
        states = np.zeros((batch_size, self.frame_stack, 84, 84), dtype=np.float32)
        next_states = np.zeros((batch_size, self.frame_stack, 84, 84), dtype=np.float32)
        for i, idx in enumerate(indices):
            transition_id = self.transition_ids[idx]
            states[i] = self._build_stacked_frames(transition_id)
            if self.dones[idx]:
                next_states[i] = states[i]
            else:
                next_states[i] = self._build_stacked_frames(transition_id + self.multi_step)
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        priorities = np.array(priorities, dtype=np.float64)
        probs = priorities / self.tree.total()
        min_prob = self.tree.min() / self.tree.total()
        if min_prob == 0:
            min_prob = 1e-8
        max_weight = (min_prob * self.size) ** (-self.priority_weight)
        weights = (probs * self.size) ** (-self.priority_weight) / max_weight
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones.astype(np.float32)),
            torch.FloatTensor(weights.astype(np.float32)),
            tree_indices,
            indices
        )

    def update_priorities(self, tree_indices, priorities):
        for idx, priority in zip(tree_indices, priorities):
            priority = max(priority, 1e-6)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.priority_exponent)

    def anneal_priority_weight(self, step, total_steps):
        self.priority_weight = self.priority_weight_init + \
            (1.0 - self.priority_weight_init) * step / total_steps

    def __len__(self):
        return self.size
