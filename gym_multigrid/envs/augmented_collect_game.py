from gym_multigrid.multigrid import *


class AugCollectGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        agents_index=[],
        balls_index=[],
        balls_reward=[],
        zero_sum=False,
        view_size=7,
        opponent_policy=[],
        **kwargs
    ):
        self.num_balls = num_balls
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum

        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        self.num_agent = len(agents_index)
        self.num_opponent = len(opponent_policy)

        assert self.num_agent == self.num_opponent + 1

        self.opponent_policy = []
        for opp in opponent_policy:
            self.opponent_policy.append(opp)

        self.prev_obs = [None] * len(opponent_policy)

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height - 1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width - 1, 0)

        for number, index, reward in zip(
            self.num_balls, self.balls_index, self.balls_reward
        ):
            for i in range(number):
                self.place_obj(Ball(self.world, index, reward))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j, a in enumerate(self.agents):
            if a.index == i or a.index == 0:
                rewards[j] += reward
            if self.zero_sum:
                if a.index != i or a.index == 0:
                    rewards[j] -= reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self._reward(i, rewards, fwd_cell.reward)

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def reset(self):
        obs = super().reset()
        for i in range(1, self.num_agent):
            self.prev_obs[i - 1] = obs[i]
        return obs[0]

    def step(self, actions):
        _actions = []

        if self.num_opponent == 0:
            actions = actions.tolist()
            if isinstance(actions, int):
                actions = [actions]

            obs, rewards, done, info = super().step(actions)
        else:
            _actions.append(int(actions))

            for i in range(self.num_opponent):
                opponent_policy = self.opponent_policy[i]
                ac, _ = opponent_policy.predict(self.prev_obs[i])
                _actions.append(ac)

            obs, rewards, done, info = super().step(_actions)

        rewards = int(rewards[0])
        for i in range(self.num_opponent):
            self.prev_obs[i] = obs[i + 1]

        return obs[0], rewards, done, info
