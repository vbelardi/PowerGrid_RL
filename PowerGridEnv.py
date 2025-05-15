import gymnasium as gym
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from gymnasium import spaces
import functools

CASE_LIST = ["case14", "case30", "case39", "case57", "case118", "case300"]

@functools.lru_cache(maxsize=128)
def cached_powerflow(net_state):
    """Run powerflow with caching to avoid redundant calculations"""
    # Convert the net to a hashable state representation
    # This is difficult - we'd need to pickle/hash the pandapower network
    # For now, we'll use a simpler approach
    pass

class PowerGridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        case_list=CASE_LIST,
        k=3,
        vmin=0.95,
        vmax=1.05,
        max_loading=1.0,
        negative_reward=-10.0,
        collapse_reward=50.0,
    ):
        super().__init__()
        self.case_list = case_list
        self.k = k
        self.vmin = vmin
        self.vmax = vmax
        self.max_loading = max_loading
        self.negative_reward = negative_reward
        self.collapse_reward = collapse_reward

        self.net = None
        self.all_line_ids = []
        self.removed_line_ids = set()
        self.removed_count = 0

        self.observation_space = None
        self.action_space = None

        self.case_name = np.random.choice(self.case_list)
        self.net = getattr(pn, self.case_name)()
        pp.runpp(self.net)

        # Prepare line indices and reset counters
        self.all_line_ids = list(self.net.line.index)
        self.removed_line_ids.clear()
        self.removed_count = 0

        # Build observation and action spaces
        n_buses = len(self.net.bus)
        n_lines = len(self.all_line_ids)
        n_gens = len(self.net.gen) + len(self.net.ext_grid)

        self.observation_space = spaces.Dict(
            {
                "bus_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_buses, 4), dtype=np.float32
                ),
                "line_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_lines, 5), dtype=np.float32
                ),
                "gen_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_gens, 2), dtype=np.float32
                ),
                "topology": spaces.Box(
                    low=0, high=n_buses - 1, shape=(n_lines, 2), dtype=np.int32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(n_lines,), dtype=np.int8
                ),
            }
        )
        self.action_space = spaces.Discrete(n_lines)

        self.use_precomputed_topology = True
        self._bus_adjacency_matrix = None

    def reset(self):
        # Load a new random network and run base power flow
        self.case_name = np.random.choice(self.case_list)
        self.net = getattr(pn, self.case_name)()
        pp.runpp(self.net)

        # Prepare line indices and reset counters
        self.all_line_ids = list(self.net.line.index)
        self.removed_line_ids.clear()
        self.removed_count = 0

        # Build observation and action spaces
        n_buses = len(self.net.bus)
        n_lines = len(self.all_line_ids)
        n_gens = len(self.net.gen) + len(self.net.ext_grid)

        self.observation_space = spaces.Dict(
            {
                "bus_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_buses, 4), dtype=np.float32
                ),
                "line_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_lines, 5), dtype=np.float32
                ),
                "gen_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_gens, 2), dtype=np.float32
                ),
                "topology": spaces.Box(
                    low=0, high=n_buses - 1, shape=(n_lines, 2), dtype=np.int32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(n_lines,), dtype=np.int8
                ),
            }
        )
        self.action_space = spaces.Discrete(n_lines)

        return self._make_observation(), {}

    def _get_bus_features(self):
        bf = np.zeros((len(self.net.bus), 4), dtype=np.float32)
        bf[:, 0] = self.net.res_bus.vm_pu.values
        bf[:, 1] = self.net.res_bus.va_degree.values
        bf[:, 2] = self.net.res_bus.p_mw.values
        bf[:, 3] = self.net.res_bus.q_mvar.values
        return bf

    def _get_line_features(self):
        lf = np.zeros((len(self.all_line_ids), 5), dtype=np.float32)
        for i, lid in enumerate(self.all_line_ids):
            line = self.net.line.loc[lid]
            lf[i, 0] = line.length_km
            lf[i, 1] = line.r_ohm_per_km
            lf[i, 2] = line.x_ohm_per_km
            lf[i, 3] = self.max_loading
            lf[i, 4] = self.net.res_line.loc[lid, "loading_percent"]
        return lf

    def _get_gen_features(self):
        total = len(self.net.gen) + len(self.net.ext_grid)
        gf = np.zeros((total, 2), dtype=np.float32)
        idx = 0
        # gens
        for gid in self.net.gen.index:
            gen = self.net.gen.loc[gid]
            gf[idx, 0] = gen.p_mw
            gf[idx, 1] = gen.vm_pu
            idx += 1
        for egid in self.net.ext_grid.index:
            eg = self.net.ext_grid.loc[egid]
            gf[idx, 0] = self.net.res_ext_grid.p_mw.loc[egid]
            gf[idx, 1] = eg.vm_pu
            idx += 1
        return gf

    def _get_topology(self):
        topo = np.zeros((len(self.all_line_ids), 2), dtype=np.int32)
        for i, lid in enumerate(self.all_line_ids):
            l = self.net.line.loc[lid]
            topo[i, 0] = l.from_bus
            topo[i, 1] = l.to_bus
        return topo

    def _get_action_mask(self):
        mask = np.ones(len(self.all_line_ids), dtype=np.int8)
        for i, lid in enumerate(self.all_line_ids):
            if lid in self.removed_line_ids:
                mask[i] = 0
        return mask

    def _make_observation(self):
        obs = {
            "bus_features": self._get_bus_features(),
            "line_features": self._get_line_features(),
            "gen_features": self._get_gen_features(),
            "topology": self._get_topology(),
            "action_mask": self._get_action_mask(),
        }

        # Add internal state to the observation for state loading
        obs["internal_state"] = {
            "removed_lines": list(self.removed_line_ids),
            "removed_count": self.removed_count,
            "case_name": self.case_name,
        }

        return obs

    def step(self, action):
        # Prevent illegal picks
        mask = self._get_action_mask()
        if action < 0 or action >= len(mask) or mask[action] == 0:
            # invalid action
            return self._make_observation(), self.negative_reward, False, False, {}

        # Remove selected line
        lid = self.all_line_ids[action]
        self.net.line.at[lid, "in_service"] = False
        self.removed_line_ids.add(lid)
        self.removed_count += 1

        collapsed = False
        # Run power flow and detect collapse
        try:
            pp.runpp(self.net)
        except pp.LoadflowNotConverged:
            collapsed = True

        # Compute violation ratios with more detail
        bus_v = self.net.res_bus.vm_pu.values

        # Penalize more for severe voltage violations
        v_deviations = np.zeros_like(bus_v)
        for i, v in enumerate(bus_v):
            if not np.isnan(v):
                if v < self.vmin:
                    v_deviations[i] = (self.vmin - v) / self.vmin  # normalized deviation
                elif v > self.vmax:
                    v_deviations[i] = (v - self.vmax) / self.vmax  # normalized deviation

        # Square the deviations to penalize larger violations more
        bus_violation = np.sum(v_deviations ** 2) / len(bus_v)

        # Similar approach for line loading
        loadings = self.net.res_line.loading_percent.values
        l_deviations = np.maximum(0, loadings - self.max_loading)  # only count overloads
        line_violation = np.sum(l_deviations ** 2) / len(loadings)

        # Calculate electrical impact - how spread out are the violations?
        topo = self._get_topology()
        spread_factor = 1.0

        if not collapsed and not np.isnan(bus_v).any():
            # Find buses with violations
            violated_buses = np.where(v_deviations > 0)[0]
            if len(violated_buses) > 1:
                # Simple electrical distance approximation
                # Higher factor if violations are spread out across grid
                from scipy.spatial.distance import pdist
                bus_coords = np.array([[self.net.bus.x[i], self.net.bus.y[i]] 
                                       if 'x' in self.net.bus.columns else [0, i] 
                                       for i in violated_buses])
                if len(bus_coords) >= 2:  # Need at least 2 points for pdist
                    distances = pdist(bus_coords)
                    spread_factor = 1.0 + np.mean(distances) / 100.0  # Normalize appropriately

        # Assign reward
        if collapsed:
            reward = self.collapse_reward
        elif np.isnan(self.net.res_bus.vm_pu.values).any():
            # Island formation
            reward = self.collapse_reward / 2
            collapsed = True
        else:
            # Weighted sum of violations with spread factor
            total_violation = bus_violation + 1.5 * line_violation  # Line violations usually more critical
            reward = total_violation * spread_factor

        # Apply diminishing returns for each additional line removed
        reward = reward * (0.95 ** (self.removed_count - 1))  # Less penalty for first line

        # Episode done after collapse or k removals
        done = collapsed or (self.removed_count >= self.k)

        info = {
            "collapsed": collapsed,
            "removed_count": self.removed_count,
            "bus_violation_ratio": bus_violation,
            "line_violation_ratio": line_violation,
        }

        return self._make_observation(), reward, done, False, info

    def load_state(self, obs):
        """
        Load environment state from an observation.
        Used for tree-based search to explore multiple paths.
        
        Args:
            obs (dict): Observation containing environment state
        """
        # Copy the observation's internal state
        if "internal_state" in obs:
            # Get the removed lines from the internal state
            self.removed_line_ids = set(obs["internal_state"]["removed_lines"])
            self.removed_count = obs["internal_state"]["removed_count"]
            self.case_name = obs["internal_state"]["case_name"]
            
            # Recreate the network state
            self.net = getattr(pn, self.case_name)()
            for line_id in self.removed_line_ids:
                if line_id in self.net.line.index:
                    self.net.line.at[line_id, 'in_service'] = False
            
            # Run power flow to get the current state
            try:
                pp.runpp(self.net)
            except pp.LoadflowNotConverged:
                # If power flow doesn't converge, that's fine, the next step will handle it
                pass
            
            # No need to call _update_action_mask since _get_action_mask generates it on demand
        else:
            # If no internal state is provided, just reset
            self.reset()

    def render(self):
        vm = self.net.res_bus.vm_pu.values
        print(
            f"Removed {self.removed_count}/{self.k} | Voltages [{vm.min():.3f}, {vm.max():.3f}]"
        )

    def close(self):
        pass


if __name__ == "__main__":
    env = PowerGridEnv(k=3)
    obs, _ = env.reset()
    print("Bus feats:", obs["bus_features"].shape)
    print("Line feats:", obs["line_features"].shape)
    print("Gen feats:", obs["gen_features"].shape)
    print("Topo:", obs["topology"].shape)
    print("Mask:", obs["action_mask"])
    done = False
    while not done:
        valid = np.where(obs["action_mask"] == 1)[0]
        action = np.random.choice(valid)
        obs, reward, done, _, info = env.step(action)
        print(f"Action={action}, Reward={reward:.2f}, Done={done}, Info={info}")
    env.render()
