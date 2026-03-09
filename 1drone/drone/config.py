from dataclasses import dataclass


@dataclass
class ScenarioCfg:
    grid_side: int = 10
    n_bases: int = 2
    n_drones: int = 2
    n_feat: int = 6

    @property
    def n_grid(self):
        return self.grid_side * self.grid_side

    @property
    def base_start(self):
        return self.n_grid

    @property
    def drone_start(self):
        return self.n_grid + self.n_bases

    @property
    def n_nodes(self):
        return self.n_grid + self.n_bases + self.n_drones

    @property
    def base_slice(self):
        return slice(self.base_start, self.base_start + self.n_bases)

    @property
    def drone_slice(self):
        return slice(self.drone_start, self.drone_start + self.n_drones)
