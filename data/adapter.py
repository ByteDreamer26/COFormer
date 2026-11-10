import tree
from abc import ABC

class DatasetAdapter(ABC):
    ''' 请改编此类, 获取数据集的 meta data, 并编写合适的后处理方法 '''
    def __init__(self, dataset_name, epi_obs, epi_act, epi_prefix=None, disable_visited_obs=False):
        self.dataset_name = dataset_name
        self.epi_observations = epi_obs     # observations of an random episode
        self.epi_actions = epi_act          # actions of an random episode
        self.epi_prefix = epi_prefix        # prefix of an random episode
        self.disable_visited_obs = disable_visited_obs

    def get_observation_dim(self):
        """Get the length of observation (num of obs token) when feed into transformer"""    
        def _compute_single_obs_dim(x):
            return x[0].size

        obs = self.post_process_obs(self.epi_observations)
        dims = tree.map_structure(_compute_single_obs_dim, obs)
        return dims

    def get_prefix_dim(self):
        """Get the length of prefix (num of act token) when feed into transformer"""
        def _compute_single_prefix_dim(x):    
            assert x.ndim == 1
            return x.size

        if self.epi_prefix is None:
            return None

        prefix = self.post_process_obs(self.epi_prefix)
        dims = tree.map_structure(_compute_single_prefix_dim, prefix)
        return dims

    def get_action_dim(self):
        """Get the length of action (num of act token) when feed into transformer"""
        act = self.post_process_act(self.epi_actions)[0]
        return act.shape[0] if len(act.shape) == 1 else 1

    def get_obs_type_spec(self):
        """type can be [float, int]"""
        def _get_obs_type(x):
            if "float" in x.dtype.name:
                return "float"
            elif "int" in x.dtype.name:
                return "int"
            else:
                raise ValueError

        obs = self.post_process_obs(self.epi_observations)
        return tree.map_structure(_get_obs_type, obs)

    def get_prefix_type_spec(self):
        """type can be [float, int]"""
        def _get_prefix_type(x):
            if "float" in x.dtype.name:
                return "float"
            elif "int" in x.dtype.name:
                return "int"
            else:
                raise ValueError

        if self.epi_prefix is None:
            return None

        prefix = self.post_process_obs(self.epi_prefix)
        return tree.map_structure(_get_prefix_type, prefix)

    def get_act_type_spec(self):
        """type can be float or int"""
        act = self.post_process_act(self.epi_actions)

        if "float" in act.dtype.name:
            return "float"
        elif "int" in act.dtype.name:
            return "int"
        else:
            raise ValueError

    def get_meta_data(self):
        obs_dims_for_spec = self.get_observation_dim()
        prefix_dims_for_spec = self.get_prefix_dim()
        obs_type_spec = self.get_obs_type_spec()

        if self.disable_visited_obs and 'visited' in obs_dims_for_spec:
            obs_dims_for_spec.pop('visited')
        if self.disable_visited_obs and 'visited' in obs_type_spec:
            obs_type_spec.pop('visited')

        return {
            'prefix_type_spec': self.get_prefix_type_spec(), 
            'obs_type_spec': obs_type_spec,
            'act_type_spec': self.get_act_type_spec(), 
            'prefix_dims_for_spec': prefix_dims_for_spec, 
            'prefix_dim': None if prefix_dims_for_spec is None else sum(tree.flatten(prefix_dims_for_spec)), 
            'obs_dims_for_spec': obs_dims_for_spec, 
            'obs_dim': sum(tree.flatten(obs_dims_for_spec)), 
            'act_dim': self.get_action_dim()
        }

    def post_process_obs(self, epi_obs):
        return epi_obs

    def post_process_prefix(self, epi_prefix):
        return epi_prefix

    def post_process_act(self, epi_act):
        return epi_act

    def recover_raw_act(self, epi_act):
        return epi_act
