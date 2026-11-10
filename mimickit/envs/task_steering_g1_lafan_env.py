"""Original Steering task with native G1 physics and canonical LaFAN observations."""
from envs.g1_lafan_env import G1LafanEnvMixin
from envs.task_steering_env import TaskSteeringEnv


class TaskSteeringG1LafanEnv(G1LafanEnvMixin, TaskSteeringEnv):
    pass
