"""Native G1 dodgeball with the released canonical LaFAN prior interface."""
from envs.g1_lafan_env import G1LafanEnvMixin
from envs.task_dodgeball_env import TaskDodgeballEnv


class TaskDodgeballG1LafanEnv(G1LafanEnvMixin, TaskDodgeballEnv):
    pass
