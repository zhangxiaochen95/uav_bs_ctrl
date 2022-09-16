REGISTRY = {}

from envs.subs_cov.subs_cov import SingleUbsCoverageEnv
from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv

REGISTRY['SingleUbsCoverageEnv'] = SingleUbsCoverageEnv
REGISTRY['MultiUbsCoverageEnv'] = MultiUbsCoverageEnv