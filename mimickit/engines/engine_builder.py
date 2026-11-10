try:
    import isaacgym.gymapi as gymapi
except ImportError:
    pass

import util.resource_util as resource_util

def build_engine(config, num_envs, device, visualize, record_video=False):
    eng_name = config["engine_name"]
    if eng_name in ("newton", "newton_1_7"):
        from engines.newton_version import check_newton_version
        check_newton_version(eng_name)

    if (eng_name == "isaac_gym"):
        import engines.isaac_gym_engine as isaac_gym_engine
        engine = resource_util.construct(isaac_gym_engine.IsaacGymEngine, config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "isaac_lab"):
        import engines.isaac_lab_engine as isaac_lab_engine
        engine = resource_util.construct(isaac_lab_engine.IsaacLabEngine, config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "newton"):
        import engines.newton_engine as newton_engine
        engine = resource_util.construct(newton_engine.NewtonEngine, config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "newton_1_7"):
        import engines.newton_1_7_engine as newton_1_7_engine
        engine = resource_util.construct(newton_1_7_engine.NewtonEngine, config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "newton_pbf"):
        from engines.newton_version import check_newton_version
        from engines.newton_pbf_engine import NewtonEngine
        check_newton_version("newton_1_7")
        engine = resource_util.construct(NewtonEngine, config, num_envs, device, visualize, record_video=record_video)
    else:
        assert False, print("Unsupported engine: {:s}".format(eng_name))

    return engine
