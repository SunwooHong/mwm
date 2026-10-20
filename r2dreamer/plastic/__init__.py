from plastic.logit_plastic_trace import LogitPlasticResidual, PlasticTrace

__all__ = ["LogitPlasticResidual", "PlasticTrace"]


def __getattr__(name: str):
    if name in ("CEMConfig", "plan_cem"):
        from plastic.cem_planner import CEMConfig, plan_cem

        return CEMConfig if name == "CEMConfig" else plan_cem
    raise AttributeError(name)
