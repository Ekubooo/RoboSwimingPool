"""Frozen-policy knee release experiment. No resets, state edits, or learning interventions."""
import csv
import math

import torch

from tools.getup.common import write_json


class ReleaseSchedule:
    """One event per slot/knee after consecutive outward commands at the lower stop."""

    def __init__(self, shape, device, dt):
        self.required = math.ceil(0.5 / dt)
        self.duration = math.ceil(1.0 / dt)
        self.count = torch.zeros(shape, dtype=torch.long, device=device)
        self.start = torch.full(shape, -1, dtype=torch.long, device=device)

    def update(self, q, requested, lower, active, launched, step):
        eligible = (active & launched)[:, None]
        pressure = eligible & (q < lower + 0.03) & (requested < lower - 1e-4)
        self.count = torch.where(pressure, self.count + 1, torch.zeros_like(self.count))
        trigger = (self.count >= self.required) & (self.start < 0)
        self.start[trigger] = step
        pulse = eligible & (self.start >= 0) & (step - self.start < self.duration)
        return trigger, pulse


class ControlProbe:
    def __init__(self, env, variant, out):
        import mujoco
        import warp as wp

        self.env, self.variant, self.out = env, variant, out
        self.char = env._get_char_id()
        self.dt = env._engine.get_timestep()
        self.step = 0
        self.names = ["left_knee_joint", "right_knee_joint"]
        kin = env._kin_char_model
        by_name = {kin.get_joint(i).name: kin.get_joint(i).dof_idx
                   for i in range(1, kin.get_num_joints())}
        self.ids = [by_name[name] for name in self.names]
        q = env._engine.get_dof_pos(self.char)
        low, high = env._engine.get_obj_dof_limits(0, self.char)
        self.low = torch.as_tensor(low, device=q.device)
        self.high = torch.as_tensor(high, device=q.device)
        self.schedule = ReleaseSchedule((q.shape[0], 2), q.device, self.dt)
        self.events, self.records = {}, []
        self.initial_q = q.cpu().tolist()
        # Read generalized joint torque, not actuator-space force or inferred PD torque.
        # Resolve solver indices by names and verify them against engine q each step.
        solver = env._engine._solver
        model = solver.mj_model
        mj_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    for i in range(model.njnt)]
        joint_ids = []
        for name in self.names:
            matches = [i for i, label in enumerate(mj_names)
                       if label == name or (label and label.endswith("_" + name))]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous MuJoCo joint mapping: {name}: {mj_names}")
            joint_ids.append(matches[0])
        self.q_indices = model.jnt_qposadr[joint_ids].tolist()
        self.v_indices = model.jnt_dofadr[joint_ids].tolist()
        self.mj_q = wp.to_torch(solver.mjw_data.qpos)
        self.torque = wp.to_torch(solver.mjw_data.qfrc_actuator)
        if self.torque.ndim != 2 or self.torque.shape[0] != q.shape[0]:
            raise ValueError("Expected one MuJoCo world per environment for torque mapping.")
        self.mapping = {name: {"qpos": qi, "dof": vi}
                        for name, qi, vi in zip(self.names, self.q_indices, self.v_indices)}

    def before_step(self, action, active, launches):
        q = self.env._engine.get_dof_pos(self.char)[:, self.ids]
        self.requested = self.env._clip_action(action)[:, self.ids].clone()
        trigger, self.pulse = self.schedule.update(
            q, self.requested, self.low[self.ids], active, launches > 0, self.step)
        for slot, knee in trigger.nonzero().tolist():
            self.events[slot, knee] = {
                "env_id": slot, "joint": self.names[knee],
                "start_time_seconds": self.step * self.dt,
                "initial_q_rad": float(q[slot, knee]), "max_q_rad": float(q[slot, knee]),
                "first_response_seconds": None, "observed_steps": 0,
                "window_complete": False, "terminal_during_window": False,
            }
        sent = action.clone()
        if self.variant == "clip":
            sent = sent.clamp(self.low, self.high)
        elif self.variant == "release":
            for knee, index in enumerate(self.ids):
                sent[:, index] = torch.where(self.pulse[:, knee], 0.6, sent[:, index])
        self.sent = self.env._clip_action(sent)[:, self.ids].clone()
        return sent

    def after_step(self, active, done):
        engine = self.env._engine
        q = engine.get_dof_pos(self.char)[:, self.ids]
        vel = engine.get_dof_vel(self.char)[:, self.ids]
        torque = self.torque[:, self.v_indices]
        if not torch.allclose(q, self.mj_q[:, self.q_indices], atol=1e-4, rtol=1e-4):
            raise RuntimeError("MuJoCo/engine knee mapping or state synchronization mismatch.")
        if not bool(torch.isfinite(q).all() & torch.isfinite(vel).all() & torch.isfinite(torque).all()):
            raise RuntimeError("Non-finite control probe state/torque.")
        packed = torch.stack((q, vel, self.requested, self.sent, torque,
                              self.pulse.float(), done[:, None].expand_as(q).float()), dim=-1).cpu()
        for slot in active.nonzero().flatten().tolist():
            for knee, name in enumerate(self.names):
                values = packed[slot, knee].tolist()
                self.records.append([slot, (self.step + 1) * self.dt, name, *values])
                event = self.events.get((slot, knee))
                if event is not None and bool(self.pulse[slot, knee]):
                    event["observed_steps"] += 1
                    event["max_q_rad"] = max(event["max_q_rad"], values[0])
                    if values[0] > float(self.low[self.ids[knee]]) + 0.15 and event["first_response_seconds"] is None:
                        event["first_response_seconds"] = event["observed_steps"] * self.dt
                    event["window_complete"] = event["observed_steps"] == self.schedule.duration
                    event["terminal_during_window"] |= values[-1] != 0
        self.step += 1

    def save(self):
        events = list(self.events.values())
        result = {
            "variant": self.variant, "event_count": len(events),
            "responded_events": sum(e["first_response_seconds"] is not None for e in events),
            "complete_windows": sum(e["window_complete"] for e in events),
            "censored_windows": sum(not e["window_complete"] for e in events),
            "events": events, "initial_dof_pos": self.initial_q, "solver_joint_mapping": self.mapping,
            "trigger": "After launch, 0.5s consecutive q<lower+0.03 and policy target<lower-0.0001; once per knee.",
            "intervention": "release: target=0.6 rad for 1s; clip: all targets clipped throughout; baseline: unchanged.",
            "response": "q>lower+0.15 during the 1s window. Early termination censors unobserved response.",
            "torque": "MuJoCo qfrc_actuator at final physics substep, Nm; not maximum/average across substeps.",
            "limits": self.low[self.ids].cpu().tolist(),
            "notes": ["Release is an artificial diagnostic, not a deployable recovery controller.",
                      "A response disproves irreversible locking for that event; nonresponse alone does not prove a solver bug.",
                      "Baseline/clip windows are observational; only release overrides the knees.",
                      "Variants share initialization/seed; later trajectories and launches may diverge after interventions/resets."],
        }
        write_json(self.out / "control_probe.json", result)
        with (self.out / "control_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "time_seconds", "joint", "q_rad", "velocity_rad_s",
                             "policy_target_rad", "sent_target_rad", "actuator_torque_nm",
                             "event_window", "done"])
            writer.writerows(self.records)
        return result
