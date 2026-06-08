from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import rosbag2_py
from control_msgs.msg import DynamicJointState
from rclpy.serialization import deserialize_message
from simlab.msg import ReferenceTargets

ARM_AXES = ("e", "d", "c", "b", "a")
ARM_STATE_AXES = ("e", "d", "c", "b")
REFERENCE_TOPIC_TYPE = "simlab/msg/ReferenceTargets"

FIELDNAMES = [
    "wall_time_sec",
    "sim_time_sec",
    "replay_time_sec",
    "profile",
    "pass_index",
    "q_alpha_axis_e",
    "q_alpha_axis_d",
    "q_alpha_axis_c",
    "q_alpha_axis_b",
    "dq_alpha_axis_e",
    "dq_alpha_axis_d",
    "dq_alpha_axis_c",
    "dq_alpha_axis_b",
    "ddq_alpha_axis_e",
    "ddq_alpha_axis_d",
    "ddq_alpha_axis_c",
    "ddq_alpha_axis_b",
    "ref_alpha_axis_e",
    "ref_alpha_axis_d",
    "ref_alpha_axis_c",
    "ref_alpha_axis_b",
    "dref_alpha_axis_e",
    "dref_alpha_axis_d",
    "dref_alpha_axis_c",
    "dref_alpha_axis_b",
    "ddref_alpha_axis_e",
    "ddref_alpha_axis_d",
    "ddref_alpha_axis_c",
    "ddref_alpha_axis_b",
    "effort_alpha_axis_e",
    "effort_alpha_axis_d",
    "effort_alpha_axis_c",
    "effort_alpha_axis_b",
    "cmd_tau_axis_e",
    "cmd_tau_axis_d",
    "cmd_tau_axis_c",
    "cmd_tau_axis_b",
    "cmd_tau_axis_a",
    "vehicle_x",
    "vehicle_y",
    "vehicle_z",
    "vehicle_roll",
    "vehicle_pitch",
    "vehicle_yaw",
    "vehicle_u",
    "vehicle_v",
    "vehicle_w",
    "vehicle_p",
    "vehicle_q",
    "vehicle_r",
    "vehicle_du",
    "vehicle_dv",
    "vehicle_dw",
    "vehicle_dp",
    "vehicle_dq",
    "vehicle_dr",
    "target_vehicle_x",
    "target_vehicle_y",
    "target_vehicle_z",
    "target_vehicle_roll",
    "target_vehicle_pitch",
    "target_vehicle_yaw",
    "target_vehicle_u",
    "target_vehicle_v",
    "target_vehicle_w",
    "target_vehicle_p",
    "target_vehicle_q",
    "target_vehicle_r",
    "target_vehicle_du",
    "target_vehicle_dv",
    "target_vehicle_dw",
    "target_vehicle_dp",
    "target_vehicle_dq",
    "target_vehicle_dr",
    "wrench_vehicle_fx",
    "wrench_vehicle_fy",
    "wrench_vehicle_fz",
    "wrench_vehicle_tx",
    "wrench_vehicle_ty",
    "wrench_vehicle_tz",
    "cmd_vehicle_fx",
    "cmd_vehicle_fy",
    "cmd_vehicle_fz",
    "cmd_vehicle_tx",
    "cmd_vehicle_ty",
    "cmd_vehicle_tz",
    "payload_mass",
    "gravity",
]


def _clean(value: float, eps: float = 1e-12) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return 0.0 if abs(value) < eps else value


def _interface_value(msg: DynamicJointState, name: str, interface: str, default: float = 0.0) -> float:
    try:
        joint_index = msg.joint_names.index(name)
        values = msg.interface_values[joint_index]
        interface_index = values.interface_names.index(interface)
        return _clean(values.values[interface_index])
    except (ValueError, IndexError):
        return _clean(default)


def _has_name(msg: DynamicJointState, name: str) -> bool:
    return name in msg.joint_names


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def _open_reader(bag_path: Path, storage_id: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)
    return reader


def _topic_type_map(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def _message_time_sec(msg, fallback_stamp_ns: int) -> float:
    stamp = getattr(msg, "header", None)
    if stamp is not None:
        return float(stamp.stamp.sec) + float(stamp.stamp.nanosec) * 1e-9
    return float(fallback_stamp_ns) * 1e-9


def _reference_topic(prefix: str) -> str:
    return f"/{prefix}/reference/targets"


def _fixed_vector(values, size: int) -> list[float]:
    out = [_clean(value) for value in list(values)[:size]]
    if len(out) < size:
        out.extend([0.0] * (size - len(out)))
    return out


def _reference_samples(
    bag_path: Path,
    *,
    storage_id: str,
    robot_prefix: str,
) -> list[tuple[float, dict[str, list[float]]]]:
    reader = _open_reader(bag_path, storage_id)
    topic_types = _topic_type_map(reader)
    reference_topic = _reference_topic(robot_prefix)
    if topic_types.get(reference_topic) != REFERENCE_TOPIC_TYPE:
        return []

    samples = []

    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic != reference_topic:
            continue
        msg = deserialize_message(data, ReferenceTargets)
        samples.append(
            (
                _message_time_sec(msg, stamp),
                {
                    "target_ned_pose": _fixed_vector(msg.ned_pose, 6),
                    "target_body_vel": _fixed_vector(msg.body_velocity, 6),
                    "target_body_acc": _fixed_vector(msg.body_acceleration, 6),
                    "arm_ref": _fixed_vector(msg.arm_position, 5),
                    "arm_dref": _fixed_vector(msg.arm_velocity, 5),
                    "arm_ddref": _fixed_vector(msg.arm_acceleration, 5),
                },
            )
        )

    return samples


def _nearest_reference_sample(
    samples: list[tuple[float, dict[str, list[float]]]],
    stamp_sec: float,
    max_age_sec: float,
) -> dict[str, list[float]] | None:
    if not samples:
        return None
    best_time, best_value = min(samples, key=lambda item: abs(item[0] - stamp_sec))
    if max_age_sec >= 0.0 and abs(best_time - stamp_sec) > max_age_sec:
        return None
    return best_value


def _extract_rows(
    bag_path: Path,
    *,
    storage_id: str,
    topic: str,
    robot_prefix: str,
    sample_period: float,
    target_max_age_sec: float,
    profile_name: str,
    pass_index: int,
    arm_pos_interface: str,
    arm_vel_interface: str,
    arm_acc_interface: str,
    arm_effort_interface: str,
    arm_cmd_interface: str,
) -> list[dict[str, float]]:
    reader = _open_reader(bag_path, storage_id)
    topic_types = _topic_type_map(reader)
    if topic not in topic_types:
        raise RuntimeError(f"topic '{topic}' not found in bag")
    if topic_types[topic] != "control_msgs/msg/DynamicJointState":
        raise RuntimeError(f"topic '{topic}' has type '{topic_types[topic]}', expected control_msgs/msg/DynamicJointState")

    vehicle_name = f"{robot_prefix}IOs"
    arm_names = [f"{robot_prefix}_axis_{axis}" for axis in ARM_AXES]
    arm_state_names = [f"{robot_prefix}_axis_{axis}" for axis in ARM_STATE_AXES]
    arm_ios_name = f"{robot_prefix}_arm_IOs"

    rows: list[dict[str, float]] = []
    first_sim_time: float | None = None
    last_written_time = -math.inf
    target_samples = _reference_samples(
        bag_path,
        storage_id=storage_id,
        robot_prefix=robot_prefix,
    )

    while reader.has_next():
        bag_topic, data, stamp = reader.read_next()
        if bag_topic != topic:
            continue

        msg = deserialize_message(data, DynamicJointState)
        if not _has_name(msg, vehicle_name) and not any(_has_name(msg, name) for name in arm_names):
            continue

        sim_time = _interface_value(msg, vehicle_name, "sim_time", default=math.nan)
        if not math.isfinite(sim_time):
            stamp_msg = msg.header.stamp
            sim_time = float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9
        if first_sim_time is None:
            first_sim_time = sim_time
        replay_time_sec = sim_time - first_sim_time
        if sample_period > 0.0 and rows and (replay_time_sec - last_written_time) < sample_period:
            continue

        q = [_interface_value(msg, name, arm_pos_interface) for name in arm_state_names]
        dq = [_interface_value(msg, name, arm_vel_interface) for name in arm_state_names]
        ddq = [_interface_value(msg, name, arm_acc_interface) for name in arm_state_names]
        effort = [_interface_value(msg, name, arm_effort_interface) for name in arm_state_names]
        cmd_tau = [_interface_value(msg, name, arm_cmd_interface) for name in arm_names]

        pose = [
            _interface_value(msg, vehicle_name, "position.x"),
            _interface_value(msg, vehicle_name, "position.y"),
            _interface_value(msg, vehicle_name, "position.z"),
            _interface_value(msg, vehicle_name, "roll"),
            _interface_value(msg, vehicle_name, "pitch"),
            _interface_value(msg, vehicle_name, "yaw"),
        ]
        body_vel = [
            _interface_value(msg, vehicle_name, "velocity.x"),
            _interface_value(msg, vehicle_name, "velocity.y"),
            _interface_value(msg, vehicle_name, "velocity.z"),
            _interface_value(msg, vehicle_name, "angular_velocity.x"),
            _interface_value(msg, vehicle_name, "angular_velocity.y"),
            _interface_value(msg, vehicle_name, "angular_velocity.z"),
        ]
        body_acc = [
            _interface_value(msg, vehicle_name, "linear_acceleration.x"),
            _interface_value(msg, vehicle_name, "linear_acceleration.y"),
            _interface_value(msg, vehicle_name, "linear_acceleration.z"),
            _interface_value(msg, vehicle_name, "angular_acceleration.x"),
            _interface_value(msg, vehicle_name, "angular_acceleration.y"),
            _interface_value(msg, vehicle_name, "angular_acceleration.z"),
        ]
        body_forces = [
            _interface_value(msg, vehicle_name, "force.x"),
            _interface_value(msg, vehicle_name, "force.y"),
            _interface_value(msg, vehicle_name, "force.z"),
            _interface_value(msg, vehicle_name, "torque.x"),
            _interface_value(msg, vehicle_name, "torque.y"),
            _interface_value(msg, vehicle_name, "torque.z"),
        ]

        reference = _nearest_reference_sample(
            target_samples,
            _message_time_sec(msg, stamp),
            target_max_age_sec,
        ) or {}
        target_pose = reference.get("target_ned_pose", [0.0] * 6)
        target_vel = reference.get("target_body_vel", [0.0] * 6)
        target_acc = reference.get("target_body_acc", [0.0] * 6)
        arm_ref = reference.get("arm_ref", [0.0] * 5)
        arm_dref = reference.get("arm_dref", [0.0] * 5)
        arm_ddref = reference.get("arm_ddref", [0.0] * 5)

        payload_mass = _interface_value(msg, arm_ios_name, "payload.mass", default=0.0)
        gravity = _interface_value(msg, arm_ios_name, "gravity", default=0.0)

        values = (
            q
            + dq
            + ddq
            + arm_ref[:4]
            + arm_dref[:4]
            + arm_ddref[:4]
            + effort
            + cmd_tau
            + pose
            + body_vel
            + body_acc
            + target_pose
            + target_vel
            + target_acc
            + body_forces
            + body_forces
            + [payload_mass, gravity]
        )

        if not _finite(values):
            continue

        row = dict(zip(FIELDNAMES, [0.0] * len(FIELDNAMES)))
        row.update(
            {
                "wall_time_sec": _message_time_sec(msg, stamp),
                "sim_time_sec": sim_time,
                "replay_time_sec": replay_time_sec,
                "profile": profile_name,
                "pass_index": pass_index,
            }
        )

        for key, value in zip(FIELDNAMES[5:], values):
            row[key] = float(value)

        rows.append(row)
        last_written_time = replay_time_sec

    if not rows:
        raise RuntimeError(f"no samples found for robot prefix '{robot_prefix}' on topic '{topic}'")
    return rows


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            out: dict[str, str | int | float] = {}
            for key in FIELDNAMES:
                if key == "profile":
                    out[key] = row.get(key, "")
                elif key == "pass_index":
                    out[key] = int(row.get(key, 0))
                else:
                    out[key] = f"{float(row.get(key, 0.0)):.9f}"
            writer.writerow(out)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a rosbag2 MCAP recording into a CmdReplay-style session CSV."
    )
    parser.add_argument("bag", help="rosbag2 bag directory, not the .mcap file inside it")
    parser.add_argument("output", help="output CSV path")
    parser.add_argument("--robot-prefix", required=True, help="robot prefix to extract, for example robot_1_")
    parser.add_argument("--topic", default="dynamic_joint_states", help="DynamicJointState topic recorded in the bag")
    parser.add_argument("--storage-id", default="mcap", help="rosbag2 storage id")
    parser.add_argument("--sample-period", type=float, default=0.0, help="minimum output sample period in seconds; 0 keeps every sample")
    parser.add_argument("--target-max-age", type=float, default=0.25, help="maximum allowed time offset when aligning desired target topics")
    parser.add_argument("--profile-name", default="", help="value to store in the profile column")
    parser.add_argument("--pass-index", type=int, default=1, help="value to store in the pass_index column")
    parser.add_argument("--arm-pos-interface", default="position", help="arm joint position interface")
    parser.add_argument("--arm-vel-interface", default="velocity", help="arm joint velocity interface")
    parser.add_argument("--arm-acc-interface", default="estimated_acceleration", help="arm joint acceleration interface")
    parser.add_argument("--arm-effort-interface", default="effort", help="arm joint effort interface")
    parser.add_argument("--arm-cmd-interface", default="effort", help="arm command interface for cmd_tau columns")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        raise SystemExit(f"bag path does not exist: {bag_path}")
    if bag_path.is_file():
        raise SystemExit("bag path must be the rosbag2 directory, not the .mcap file")

    output_path = Path(args.output).expanduser().resolve()
    rows = _extract_rows(
        bag_path,
        storage_id=args.storage_id,
        topic=args.topic,
        robot_prefix=args.robot_prefix,
        sample_period=max(0.0, float(args.sample_period)),
        target_max_age_sec=float(args.target_max_age),
        profile_name=str(args.profile_name),
        pass_index=int(args.pass_index),
        arm_pos_interface=str(args.arm_pos_interface),
        arm_vel_interface=str(args.arm_vel_interface),
        arm_acc_interface=str(args.arm_acc_interface),
        arm_effort_interface=str(args.arm_effort_interface),
        arm_cmd_interface=str(args.arm_cmd_interface),
    )
    _write_csv(output_path, rows)

    duration = rows[-1]["replay_time_sec"] - rows[0]["replay_time_sec"]
    print(f"wrote replay recording CSV: {output_path}")
    print(f"samples: {len(rows)}, duration: {duration:.3f}s, robot_prefix: {args.robot_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
