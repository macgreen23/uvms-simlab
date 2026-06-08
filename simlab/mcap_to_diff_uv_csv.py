from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import rosbag2_py
from control_msgs.msg import DynamicJointState
from rclpy.serialization import deserialize_message
from simlab.msg import ReferenceTargets

FIELDNAMES = (
    "timestamp,ros_time,base_x_force,base_y_force,base_z_force,base_x_torque,base_y_torque,base_z_torque,"
    "base_x,base_y,base_z,base_roll,base_pitch,base_yaw,base_dx,base_dy,base_dz,base_vel_roll,base_vel_pitch,"
    "base_vel_yaw,effort_alpha_axis_e,effort_alpha_axis_d,effort_alpha_axis_c,effort_alpha_axis_b,"
    "q_alpha_axis_e,q_alpha_axis_d,q_alpha_axis_c,q_alpha_axis_b,dq_alpha_axis_e,dq_alpha_axis_d,dq_alpha_axis_c,"
    "dq_alpha_axis_b,imu_roll,imu_pitch,imu_yaw,imu_roll_unwrap,imu_pitch_unwrap,imu_yaw_unwrap,imu_q_w,imu_q_x,"
    "imu_q_y,imu_q_z,imu_ang_vel_x,imu_ang_vel_y,imu_ang_vel_z,imu_linear_acc_x,imu_linear_acc_y,imu_linear_acc_z,"
    "depth_from_pressure2,dvl_roll,dvl_pitch,dvl_yaw,dvl_speed_x,dvl_speed_y,dvl_speed_z,base_x_ref,base_y_ref,"
    "base_z_ref,base_roll_ref,base_pitch_ref,base_yaw_ref,q_alpha_axis_e_ref,q_alpha_axis_d_ref,q_alpha_axis_c_ref,"
    "q_alpha_axis_b_ref,q_alpha_axis_a_ref,position.x,position.y,position.z,roll,pitch,yaw,orientation.w,"
    "orientation.x,orientation.y,orientation.z,velocity.x,velocity.y,velocity.z,angular_velocity.x,"
    "angular_velocity.y,angular_velocity.z,position_estimate.x,position_estimate.y,position_estimate.z,"
    "roll_estimate,pitch_estimate,yaw_estimate,orientation_estimate.w,orientation_estimate.x,orientation_estimate.y,"
    "orientation_estimate.z,velocity_estimate.x,velocity_estimate.y,velocity_estimate.z,angular_velocity_estimate.x,"
    "angular_velocity_estimate.y,angular_velocity_estimate.z,linear_acceleration.x,linear_acceleration.y,"
    "linear_acceleration.z,angular_acceleration.x,angular_acceleration.y,angular_acceleration.z,P_x_x,P_y_y,P_z_z,"
    "P_roll_roll,P_pitch_pitch,P_yaw_yaw,P_u_u,P_v_v,P_w_w,P_p_p,P_q_q,P_r_r,payload.mass,payload.Ixx,payload.Iyy,"
    "payload.Izz"
).split(",")

ARM_AXES = ("e", "d", "c", "b", "a")
ARM_STATE_AXES = ("e", "d", "c", "b")
REFERENCE_TOPIC_TYPE = "simlab/msg/ReferenceTargets"


def _clean(value: float, eps: float = 1e-12) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return 0.0 if abs(value) < eps else value


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def _interface_value(
    msg: DynamicJointState,
    name: str,
    interface_names: str | tuple[str, ...],
    default: float = 0.0,
) -> float:
    candidates = (interface_names,) if isinstance(interface_names, str) else interface_names
    try:
        joint_index = msg.joint_names.index(name)
        values = msg.interface_values[joint_index]
        for interface in candidates:
            try:
                interface_index = values.interface_names.index(interface)
            except ValueError:
                continue
            return _clean(values.values[interface_index])
    except (ValueError, IndexError):
        pass
    return _clean(default)


def _has_name(msg: DynamicJointState, name: str) -> bool:
    return name in msg.joint_names


def _open_reader(bag_path: Path, storage_id: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)
    return reader


def _topic_type_map(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def _stamp_sec(stamp_ns: int) -> float:
    return float(stamp_ns) * 1e-9


def _message_time_sec(msg, fallback_stamp_ns: int) -> float:
    header = getattr(msg, "header", None)
    if header is not None:
        return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
    return _stamp_sec(fallback_stamp_ns)


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
                _stamp_sec(stamp),
                {
                    "target_ned_pose": _fixed_vector(msg.ned_pose, 6),
                    "target_body_vel": _fixed_vector(msg.body_velocity, 6),
                    "target_body_acc": _fixed_vector(msg.body_acceleration, 6),
                    "arm_ref": _fixed_vector(msg.arm_position, 5),
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


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        _clean(cr * cp * cy + sr * sp * sy),
        _clean(sr * cp * cy - cr * sp * sy),
        _clean(cr * sp * cy + sr * cp * sy),
        _clean(cr * cp * sy - sr * sp * cy),
    ]


def _value_or_quat(
    msg: DynamicJointState,
    name: str,
    quat_fields: tuple[str, str, str, str],
    rpy_fields: tuple[str, str, str],
    *,
    default: float = 0.0,
) -> list[float]:
    quat = [_interface_value(msg, name, field, default=math.nan) for field in quat_fields]
    if all(math.isfinite(value) for value in quat):
        return [_clean(value) for value in quat]
    roll, pitch, yaw = (_interface_value(msg, name, field, default=default) for field in rpy_fields)
    return _rpy_to_quat(roll, pitch, yaw)


def _extract_rows(
    bag_path: Path,
    *,
    storage_id: str,
    topic: str,
    robot_prefix: str,
    sample_period: float,
    target_max_age_sec: float,
    payload_mass: float,
    payload_ixx: float,
    payload_iyy: float,
    payload_izz: float,
) -> list[dict[str, float]]:
    reader = _open_reader(bag_path, storage_id)
    topic_types = _topic_type_map(reader)
    if topic not in topic_types:
        raise RuntimeError(f"topic '{topic}' not found in bag")
    if topic_types[topic] != "control_msgs/msg/DynamicJointState":
        raise RuntimeError(
            f"topic '{topic}' has type '{topic_types[topic]}', expected control_msgs/msg/DynamicJointState"
        )

    vehicle_name = f"{robot_prefix}IOs"
    arm_names = [f"{robot_prefix}_axis_{axis}" for axis in ARM_AXES]
    arm_state_names = [f"{robot_prefix}_axis_{axis}" for axis in ARM_STATE_AXES]
    arm_ios_name = f"{robot_prefix}_arm_IOs"

    rows: list[dict[str, float]] = []
    first_stamp: float | None = None
    first_ros_time: float | None = None
    last_written_time = -math.inf
    target_samples = _reference_samples(bag_path, storage_id=storage_id, robot_prefix=robot_prefix)

    while reader.has_next():
        bag_topic, data, stamp = reader.read_next()
        if bag_topic != topic:
            continue

        msg = deserialize_message(data, DynamicJointState)
        if not _has_name(msg, vehicle_name) and not any(_has_name(msg, name) for name in arm_names):
            continue

        stamp_sec = _stamp_sec(stamp)
        ros_time_abs = _interface_value(msg, vehicle_name, ("sim_time", "ros_time"), default=stamp_sec)
        if first_stamp is None:
            first_stamp = stamp_sec
        if first_ros_time is None:
            first_ros_time = ros_time_abs
        timestamp = stamp_sec - first_stamp
        ros_time = ros_time_abs - first_ros_time
        if sample_period > 0.0 and rows and (timestamp - last_written_time) < sample_period:
            continue

        q = [_interface_value(msg, name, "position") for name in arm_state_names]
        dq = [_interface_value(msg, name, "velocity") for name in arm_state_names]
        effort = [_interface_value(msg, name, "effort") for name in arm_state_names]

        base_pose = [
            _interface_value(msg, vehicle_name, "position.x"),
            _interface_value(msg, vehicle_name, "position.y"),
            _interface_value(msg, vehicle_name, "position.z"),
            _interface_value(msg, vehicle_name, "roll"),
            _interface_value(msg, vehicle_name, "pitch"),
            _interface_value(msg, vehicle_name, "yaw"),
        ]
        base_quat = _value_or_quat(
            msg,
            vehicle_name,
            ("orientation.w", "orientation.x", "orientation.y", "orientation.z"),
            ("roll", "pitch", "yaw"),
        )
        base_vel = [
            _interface_value(msg, vehicle_name, "velocity.x"),
            _interface_value(msg, vehicle_name, "velocity.y"),
            _interface_value(msg, vehicle_name, "velocity.z"),
            _interface_value(msg, vehicle_name, "angular_velocity.x"),
            _interface_value(msg, vehicle_name, "angular_velocity.y"),
            _interface_value(msg, vehicle_name, "angular_velocity.z"),
        ]
        base_acc = [
            _interface_value(msg, vehicle_name, "linear_acceleration.x"),
            _interface_value(msg, vehicle_name, "linear_acceleration.y"),
            _interface_value(msg, vehicle_name, "linear_acceleration.z"),
            _interface_value(msg, vehicle_name, "angular_acceleration.x"),
            _interface_value(msg, vehicle_name, "angular_acceleration.y"),
            _interface_value(msg, vehicle_name, "angular_acceleration.z"),
        ]
        base_forces = [
            _interface_value(msg, vehicle_name, "force.x"),
            _interface_value(msg, vehicle_name, "force.y"),
            _interface_value(msg, vehicle_name, "force.z"),
            _interface_value(msg, vehicle_name, "torque.x"),
            _interface_value(msg, vehicle_name, "torque.y"),
            _interface_value(msg, vehicle_name, "torque.z"),
        ]
        base_est_pose = [
            _interface_value(msg, vehicle_name, "position_estimate.x"),
            _interface_value(msg, vehicle_name, "position_estimate.y"),
            _interface_value(msg, vehicle_name, "position_estimate.z"),
            _interface_value(msg, vehicle_name, "roll_estimate"),
            _interface_value(msg, vehicle_name, "pitch_estimate"),
            _interface_value(msg, vehicle_name, "yaw_estimate"),
        ]
        base_est_quat = _value_or_quat(
            msg,
            vehicle_name,
            (
                "orientation_estimate.w",
                "orientation_estimate.x",
                "orientation_estimate.y",
                "orientation_estimate.z",
            ),
            ("roll_estimate", "pitch_estimate", "yaw_estimate"),
        )
        base_est_vel = [
            _interface_value(msg, vehicle_name, "velocity_estimate.x"),
            _interface_value(msg, vehicle_name, "velocity_estimate.y"),
            _interface_value(msg, vehicle_name, "velocity_estimate.z"),
            _interface_value(msg, vehicle_name, "angular_velocity_estimate.x"),
            _interface_value(msg, vehicle_name, "angular_velocity_estimate.y"),
            _interface_value(msg, vehicle_name, "angular_velocity_estimate.z"),
        ]
        base_est_acc = [
            _interface_value(msg, vehicle_name, "linear_acceleration.x"),
            _interface_value(msg, vehicle_name, "linear_acceleration.y"),
            _interface_value(msg, vehicle_name, "linear_acceleration.z"),
            _interface_value(msg, vehicle_name, "angular_acceleration.x"),
            _interface_value(msg, vehicle_name, "angular_acceleration.y"),
            _interface_value(msg, vehicle_name, "angular_acceleration.z"),
        ]
        imu_roll = _interface_value(msg, vehicle_name, "imu_roll")
        imu_pitch = _interface_value(msg, vehicle_name, "imu_pitch")
        imu_yaw = _interface_value(msg, vehicle_name, "imu_yaw")
        imu_q = _value_or_quat(
            msg,
            vehicle_name,
            ("imu_q_w", "imu_q_x", "imu_q_y", "imu_q_z"),
            ("imu_roll", "imu_pitch", "imu_yaw"),
        )
        imu_ang_vel = [
            _interface_value(msg, vehicle_name, ("imu_ang_vel_x", "imu_angular_vel_x")),
            _interface_value(msg, vehicle_name, ("imu_ang_vel_y", "imu_angular_vel_y")),
            _interface_value(msg, vehicle_name, ("imu_ang_vel_z", "imu_angular_vel_z")),
        ]
        imu_linear_acc = [
            _interface_value(msg, vehicle_name, ("imu_linear_acc_x", "imu_linear_acceleration_x")),
            _interface_value(msg, vehicle_name, ("imu_linear_acc_y", "imu_linear_acceleration_y")),
            _interface_value(msg, vehicle_name, ("imu_linear_acc_z", "imu_linear_acceleration_z")),
        ]
        dvl_roll = _interface_value(msg, vehicle_name, ("dvl_roll", "dvl_gyro_roll"))
        dvl_pitch = _interface_value(msg, vehicle_name, ("dvl_pitch", "dvl_gyro_pitch"))
        dvl_yaw = _interface_value(msg, vehicle_name, ("dvl_yaw", "dvl_gyro_yaw"))
        dvl_speed = [
            _interface_value(msg, vehicle_name, "dvl_speed_x"),
            _interface_value(msg, vehicle_name, "dvl_speed_y"),
            _interface_value(msg, vehicle_name, "dvl_speed_z"),
        ]
        depth = _interface_value(msg, vehicle_name, "depth_from_pressure2")

        base_ref = [0.0] * 6
        q_ref = [0.0] * 5
        reference = _nearest_reference_sample(target_samples, stamp_sec, target_max_age_sec) or {}
        target_pose = reference.get("target_ned_pose", [0.0] * 6)
        arm_ref = reference.get("arm_ref", [0.0] * 5)
        if target_pose:
            base_ref = target_pose[:6]
        if arm_ref:
            q_ref = arm_ref[:5]

        payload_mass_value = _interface_value(msg, arm_ios_name, "payload.mass", default=payload_mass)
        payload_ixx_value = _interface_value(msg, arm_ios_name, ("payload.Ixx", "payload_inertia.x"), default=payload_ixx)
        payload_iyy_value = _interface_value(msg, arm_ios_name, ("payload.Iyy", "payload_inertia.y"), default=payload_iyy)
        payload_izz_value = _interface_value(msg, arm_ios_name, ("payload.Izz", "payload_inertia.z"), default=payload_izz)

        row = {key: 0.0 for key in FIELDNAMES}
        row.update(
            {
                "timestamp": timestamp,
                "ros_time": ros_time,
                "base_x_force": base_forces[0],
                "base_y_force": base_forces[1],
                "base_z_force": base_forces[2],
                "base_x_torque": base_forces[3],
                "base_y_torque": base_forces[4],
                "base_z_torque": base_forces[5],
                "base_x": base_pose[0],
                "base_y": base_pose[1],
                "base_z": base_pose[2],
                "base_roll": base_pose[3],
                "base_pitch": base_pose[4],
                "base_yaw": base_pose[5],
                "base_dx": base_vel[0],
                "base_dy": base_vel[1],
                "base_dz": base_vel[2],
                "base_vel_roll": base_vel[3],
                "base_vel_pitch": base_vel[4],
                "base_vel_yaw": base_vel[5],
                "effort_alpha_axis_e": effort[0],
                "effort_alpha_axis_d": effort[1],
                "effort_alpha_axis_c": effort[2],
                "effort_alpha_axis_b": effort[3],
                "q_alpha_axis_e": q[0],
                "q_alpha_axis_d": q[1],
                "q_alpha_axis_c": q[2],
                "q_alpha_axis_b": q[3],
                "dq_alpha_axis_e": dq[0],
                "dq_alpha_axis_d": dq[1],
                "dq_alpha_axis_c": dq[2],
                "dq_alpha_axis_b": dq[3],
                "imu_roll": imu_roll,
                "imu_pitch": imu_pitch,
                "imu_yaw": imu_yaw,
                "imu_roll_unwrap": _interface_value(msg, vehicle_name, "imu_roll_unwrap"),
                "imu_pitch_unwrap": _interface_value(msg, vehicle_name, "imu_pitch_unwrap"),
                "imu_yaw_unwrap": _interface_value(msg, vehicle_name, "imu_yaw_unwrap"),
                "imu_q_w": imu_q[0],
                "imu_q_x": imu_q[1],
                "imu_q_y": imu_q[2],
                "imu_q_z": imu_q[3],
                "imu_ang_vel_x": imu_ang_vel[0],
                "imu_ang_vel_y": imu_ang_vel[1],
                "imu_ang_vel_z": imu_ang_vel[2],
                "imu_linear_acc_x": imu_linear_acc[0],
                "imu_linear_acc_y": imu_linear_acc[1],
                "imu_linear_acc_z": imu_linear_acc[2],
                "depth_from_pressure2": depth,
                "dvl_roll": dvl_roll,
                "dvl_pitch": dvl_pitch,
                "dvl_yaw": dvl_yaw,
                "dvl_speed_x": dvl_speed[0],
                "dvl_speed_y": dvl_speed[1],
                "dvl_speed_z": dvl_speed[2],
                "base_x_ref": base_ref[0],
                "base_y_ref": base_ref[1],
                "base_z_ref": base_ref[2],
                "base_roll_ref": base_ref[3],
                "base_pitch_ref": base_ref[4],
                "base_yaw_ref": base_ref[5],
                "q_alpha_axis_e_ref": q_ref[0],
                "q_alpha_axis_d_ref": q_ref[1],
                "q_alpha_axis_c_ref": q_ref[2],
                "q_alpha_axis_b_ref": q_ref[3],
                "q_alpha_axis_a_ref": q_ref[4],
                "position.x": base_pose[0],
                "position.y": base_pose[1],
                "position.z": base_pose[2],
                "roll": base_pose[3],
                "pitch": base_pose[4],
                "yaw": base_pose[5],
                "orientation.w": base_quat[0],
                "orientation.x": base_quat[1],
                "orientation.y": base_quat[2],
                "orientation.z": base_quat[3],
                "velocity.x": base_vel[0],
                "velocity.y": base_vel[1],
                "velocity.z": base_vel[2],
                "angular_velocity.x": base_vel[3],
                "angular_velocity.y": base_vel[4],
                "angular_velocity.z": base_vel[5],
                "position_estimate.x": base_est_pose[0],
                "position_estimate.y": base_est_pose[1],
                "position_estimate.z": base_est_pose[2],
                "roll_estimate": base_est_pose[3],
                "pitch_estimate": base_est_pose[4],
                "yaw_estimate": base_est_pose[5],
                "orientation_estimate.w": base_est_quat[0],
                "orientation_estimate.x": base_est_quat[1],
                "orientation_estimate.y": base_est_quat[2],
                "orientation_estimate.z": base_est_quat[3],
                "velocity_estimate.x": base_est_vel[0],
                "velocity_estimate.y": base_est_vel[1],
                "velocity_estimate.z": base_est_vel[2],
                "angular_velocity_estimate.x": base_est_vel[3],
                "angular_velocity_estimate.y": base_est_vel[4],
                "angular_velocity_estimate.z": base_est_vel[5],
                "linear_acceleration.x": base_est_acc[0],
                "linear_acceleration.y": base_est_acc[1],
                "linear_acceleration.z": base_est_acc[2],
                "angular_acceleration.x": base_est_acc[3],
                "angular_acceleration.y": base_est_acc[4],
                "angular_acceleration.z": base_est_acc[5],
                "P_x_x": _interface_value(msg, vehicle_name, "P_x_x"),
                "P_y_y": _interface_value(msg, vehicle_name, "P_y_y"),
                "P_z_z": _interface_value(msg, vehicle_name, "P_z_z"),
                "P_roll_roll": _interface_value(msg, vehicle_name, "P_roll_roll"),
                "P_pitch_pitch": _interface_value(msg, vehicle_name, "P_pitch_pitch"),
                "P_yaw_yaw": _interface_value(msg, vehicle_name, "P_yaw_yaw"),
                "P_u_u": _interface_value(msg, vehicle_name, "P_u_u"),
                "P_v_v": _interface_value(msg, vehicle_name, "P_v_v"),
                "P_w_w": _interface_value(msg, vehicle_name, "P_w_w"),
                "P_p_p": _interface_value(msg, vehicle_name, "P_p_p"),
                "P_q_q": _interface_value(msg, vehicle_name, "P_q_q"),
                "P_r_r": _interface_value(msg, vehicle_name, "P_r_r"),
                "payload.mass": payload_mass_value,
                "payload.Ixx": payload_ixx_value,
                "payload.Iyy": payload_iyy_value,
                "payload.Izz": payload_izz_value,
            }
        )

        numeric_values = [float(value) for key, value in row.items() if key != "timestamp" and key != "ros_time"]
        if not _finite(numeric_values):
            continue

        rows.append(row)
        last_written_time = timestamp

    if not rows:
        raise RuntimeError(f"no samples found for robot prefix '{robot_prefix}' on topic '{topic}'")
    return rows


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (f"{float(row[key]):.9f}" if key not in {"timestamp", "ros_time"} else f"{float(row[key]):.9f}")
                    for key in FIELDNAMES
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a rosbag2 MCAP recording into the CSV format used by diff_uv dynamic identification."
    )
    parser.add_argument("bag", help="rosbag2 bag directory, not the .mcap file inside it")
    parser.add_argument("output", help="output CSV path")
    parser.add_argument("--robot-prefix", required=True, help="robot prefix to extract, for example robot_1_")
    parser.add_argument("--topic", default="dynamic_joint_states", help="DynamicJointState topic recorded in the bag")
    parser.add_argument("--storage-id", default="mcap", help="rosbag2 storage id")
    parser.add_argument("--sample-period", type=float, default=0.0, help="minimum output sample period in seconds")
    parser.add_argument(
        "--target-max-age",
        type=float,
        default=0.25,
        help="maximum allowed time offset when aligning reference targets",
    )
    parser.add_argument("--payload-mass", type=float, default=0.0, help="fallback payload mass for missing samples")
    parser.add_argument("--payload-ixx", type=float, default=0.0, help="fallback payload Ixx for missing samples")
    parser.add_argument("--payload-iyy", type=float, default=0.0, help="fallback payload Iyy for missing samples")
    parser.add_argument("--payload-izz", type=float, default=0.0, help="fallback payload Izz for missing samples")
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
        payload_mass=float(args.payload_mass),
        payload_ixx=float(args.payload_ixx),
        payload_iyy=float(args.payload_iyy),
        payload_izz=float(args.payload_izz),
    )
    _write_csv(output_path, rows)

    duration = rows[-1]["timestamp"] - rows[0]["timestamp"]
    print(f"wrote diff_uv CSV: {output_path}")
    print(f"samples: {len(rows)}, duration: {duration:.3f}s, robot_prefix: {args.robot_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
