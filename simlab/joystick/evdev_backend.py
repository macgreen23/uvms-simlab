from pathlib import Path

from evdev import InputDevice, ecodes
import yaml

INPUT_DEVICES = Path("/dev/input/by-id")


def discover_joysticks() -> list[Path]:
    """Discover event-joystick symlinks in /dev/input/by-id."""
    return sorted(
        device
        for device in INPUT_DEVICES.glob("*-event-joystick")
        if device.is_symlink() and device.exists()
    )

def load_configs() -> list[dict]:
    """Load joystick configuration from YAML file."""
    config_path = Path(__file__).parent / "joystick_config.yaml"
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return []

    with open(config_path, "r") as f:
        try:
            configs = yaml.safe_load(f)
            if not isinstance(configs, list):
                print("Configuration file must contain a list of joystick configs.")
                print(f"Invalid configuration format: {configs}")
                return []
            return configs
        except yaml.YAMLError as exc:
            print(f"Error parsing configuration file: {exc}")
            return []

def filter_joysticks(
    joysticks: list[Path],
    vendor_ids: set[int] | None = None,
    name_keywords: list[str] | None = None,
) -> list[Path]:
    """Filter discovered devices by actual metadata and gamepad capabilities."""
    filtered = []

    for path in joysticks:
        try:
            device = InputDevice(str(path))
            try:
                if (
                    vendor_ids is not None
                    and device.info.vendor not in vendor_ids
                ):
                    print(f"Skipping {path}: vendor ID not in configured list. Found: 0x{device.info.vendor:04x}")
                    continue

                if name_keywords is not None:
                    name = (device.name or "").casefold()
                    if not any(
                        keyword.casefold() in name
                        for keyword in name_keywords
                    ):
                        print(f"Skipping {path}: name does not match keywords")
                        continue

                filtered.append(path)
            finally:
                device.close()

        except OSError as exc:
            print(f"Could not inspect {path}: {exc}")

    return filtered


def main() -> None:
    # configured_joysticks = [
    #     {
    #         "vendor_id": 0x045E,
    #         "name_keywords": ["microsoft", "xbox"],
    #     },
    #     {
    #         "vendor_id": 0x054C,
    #         "name_keywords": ["sony", "dualshock"],
    #     },
    # ]

    configured_joysticks = load_configs()
    print(f"Loaded {len(configured_joysticks)} joystick configurations.")
    joysticks = discover_joysticks()

    # Each config is a separate matching rule.
    filtered_joysticks = []

    for config in configured_joysticks:
        matches = filter_joysticks(
            joysticks,
            vendor_ids={config["vendor_id"]},
            name_keywords=config["name_keywords"],
        )
        filtered_joysticks.extend(matches)

    # Deduplicate devices matching multiple rules.
    filtered_joysticks = list(dict.fromkeys(filtered_joysticks))

    if not filtered_joysticks:
        print("No configured gamepad devices found.")
        return

    print("Configured gamepad devices:")
    for joystick in filtered_joysticks:
        device = InputDevice(str(joystick))
        try:
            print(
                f" - {joystick.name}\n"
                f"   Device: {device.name}\n"
                f"   Vendor: 0x{device.info.vendor:04x}\n"
                f"   Product: 0x{device.info.product:04x}"
            )
        finally:
            device.close()


if __name__ == "__main__":
    main()
