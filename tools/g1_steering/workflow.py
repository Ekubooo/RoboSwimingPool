"""Fresh G1 Steering RL with the released frozen LaFAN prior and published budget."""
from tools.g1_dodgeball.lafan import main


if __name__ == "__main__":
    raise SystemExit(main(task="steering"))
