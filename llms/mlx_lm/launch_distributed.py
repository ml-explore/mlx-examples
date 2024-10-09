import argparse
import os
import sys
from subprocess import DEVNULL, TimeoutExpired, run
from tempfile import NamedTemporaryFile


def build_parser():
    parser = argparse.ArgumentParser(
        description="MLX-LM MPI helper.",
        epilog="""
            The goal of this helper is to make it a bit easier to launch MPI
            jobs on Mac with MLX. It will try to make sure that the machines
            are reachable with ssh, that MPI is available to all 

            Example:

                # Launch with host names
                mlx_lm.launch_distributed --hosts studio-0,studio-1 -- python -m mlx_lm.lora --train ...

                # Launch with IPs
                mlx_lm.launch_distributed --hosts 1.2.3.4,1.2.3.5 -- python -m mlx_lm.lora --train ...

                # More complicated launch which sets various MPI parameters
                mlx_lm.launch_distributed --hosts studio-0,studio-1 --connections-per-peer 4 \\
                    --env MY_ENV_VAR=1 --env MY_ENV_VAR=2 -- python -m mlx_lm.lora --train ...
            """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--hosts",
        help="Hosts to run the command on with MPI",
    )
    parser.add_argument(
        "--hostfile",
        help="A plaintext file with one host per line",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Set environment variables to all nodes",
    )
    parser.add_argument(
        "--connections-per-peer",
        type=int,
        default=4,
        help="How many TCP connections to open between hosts",
    )
    parser.add_argument(
        "--skip-ssh-check",
        action="store_true",
        help="Do not check for ssh connectivity before launching",
    )
    parser.add_argument(
        "--skip-python-path",
        action="store_true",
        help="Do not use the current executable as the python path",
    )

    parser.add_argument("cmd", nargs="+", help="The command to run")

    return parser


def parse_hosts(args):
    if args.hosts is not None:
        return args.hosts.split(",")
    elif args.hostfile is not None:
        with open(args.hostfile, "r") as f:
            return [l.strip() for l in f]
    else:
        raise ValueError("Either --hosts or --hostfile need to be provided")


def check_connection(hosts):
    for h in hosts:
        success = True
        try:
            ret = run(["ssh", h, "ls"], stdout=DEVNULL, stderr=DEVNULL, timeout=1)
        except TimeoutExpired:
            success = False
        if not success or ret.returncode != 0:
            raise ValueError(
                f"Failed to connect to '{h}' via ssh. "
                f"Make sure that '{h}' is accessible without a password."
            )


def get_mpirun():
    ret = run(["which", "mpirun"], capture_output=True)
    if ret.returncode != 0:
        raise ValueError(
            "Couldn't find 'mpirun' in the path. " "Make sure you have MPI installed."
        )
    return ret.stdout


def main():
    parser = build_parser()
    args = parser.parse_args()

    hosts = parse_hosts(args)

    if not args.skip_ssh_check:
        check_connection(hosts)

    # Get mpirun and python executables
    cmd = args.cmd
    mpirun = get_mpirun()
    executable = cmd.pop(0)
    if executable == "python":
        executable = sys.executable

    # Make the hostfile
    with NamedTemporaryFile() as hostfile:
        for h in hosts:
            hostfile.write(f"{h} slots=1\n")
        hostfile.flush()

        mpiargs = []
        for e in env:
            mpiargs.extend(("-x", e))
        mpiargs.extend(("--mca", "btl_tcp_links", str(args.connections_per_peer)))

        run_args = [
            mpirun,
            *mpiargs,
            "--hostfile",
            hostfile.name,
            "--",
            executable,
            *cmd,
        ]
        print(f"Running `{''.join(run_args)}`", flush=True)
        os.execv(run_args.pop(0), run_args)


if __name__ == "__main__":
    main()
