import argparse

from .. import __version__
from .run_api import RunDiffuzersAPICommand
from .run_app import RunDiffuzersAppCommand


def main():
    parser = argparse.ArgumentParser(
        "Diffuzers CLI",
        usage="diffuzers <command> [<args>]",
        epilog="For more information about a command, run: `diffuzers <command> --help`",
    )
    parser.add_argument("--version", "-v", help="Display diffuzers version", action="store_true")
    commands_parser = parser.add_subparsers(help="commands")

    # Register commands
    RunDiffuzersAppCommand.register_subcommand(commands_parser)
    RunDiffuzersAPICommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)
    command.run()


if __name__ == "__main__":
    main()
