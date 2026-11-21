"""Default: metadata-only plan on stdout. --initialize writes only plan manifests."""
import argparse
import json

from .checkpoints import Checkpoints
from .plan import build_plan, OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize', metavar='RUN_ID', help='Create a new no-fit run manifest')
    parser.add_argument('--resume', metavar='RUN_ID', help='Verify an existing no-fit run')
    args = parser.parse_args()
    if args.initialize and args.resume:
        parser.error('Choose initialize or resume, not both')
    plan = build_plan()
    if args.initialize:
        store = Checkpoints.create(OUTPUT_ROOT, args.initialize, plan)
        print(f'Initialized no-fit plan: {store.path}')
    elif args.resume:
        store = Checkpoints.resume(OUTPUT_ROOT, args.resume, plan)
        print(f'Verified no-fit plan: {store.path}')
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
