"""Explicit user-approved successor of the expired first DOC pilot; no deadline edits."""
from . import doc_pilot_io as rt

DONOR = rt.OUTPUT / 'doc-pilot-20261120t105740310195z'
REQUEST_SHA = 'ed82ca9f91fc4b89d475f1672938b7bd6142aef5deedc4234249ddd9ae30b598'
COMMIT_SHA = 'bb9391c412475755980a35c660d91fef453b03667dd2b2c36275047d9b673d4c'
COMMIT_NAME = 'commit-004376.json'


def binding():
    rt.require(rt.file_hash(DONOR / 'request.json') == REQUEST_SHA, 'Donor request drift')
    commits = sorted((DONOR / 'fit').glob('commit-*.json'))
    rt.require(commits and commits[-1].name == COMMIT_NAME and rt.file_hash(commits[-1]) == COMMIT_SHA,
               'Donor final commit drift')
    rt.require(not (DONOR / 'fit-complete.json').exists() and not (DONOR / 'seal.json').exists(),
               'Expected incomplete unsealed donor')
    record = rt.read(commits[-1])
    payload = commits[-1].with_name(COMMIT_NAME.replace('commit-', 'payload-').replace('.json', '.pt'))
    rt.require(rt.file_hash(payload) == record['payload_sha256'], 'Donor state payload drift')
    return {'authorization_date': '2026-11-20', 'donor_run_id': DONOR.name,
            'donor_request_sha256': REQUEST_SHA, 'donor_commit': COMMIT_NAME,
            'donor_commit_sha256': COMMIT_SHA, 'donor_payload_sha256': record['payload_sha256'],
            'completed_epochs': 38, 'next_epoch': 39, 'cursor': 534528, 'updates': 381250,
            'best_epoch': 38, 'best_validation_bce': 0.4778781863738218, 'bad_epochs': 0,
            'total_epoch_cap': 100, 'original_deadline_preserved': True,
            'renewed_fit_seconds': 7200, 'renewed_postfit_seconds': 900}
