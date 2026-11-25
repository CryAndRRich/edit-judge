"""Apply an explicitly approved group proposal without inventing independent votes."""

import json

from ...common.io import write_json, write_jsonl
from ...paths import FINAL_PAIRS
from .common import dataset, digest, fingerprint, frozen_json, round_dir, selected
from .merge import validate_decision


def approve(number: int, approver: str, confirmation: str) -> dict:
    if not approver.strip() or not confirmation.strip():
        raise ValueError('An approver and explicit confirmation are required')
    folder = round_dir(number)
    if (folder / 'applied.json').exists():
        raise ValueError('This round is already applied; use a later round for corrections')
    document = json.loads((folder / 'group-proposals.json').read_text(encoding='utf-8'))
    mapping = json.loads((folder / 'draft-map.json').read_text(encoding='utf-8'))
    picks = selected(number)
    _, samples = dataset()
    ids = {row['id'] for row in picks}
    if (not document.get('draft_complete') or set(document['records']) != ids
            or len(mapping['items']) != len(ids) or {r['id'] for r in mapping['items']} != ids
            or mapping['pairs_sha256'] != fingerprint(FINAL_PAIRS)
            or mapping['selection_sha256'] != fingerprint(folder / 'selected.jsonl')):
        raise ValueError('Proposal, mapping, selection or source labels do not match')
    decisions = []
    for index, item in enumerate(mapping['items']):
        record = document['records'][item['id']]
        keys = set(samples[item['id']]['keys'])
        if (record['index'] != index or record['id'] != item['id']
                or set(item['labels'].values()) != keys):
            raise ValueError('Proposal index or edit mapping changed')
        retained = bool(record['skipped'])
        if not retained and record['ranking'] != [item['labels'][k] for k in record['displayed_ranking']]:
            raise ValueError('Displayed and original-key rankings differ')
        decision = {'id': item['id'], 'reviewers': [approver.strip()],
                    'review_mode': 'user_approved_group_proposal', 'removed': False,
                    'retained_original': retained,
                    'ranking': samples[item['id']]['ranking'] if retained else record['ranking'],
                    'confidence': 0.8, 'uncertain_pairs': [] if retained else record['uncertain_pairs'],
                    'reason': record['reason'], 'approval_confirmation': confirmation.strip()}
        validate_decision(decision, keys, [approver.strip()])
        decisions.append(decision)
    approval = {'round': number, 'review_mode': 'user_approved_group_proposal',
                'approver': approver.strip(), 'confirmation': confirmation.strip(),
                'proposal_group': document['reviewer'], 'independent_reviews': 0,
                'proposals_sha256': fingerprint(folder / 'group-proposals.json'),
                'mapping_sha256': fingerprint(folder / 'draft-map.json'),
                'selection_sha256': fingerprint(folder / 'selected.jsonl'),
                'pairs_sha256': fingerprint(FINAL_PAIRS), 'decisions_sha256': digest(decisions)}
    frozen_json(folder / 'user-approval.json', approval)
    write_jsonl(folder / 'decisions.jsonl', decisions)
    result = {**approval, 'complete': True, 'decisions': len(decisions), 'pending': 0,
              'retained_samples': sum(r['retained_original'] for r in decisions),
              'pairwise_agreement': None, 'krippendorff_alpha': None, 'calibration_alpha': None,
              'calibration_samples_received': 0}
    write_json(folder / 'merge-summary.json', result)
    return result


def validate_approval(number, decisions):
    """Require the frozen explicit approval, not just a caller-supplied mode flag."""
    folder = round_dir(number)
    approval = json.loads((folder / 'user-approval.json').read_text(encoding='utf-8'))
    if (approval['review_mode'] != 'user_approved_group_proposal' or not approval['confirmation'].strip()
            or approval['decisions_sha256'] != digest(decisions)
            or approval['pairs_sha256'] != fingerprint(FINAL_PAIRS)
            or approval['proposals_sha256'] != fingerprint(folder / 'group-proposals.json')
            or approval['mapping_sha256'] != fingerprint(folder / 'draft-map.json')
            or approval['selection_sha256'] != fingerprint(folder / 'selected.jsonl')):
        raise ValueError('Approved proposal or its inputs changed')
    return approval
