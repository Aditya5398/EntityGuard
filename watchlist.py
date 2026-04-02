"""
watchlist.py
Denied party watchlist + test transactions.
Mirrors OFAC SDN list structure with name aliases and metadata.
"""

import pandas as pd

WATCHLIST = [
    {'id': 1, 'name': 'Al Baraka Trading Company',
     'aliases': ['Al-Baraka Trade Co', 'Albaraka Trading', 'Al Baraka Trade'],
     'country': 'Iran', 'type': 'entity'},

    {'id': 2, 'name': 'Mohammad Hassan Al-Rashid',
     'aliases': ['Mohammed H. Al Rashid', 'M.H. Alrashid', 'Mohammad Al Rashid'],
     'country': 'Syria', 'type': 'individual'},

    {'id': 3, 'name': 'Global Maritime Holdings LLC',
     'aliases': ['Global Maritime Holdings', 'GMH Trading', 'Global Maritime Ltd'],
     'country': 'North Korea', 'type': 'entity'},

    {'id': 4, 'name': 'Aziz Khan Export Import',
     'aliases': ['A.K. Export Import', 'Aziz Khan Exports', 'AK Export Import Co'],
     'country': 'Pakistan', 'type': 'entity'},

    {'id': 5, 'name': 'Viktor Petrov Consulting',
     'aliases': ['V. Petrov Consulting', 'Petrov Consulting Group', 'Viktor Petroff'],
     'country': 'Russia', 'type': 'entity'},

    {'id': 6, 'name': 'Huang Wei International',
     'aliases': ['H.W. International', 'HWI Trading', 'Huang Wei Intl'],
     'country': 'China', 'type': 'entity'},

    {'id': 7, 'name': 'Tariq Abdullah Al-Mansouri',
     'aliases': ['T.A. Al Mansouri', 'Tariq A. Almansouri', 'Tariq Almansouri'],
     'country': 'Yemen', 'type': 'individual'},

    {'id': 8, 'name': 'Caspian Sea Petroleum Corp',
     'aliases': ['Caspian Petroleum', 'CSP Corp', 'Caspian Sea Petro'],
     'country': 'Iran', 'type': 'entity'},
]

# Transactions to screen — 5 are real matches, 5 are clean
TEST_TRANSACTIONS = [
    # SHOULD MATCH
    {'tx_id': 'T001', 'counterparty': 'Al Baraka Trading Co.', 'expected_match': 1},
    {'tx_id': 'T002', 'counterparty': 'Mohammed Hassan Al Rasheed', 'expected_match': 2},
    {'tx_id': 'T003', 'counterparty': 'Global Maritime Holding LLC', 'expected_match': 3},
    {'tx_id': 'T004', 'counterparty': 'Aziz Khan Export & Import', 'expected_match': 4},
    {'tx_id': 'T005', 'counterparty': 'Viktor Petroff Consulting', 'expected_match': 5},
    # SHOULD NOT MATCH (clean)
    {'tx_id': 'T006', 'counterparty': 'Amazon Global Trading Inc.', 'expected_match': None},
    {'tx_id': 'T007', 'counterparty': 'Pacific Maritime Holdings', 'expected_match': None},
    {'tx_id': 'T008', 'counterparty': 'John Smith Consulting LLC', 'expected_match': None},
    {'tx_id': 'T009', 'counterparty': 'Ali Hassan Import Export', 'expected_match': None},
    {'tx_id': 'T010', 'counterparty': 'Global Trading Partners Inc', 'expected_match': None},
]

COMPLIANCE_DOCS = [
    {
        'doc_id': 'OFAC-001',
        'title': 'Iran Transactions Regulations 31 CFR 560',
        'content': (
            "Section 560.204 Prohibited transactions. No US person may engage in any "
            "transaction with Iran or involving Iranian property unless licensed by OFAC. "
            "Iranian entities include companies incorporated in Iran. "
            "Key terms: Al Baraka, Caspian, Tehran, NIOC indicate Iranian connection."
        )
    },
    {
        'doc_id': 'OFAC-002',
        'title': 'Syria Sanctions Executive Order 13582',
        'content': (
            "Blocks property of Syrian government and key associates. "
            "Prohibited: any US person dealing with listed Syrian individuals. "
            "Names on list include variations: Al, Al-, Al_ are equivalent prefixes. "
            "Common given name spellings: Mohammed, Mohammad, Muhammad are equivalent."
        )
    },
    {
        'doc_id': 'POLICY-001',
        'title': 'Amazon Screening Policy v3.2',
        'content': (
            "Threshold for automatic block: Bayesian posterior above 0.85. "
            "Threshold for manual review: Bayesian posterior above 0.30. "
            "LLM confidence required to override manual review: above 0.90. "
            "Name variations to check: LLC Ltd Corp, Al Al- Al_, "
            "Mohammed Mohammad Muhammad, abbreviated middle initials."
        )
    }
]


def get_expanded_watchlist():
    """Return watchlist with aliases expanded as separate rows."""
    rows = []
    for entry in WATCHLIST:
        rows.append({'watch_id': entry['id'], 'name': entry['name']})
        for alias in entry['aliases']:
            rows.append({'watch_id': entry['id'], 'name': alias})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = get_expanded_watchlist()
    print(f"Watchlist: {len(WATCHLIST)} entities, {len(df)} total names")
    print(df.to_string(index=False))
