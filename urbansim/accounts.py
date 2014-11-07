"""
An Account class for tracking monetary transactions during UrbanSim runs.

"""
from collections import namedtuple

import pandas as pd
import toolz


Transaction = namedtuple('Transaction', ('amount', 'subaccount', 'metadata'))

# column names that are always present in DataFrames of transactions
COLS = ['amount', 'subaccount']


def _column_names_from_metadata(dicts):
    """
    Get the unique set of keys from a list of dictionaries.

    Parameters
    ----------
    dicts : iterable
        Sequence of dictionaries.

    Returns
    -------
    keys : list
        Unique set of keys.

    """
    return list(toolz.unique(toolz.concat(dicts)))


class Account(object):
    """
    Keeps a record of transactions, metadata, and a running balance.

    Parameters
    ----------
    name : str
        Arbitrary name for this account used in some output.
    balance : float, optional
        Starting balance for the account.

    """
    def __init__(self, name, balance=0):
        self.name = name
        self.balance = balance
        self.transactions = []

    def add_transaction(self, amount, subaccount=None, metadata=None):
        """
        Add a new transaction to the account.

        Parameters
        ----------
        amount : float
            Negative for withdrawls, positive for deposits.
        subaccount : object, optional
            Any indicator of a subaccount to which this transaction applies.
        metadata : dict, optional
            Any extra metadata to record with the transaction.
            (E.g. Info about where the money is coming from or going.)
            May not contain keys 'amount' or 'subaccount'.

        """
        metadata = metadata or {}
        self.transactions.append(Transaction(amount, subaccount, metadata))
        self.balance += amount

    def add_transactions(self, transactions):
        """
        Add a collection of transactions to the account.

        Parameters
        ----------
        transactions : iterable
            Should be tuples of amount, subaccount, and metadata as would
            be passed to `add_transaction`.

        """
        for t in transactions:
            self.add_transaction(*t)

    def to_frame(self):
        """
        Return transactions as a pandas DataFrame.

        """
        col_names = _column_names_from_metadata(
            t.metadata for t in self.transactions)

        trow = lambda t: (
            toolz.concatv(
                (t.amount, t.subaccount),
                (t.metadata.get(c) for c in col_names)))
        rows = [trow(t) for t in self.transactions]

        return pd.DataFrame(rows, columns=COLS + col_names)