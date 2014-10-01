"""
An Account class for tracking monetary transactions during UrbanSim runs.

"""
from collections import namedtuple


Transaction = namedtuple('Transaction', ('amount', 'subaccount', 'metadata'))


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

        """
        metadata = metadata or {}
        t = Transaction(amount, subaccount, metadata)
        self.transactions.append(t)
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
