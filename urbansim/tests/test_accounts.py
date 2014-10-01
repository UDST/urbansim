import pytest

from .. import accounts


@pytest.fixture(scope='module')
def acc_name():
    return 'test'


@pytest.fixture(scope='module')
def acc_bal():
    return 1000


@pytest.fixture
def acc(acc_name, acc_bal):
    return accounts.Account(acc_name, acc_bal)


def test_init(acc, acc_name):
    assert acc.name == acc_name
    assert acc.balance == 1000
    assert acc.transactions == []


def test_add_transaction(acc, acc_bal):
    amount = -50
    subaccount = ('a', 'b', 'c')
    metadata = {'for': 'light speed engine'}
    acc.add_transaction(amount, subaccount, metadata)

    assert len(acc.transactions) == 1
    assert acc.balance == acc_bal + amount

    t = acc.transactions[-1]
    assert isinstance(t, accounts.Transaction)
    assert t.amount == amount
    assert t.subaccount == subaccount
    assert t.metadata == metadata


def test_add_transactions(acc, acc_bal):
    t1 = accounts.Transaction(200, ('a', 'b', 'c'), None)
    t2 = (-50, None, {'to': 'Acme Corp.'})
    acc.add_transactions((t1, t2))

    assert len(acc.transactions) == 2
    assert acc.balance == acc_bal + t1[0] + t2[0]
