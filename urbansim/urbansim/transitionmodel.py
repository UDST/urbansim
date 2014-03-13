import numpy as np
import pandas as pd


def estimate(dset, config, year=None, show=True, variables=None):
    return


def simulate(dset, config, year=None, show=True, variables=None):
    assert "table" in config
    assert "writetotmp" in config
    assert "geography_field" in config
    assert "control_totals" in config or "vacancy_targets" in config
    assert "amount_field" in config or "vacancy_targets" in config
    assert "first_year" in config
    year_step = 1 if "year_step" not in config else config["year_step"]
    first_year = config["first_year"]
    curyear = year
    prevyear = curyear - year_step
    hhs = eval(config["table"])
    outtblname = config["writetotmp"]
    if curyear == first_year:
        hhs["_year_added_"] = np.array([curyear] * len(hhs.index))
        dset.save_tmptbl(outtblname, hhs)
        return

    if show:
        print hhs.describe()

    if "control_totals" in config:
        control_totals = eval(config["control_totals"])
        cur_ct = control_totals.ix[curyear]
        prev_ct = control_totals.ix[prevyear]

    if "vacancy_targets" in config:

        va_cfg = config["vacancy_targets"]
        assert "targets" in va_cfg and \
            "supply" in va_cfg and \
            "demands" in va_cfg
        demands = va_cfg["demands"]
        num = eval(demands[0])
        for item in demands[1:]:
            num = num.add(eval(item), fill_value=0)
        denom = eval(va_cfg["supply"])
        print "Numerator:\n", num
        print "Denominator:\n", denom
        vacancy = (denom - num) / denom
        if "negative_vacancy" in config and \
                config["negative_vacancy"] is False:
            vacancy[vacancy < 0] = 0
        print "Vacancy = (denom-num)/denom:\n", vacancy
        targets = eval(va_cfg["targets"])
        target_vacancy = targets[year]
        print "Minimum vacancy (from target_vacancy table):\n", target_vacancy
        vacancy_diff = (target_vacancy - vacancy).dropna()
        print "Vacancy diff = target-actual:\n", vacancy_diff
        newunits = (vacancy_diff[vacancy_diff > 0] * denom).dropna()
        print "New units to build (vacancy_diff * denom):\n", newunits
        control_totals = cur_ct = newunits.reset_index()
        prev_ct = None
        config["amount_field"] = 0

    cols = []
    for col in control_totals.columns:
        if col != config["amount_field"]:
            cols.append(col)
    if type(cur_ct) == pd.DataFrame:
        if prev_ct is not None:
            cnt = cur_ct.reset_index(drop=True).set_index(
                cols) - prev_ct.reset_index(drop=True).set_index(cols)
        else:
            cnt = cur_ct.reset_index(drop=True).set_index(cols)
    else:
        cnt = cur_ct - prev_ct
    print "Adding %d agents" % cnt.sum()
    newhh = []
    if type(cur_ct) == pd.DataFrame:
        for row in cnt.iterrows():
            index, row = row
            subset = hhs
            if type(index) in [np.int32, np.int64]:
                index = [index]
            for col, name in zip(index, cols):
                subset = subset[subset[name] == col]
            num = row.values[0]
            if num == 0:
                continue
            tmphh = hhs.ix[np.random.choice(subset.index.values, num)]
            if "size_field" in config:
                tmphh = tmphh[
                    np.cumsum(tmphh[config["size_field"]].values) < num]
            newhh.append(tmphh)
    else:
        num = cnt.values[0]
        if num != 0:
            newhh.append(
                hhs.ix[np.random.choice(hhs.index.values, num, replace=False)])

    if not newhh:
        return  # no new agents
    newhh = pd.concat(newhh)
    newhh[config["geography_field"]] = -1
    newhh["_year_added_"] = np.array([curyear] * len(newhh.index))

    if hhs.index.values.dtype not in [np.int32, np.int64]:
        raise Exception("Only unique integer labels are allowed")
    newhh = newhh.set_index(
        np.arange(len(newhh.index)) + np.amax(hhs.index.values) + 1)
    hhs = pd.concat([hhs, newhh])
    dset.save_tmptbl(outtblname, hhs)
    dset.save_table(outtblname)

    print "Histogram of agents by year:\n", hhs._year_added_.value_counts()
