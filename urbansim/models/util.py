def apply_filter_query(df, filters=None):
    """
    Use the DataFrame.query method to filter a table down to the
    desired rows.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : list of str, optional
        List of filters to apply. Will be joined together with
        ' and ' and passed to DataFrame.query.
        If not supplied no filtering will be done.

    Returns
    -------
    filtered_df : pandas.DataFrame

    """
    if filters:
        query = ' and '.join(filters)
        return df.query(query)
    else:
        return df
