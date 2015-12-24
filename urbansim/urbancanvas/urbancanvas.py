import cStringIO

import orca
import numpy as np
import pandas as pd
import psycopg2
import pandas.io.sql as sql
import dataset

def buildings_to_uc(new_buildings, year):
    """
    Parameters
    ----------
    new_buildings : pandas.DataFrame
        DataFrame of buildings to export.
    year : int
        Simulation year
    Returns
    -------
    None
    """

    # Checking that building_id is index or exists as column
    if (new_buildings.index.name != 'building_id') & ('building_id' not in new_buildings.columns):
        print 'Index of buildings must be "building_id" or "building_id" column must exist. Skipping export-to-Urban-Canvas.'
        return None

    if 'building_id' not in new_buildings.columns:
        new_buildings = new_buildings.reset_index()

    # Urban Canvas database connection
    conn_string = orca.get_injectable('conn_string')
    if len(conn_string) == 0:
        print 'A "conn_string" injectable must be registered and populated. Skipping export-to-Urban-Canvas.'
        return None

    if 'uc_conn' not in orca.list_injectables():
        conn = psycopg2.connect(conn_string)
        cur = conn.cursor()

        orca.add_injectable('uc_conn', conn)
        orca.add_injectable('uc_cur', cur)

    else:
        conn = orca.get_injectable('uc_conn')
        cur = orca.get_injectable('uc_cur')

    def exec_sql_uc(query):
        try:
            cur.execute(query)
            conn.commit()
        except:
            conn = psycopg2.connect(conn_string)
            cur = conn.cursor()
            orca.add_injectable('uc_conn', conn)
            orca.add_injectable('uc_cur', cur)
            cur.execute(query)
            conn.commit()

    def get_val_from_uc_db(query):
        try:
            result = sql.read_frame(query, conn)
            return result.values[0][0]
        except:
            conn=psycopg2.connect(conn_string)
            cur = conn.cursor()
            orca.add_injectable('uc_conn', conn)
            orca.add_injectable('uc_cur', cur)
            result = sql.read_frame(query, conn)
            result2 = sql.read_frame("select column_name from Information_schema.columns where table_name like 'building' ", conn)
            print result2
            return result.values[0][0]

    max_bid = get_val_from_uc_db("select max(building_id) FROM building where building_id<100000000;")
    new_buildings.building_id = np.arange(max_bid+1, max_bid+1+len(new_buildings))

    if 'projects_num' not in orca.list_injectables():
        exec_sql_uc("INSERT INTO scenario(id, name, type) select nextval('scenario_id_seq'), 'Run #' || cast(currval('scenario_id_seq') as character varying), 1;")
        nextval = get_val_from_uc_db("SELECT MAX(ID) FROM SCENARIO WHERE ID < 100000;")
        exec_sql_uc("INSERT INTO scenario(id, name, type) select nextval('scenario_id_seq'), 'Run #' || cast(currval('scenario_id_seq') as character varying), 1;")
        nextval = get_val_from_uc_db("SELECT MAX(ID) FROM SCENARIO WHERE ID < 1000000;")
        orca.add_injectable('projects_num', nextval)

        exec_sql_uc("INSERT INTO scenario_project(scenario, project) VALUES(%s, 1);" % nextval)
        exec_sql_uc("INSERT INTO scenario_project(scenario, project) VALUES(%s, %s);" % (nextval, nextval))

    else:
        nextval = orca.get_injectable('projects_num')

    nextval = '{'+str(nextval)+ '}'
    new_buildings['projects'] = nextval

    valid_from = '{'+ str(year) + '-1-1}'
    new_buildings['valid_from'] = valid_from
    print 'Exporting %s buildings to Urban Canvas database for project %s and year %s.' % (len(new_buildings),nextval,year)
    output = cStringIO.StringIO()
    new_buildings.to_csv('buildings_for_eddie.csv')
    new_buildings.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)

    cur.copy_from(output, 'building', columns=tuple(new_buildings.columns.values.astype('U').tolist()))

    test = pd.read_sql("select projects from building where (year_built>2010)", conn)
    print test
    conn.commit()

def get_development_projects():
    conn_string = orca.get_injectable('conn_string')
    if len(conn_string) == 0:
        print 'A "conn_string" injectable must be registered and populated. Skipping export-to-Urban-Canvas.'
        return None
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()

    print "Loading committed development projects table"
    devproj_query = "select id, placetype_id as building_type_id, duration, buildings_number, average_floors as stories, sqft as non_residential_sqft, sqft_unit as sqft_per_unit, units as residential_units, Name as name, start_date from developmentprojects where committed = 'TRUE';"
    devproj = sql.read_frame(devproj_query,conn)
    devproj['year_built'] = devproj.start_date.astype('object').astype('str')
    devproj.year_built = devproj.year_built.str.slice(start=0, stop=4)
    devproj.year_built = devproj.year_built.astype('int')

    print "Loading development project parcels"
    dp_pcl_query = "select developmentprojects_parcels.development_project, developmentprojects_parcels.parcel_id, parcel.parcel_acres from developmentprojects_parcels, parcel where developmentprojects_parcels.parcel_id = parcel.parcel_id;"
    dp_pcl = sql.read_frame(dp_pcl_query, conn)
    devproject_parcel_ids = dp_pcl.groupby('development_project').parcel_id.max().reset_index()  ##In future, use the parcel_acres field on this tbl too

    scheduled_development_events = pd.merge(devproject_parcel_ids, devproj, left_on='development_project', right_on='id')
    scheduled_development_events = scheduled_development_events.rename(columns={'development_project':'scheduled_development_event_id',
                                                                                'building_type_id':'development_type_id'})
    scheduled_development_events = scheduled_development_events[['scheduled_development_event_id', 'year_built', 'development_type_id', 'stories', u'non_residential_sqft', 'sqft_per_unit', 'residential_units', 'parcel_id']]
    for col in scheduled_development_events:
        scheduled_development_events[col] = scheduled_development_events[col].astype('int')

    return scheduled_development_events