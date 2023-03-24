from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import expression

import re

_junit_table_engine = None
_table = None


def junit_table():
    global _junit_table_engine, _table
    if _table is None:
        _junit_table_engine = create_engine('bigquery://openshift-gce-devel')
        _table = Table('ci_analysis_us.junit', MetaData(bind=_junit_table_engine), autoload=True)
    return _table


# https://docs.sqlalchemy.org/en/14/orm/declarative_styles.html
Base = declarative_base()


DATETIME_INPUT_PATTERN = re.compile(r'(\d\d\d\d-\d\d-\d\d)[T: ]?(\d\d:\d\d).*')


class Junit(Base):
    # https://docs.sqlalchemy.org/en/14/orm/mapping_columns.html
    __table__ = junit_table()
    __mapper_args__ = {'primary_key': [__table__.c.schema_level, __table__.c.file_path, __table__.c.test_id, __table__.c.success_val]}

    schema_level = __table__.c.schema_level
    prowjob_build_id = __table__.c.prowjob_build_id
    file_path = __table__.c.file_path
    test_name = __table__.c.test_name
    duration_ms = __table__.c.duration_ms
    success = __table__.c.success
    skipped = __table__.c.skipped
    modified_time = __table__.c.modified_time
    test_id = __table__.c.test_id
    branch = __table__.c.branch
    prowjob_name = __table__.c.prowjob_name
    success_val = __table__.c.success_val
    network = __table__.c.network
    platform = __table__.c.platform
    arch = __table__.c.arch
    upgrade = __table__.c.upgrade
    variants = __table__.c.variants
    flake_count = __table__.c.flake_count

    @classmethod
    def platform_drill_key(cls) -> func.GenericFunction:
        return func.concat(cls.network, ' ', cls.upgrade, ' ', cls.platform, ' ', cls.arch)

    @classmethod
    def format_modified_time(cls, s: str):
        m = DATETIME_INPUT_PATTERN.match(s)
        if not m:
            raise ValueError(f'Invalid datetime format: {s}')
        return func.datetime(func.timestamp(m.group(1) + ' ' + m.group(2) + ':00+00'))


select = expression.select
any_value = func.any_value
count = func.count
sum = func.sum

