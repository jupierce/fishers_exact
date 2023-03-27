from typing import Any, Dict
from pypika import Query, Table, Field
from pypika import functions as fn


class BigQueryQuery(Query):
    @classmethod
    def _builder(cls, **kwargs: Any) -> "QueryBuilder":
        builder = super()._builder(**kwargs)
        fn.Timestamp()
        builder.QUOTE_CHAR = '`'
        return builder


class BigQueryTable(Table):

    def __init__(self, name: str):
        super().__init__(name, query_cls=BigQueryQuery)

    def get_sql(self, **kwargs: Any) -> str:
        if kwargs is None:
            kwargs = dict()
        # bigquery uses backtick for table names
        kwargs['quote_char'] = '`'
        return super().get_sql(**kwargs)


class Datetime(fn.Function):
    def __init__(self, term, alias=None):
        super(Datetime, self).__init__("DATETIME", term, alias=alias)


def and_of_all_fields(and_all: Dict[Field, Any]):
    expr = None
    for field, val in and_all.items():
        if expr is None:
            expr = field == val
        else:
            expr &= field == val
    return expr
