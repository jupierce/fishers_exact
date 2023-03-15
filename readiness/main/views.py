import re
import math
import mmap
from typing import Dict, NamedTuple, List
from enum import Enum

from django.utils.html import format_html
from django.http import HttpResponse
import django_tables2 as tables

import fast_fisher.fast_fisher_cython

from django.shortcuts import render
from readiness.settings import BASE_DIR

from google.cloud import bigquery

TestId = str
ComponentName = str
CapabilityName = str

MAX_CELL_SAMPLES = 512   # must be a power of 2 and divisible by 8
MULTIPLICATION_SHIFTS = int(math.log2(MAX_CELL_SAMPLES))

if (1 << MULTIPLICATION_SHIFTS) != MAX_CELL_SAMPLES or MAX_CELL_SAMPLES//8*8 != MAX_CELL_SAMPLES:
    print(f'MAX_CELL_SAMPLES must be a power of 2 and divisible by 8')
    exit(1)

ALPHA = 0.05

table_bin = open(f'{BASE_DIR}/../table.bin', 'rb')
table_mmap = mmap.mmap(table_bin.fileno(), length=0, access=mmap.ACCESS_READ)


def fisher_offset(a, b, c, d) -> int:
    if a > 500 or b > 500:
        reducer = min((500 / max(a, 1)), (500 / max(b, 1)))  # use max in case one of the values is 0
        a = int(a * reducer)
        b = int(b * reducer)
    if c > 500 or d > 500:
        reducer = min((500 / max(c, 1)), (500 / max(d, 1)))
        c = int(c * reducer)
        d = int(d * reducer)
    return (a << (MULTIPLICATION_SHIFTS * 3)) + (b << (MULTIPLICATION_SHIFTS * 2)) + (c << (MULTIPLICATION_SHIFTS * 1)) + d


def fisher_significant(a, b, c, d) -> bool:
    offset = fisher_offset(a, b, c, d)
    bin_offset = offset // 8
    return table_mmap[bin_offset] & (1 << (offset % 8)) > 0


class TestRecord(NamedTuple):
    test_id: TestId
    test_name: str
    success_count: int
    failure_count: int
    total_count: int


class Conclusion(Enum):
    MISSING_IN_BASIS = 1
    SIGNIFICANT_IMPROVEMENT = 2
    SIGNIFICANT_REGRESSION = 3


ConclusionMap = Dict[TestId, Conclusion]


def has_regression(records: List[TestRecord], conclusions: ConclusionMap) -> bool:
    """
    :param records: The records to analyze
    :param conclusions: The conclusions for all tests analyzed
    :return: Returns True if even on TestRecord is a regression.
    """
    for record in records:
        conclusion = conclusions.get(record.test_id, None)
        if conclusion == Conclusion.SIGNIFICANT_REGRESSION:
            return True
    return False


class CapabilityRecords(NamedTuple):
    name: CapabilityName
    test_records: Dict[TestId, TestRecord]

    def has_regressed(self, conclusions) -> bool:
        return has_regression(self.test_records.values(), conclusions=conclusions)

    def register_test_record(self, test_record: TestRecord):
        self.test_records[test_record.test_id] = test_record


ByCapability = Dict[CapabilityName, CapabilityRecords]


class ComponentRecords(NamedTuple):
    name: ComponentName
    test_records: Dict[TestId, TestRecord]
    capabilities: ByCapability

    def has_regressed(self, conclusions) -> bool:
        return has_regression(self.test_records.values(), conclusions=conclusions)

    def register_test_record(self, test_record: TestRecord):
        self.test_records[test_record.test_id] = test_record

    def register_test_record_capability(self, capability_name: CapabilityName, test_record: TestRecord):
        if capability_name not in self.capabilities:
            self.capabilities[capability_name] = CapabilityRecords(capability_name, dict())
        self.capabilities[capability_name].test_records[test_record.test_id] = test_record


ById = Dict[TestId, TestRecord]
ByComponent = Dict[ComponentName, ComponentRecords]


def index(request):
    return render(request, "main/index.html")


component_pattern = re.compile(r'\[[^\\]+?\]')


def categorize(rows) -> (ById, ByComponent):
    by_id: ById = dict()
    by_component: ByComponent = dict()

    for row in rows:
        if row['success_count'] is None:
            # Not sure why, but bigquery can return None for some SUMs.
            # Potentially because success_val was not set on all original rows
            continue
        r = TestRecord(
            test_id=row['test_id'],
            test_name=row['test_name'],
            success_count=row['success_count'],
            failure_count=row['total_count'] - row['success_count'],
            total_count=row['total_count'],
        )
        by_id[r.test_id] = r
        labels = re.findall(component_pattern, r.test_name)
        for label in labels:
            if label not in by_component:
                by_component[label] = ComponentRecords(name=label, test_records=dict(), capabilities=dict())
            by_component[label].register_test_record(r)

            capability = r.test_id[:1]
            by_component[label].register_test_record_capability(capability, r)

    return by_id, by_component


def calculate_conclusions(basis_by_id: ById, samples_by_id: ById) -> ConclusionMap:
    by_conclusion: ConclusionMap = dict()
    for test_id, sample_result in samples_by_id.items():
        basis_result = basis_by_id.get(test_id, None)
        if not basis_result or basis_result.total_count == 0:
            by_conclusion[test_id] = Conclusion.MISSING_IN_BASIS
            continue

        basis_pass_percentage = basis_result.success_count / basis_result.total_count
        sample_pass_percentage = sample_result.success_count / sample_result.total_count
        improved = sample_pass_percentage >= basis_pass_percentage

        significant = fisher_significant(
            sample_result.failure_count,
            sample_result.success_count,
            basis_result.failure_count,
            basis_result.success_count,
        )

        if significant:
            by_conclusion[test_id] = Conclusion.SIGNIFICANT_IMPROVEMENT if improved else Conclusion.SIGNIFICANT_REGRESSION

    return by_conclusion


class AllComponentsTable(tables.Table):
    name = tables.Column()
    regression = tables.Column()

    def __init__(self, data, new_key, params):
        super().__init__(data)
        self.params = params
        self.new_key = new_key

    class Meta:
        attrs = {"class": "paleblue"}

    def render_name(self, value):
        params_str = '&'.join([f'{key}={value}' for key, value in self.params.items()])
        return format_html(f'<a href="/main/report?{params_str}&{self.new_key}={value}">{value}</a>')


def report(request):
    basis_release = request.GET['basis_release']
    basis_start_dt = request.GET['basis_start_dt']
    basis_end_dt = request.GET['basis_end_dt']

    sample_release = request.GET['sample_release']
    sample_start_dt = request.GET['sample_start_dt']
    sample_end_dt = request.GET['sample_end_dt']

    target_component_name = request.GET.get('component', None)
    target_capability_name = request.GET.get('capability', None)
    target_test_id = request.GET.get('test_id', None)

    bq = bigquery.Client()
    q = f'''
        SELECT test_id, ANY_VALUE(test_name) as test_name, COUNT(*) as total_count, SUM(success_val) as success_count 
        FROM `openshift-gce-devel.ci_analysis_us.junit` 
        WHERE modified_time >= DATETIME(TIMESTAMP "{basis_start_dt}:00+00") AND modified_time < DATETIME(TIMESTAMP "{basis_end_dt}:00+00") 
              AND branch = "{basis_release}" AND file_path NOT LIKE "%/junit_operator.xml" GROUP BY test_id        
    '''

    basis_rows = bq.query(q)
    basis_by_id, basis_by_component = categorize(basis_rows)

    q = f'''
        SELECT test_id, ANY_VALUE(test_name) as test_name, COUNT(*) as total_count, SUM(success_val) as success_count
        FROM `openshift-gce-devel.ci_analysis_us.junit` 
        WHERE modified_time >= DATETIME(TIMESTAMP "{sample_start_dt}:00+00") AND modified_time < DATETIME(TIMESTAMP "{sample_end_dt}:00+00") 
              AND branch = "{sample_release}" AND file_path NOT LIKE "%/junit_operator.xml" GROUP BY test_id        
    '''

    sample_rows = bq.query(q)
    samples_by_id, samples_by_component = categorize(sample_rows)

    conclusions_by_id = calculate_conclusions(basis_by_id=basis_by_id, samples_by_id=samples_by_id)

    if target_component_name and target_component_name not in samples_by_component:
        return HttpResponse(f'Component not found: {target_component_name}')

    context = {
        'basis_release': basis_release,
        'basis_start_dt': basis_start_dt,
        'basis_end_dt': basis_end_dt,
        'sample_release': sample_release,
        'sample_start_dt': sample_start_dt,
        'sample_end_dt': sample_end_dt,
    }

    if target_test_id:  # Rendering a specifically requested test id
        if target_test_id not in samples_by_id:
            return HttpResponse(f'Capability {target_capability_name} not found in component {target_component_name}')

        sample_test_record = samples_by_id[target_test_id]
        basis_test_record = basis_by_id.get(target_test_id, TestRecord(test_name='', total_count=0, success_count=0, failure_count=0, test_id=target_test_id))
        context['sample_test'] = sample_test_record
        context['basis_test'] = basis_test_record
        context['fishers_exact'] = str(fast_fisher.fast_fisher_cython.fisher_exact(
            sample_test_record.failure_count, sample_test_record.success_count,
            basis_test_record.failure_count, basis_test_record.success_count,
            alternative='greater'
        ))

        return render(request, 'main/report-test.html', context)

    if not target_component_name:  # Rendering all components
        component_summary: List[Dict] = list()
        for component_name in sorted(list(samples_by_component.keys())):
            component_summary.append(
                {
                    'name': component_name,
                    'regression': samples_by_component[component_name].has_regressed(conclusions_by_id)
                }
            )
        table = AllComponentsTable(component_summary,
                                   new_key='component',
                                   params=context)
        context['table'] = table
        return render(request, 'main/report-table.html', context)
    else:  # Rendering all of a specific component's capabilities
        context['component'] = target_component_name
        if not target_capability_name:
            capability_summary: Dict[CapabilityName, bool] = dict()
            component_records = samples_by_component[target_component_name]
            for capability_name in sorted(component_records.capabilities.keys()):
                capability_summary[capability_name] = component_records.capabilities[capability_name].has_regressed(conclusions_by_id)

            context['summary'] = capability_summary
            return render(request, 'main/report-capabilities.html', context)
        else:  # Rendering the capabilities of a specific component
            test_summary: Dict[CapabilityName, (TestId, bool)] = dict()
            context['capability'] = target_capability_name
            component_records = samples_by_component[target_component_name]
            if target_capability_name not in component_records.capabilities:
                return HttpResponse(f'Capability {target_capability_name} not found in component {target_component_name}')
            capability_records = component_records.capabilities[target_capability_name]
            for tr in sorted(list(capability_records.test_records.values()), key=lambda x: x.test_name):
                test_summary[tr.test_name] = (tr.test_id, has_regression([tr], conclusions_by_id))

            context['summary'] = test_summary
            return render(request, 'main/report-tests.html', context)
